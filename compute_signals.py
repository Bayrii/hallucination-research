"""
Stage 5 — Compute the three detection signals (plus a free baseline).

Reads generations.jsonl, writes signals.jsonl keyed by the same qa_id.

SIGNALS
-------
1. LOG-PROB       mean / min / std token log-prob, plus answer token count.
                  The token count is not decoration: mean log-prob correlates
                  with length, so without the covariate you cannot tell a
                  confidence effect from a length effect.

2. SELF-CONSISTENCY over the 5 sampled generations. Two methods, --consistency-method:

     nli        mutual entailment across all 10 sample pairs (20 directed).
                Catches contradiction and paraphrase. ~20 NLI calls per item.
     embedding  mean pairwise cosine of MiniLM embeddings. Much faster (the
                actual ratio lands in the timing table below — don't quote a
                figure you haven't measured), but blind to negation: "X improves
                Y" and "X does not improve Y" embed almost identically. Good for
                iterating; report the NLI variant.
     both       computes each; lets you show they agree (or don't).

3. NLI ENTAILMENT between answer and context, with SENTENCE-LEVEL CHUNKING.
   These cross-encoders are trained on short sentence pairs; feeding a whole
   200-word abstract as premise measurably degrades them. So we build an
   entailment matrix over (context sentence x answer claim) and aggregate:

       per claim : max over context sentences  (is it supported ANYWHERE?)
       overall   : min over claims             (strict — weakest claim rules)
                   mean over claims            (lenient variant, also stored)

   This is the SummaC/FactCC-style aggregation and is the main reason to expect
   this signal to beat a naive whole-abstract entailment score.

4. BASELINE (free): answer-context embedding cosine + ROUGE-L overlap. If this
   matches the NLI signal, that is a finding worth reporting, not an
   embarrassment — it would mean you need nothing more than string overlap.

VRAM / ORDER
------------
Models are loaded ONE AT A TIME and explicitly freed between phases.
UNLOAD THE GENERATOR FIRST or you will OOM on a 6 GB card:

    python run_pipeline.py --unload-when-done --check   # or: ollama stop <model>

    all-MiniLM-L6-v2            ~0.4 GB
    nli-deberta-v3-base         ~0.8 GB (fp32)

Runtime: ~20-40 min on CPU for 150 items with --consistency-method nli;
~5 min with embedding. Both are fine on the 3060.

Usage:
    python compute_signals.py
    python compute_signals.py --consistency-method embedding --device cuda
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from common import (
    DATA_DIR,
    Timer,
    die,
    info,
    log_run_config,
    read_jsonl,
    require_file,
    rule,
    set_seed,
    split_sentences,
    utc_now,
    warn,
    write_jsonl,
)

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =============================================================================
# Signal 1 — token log-probability
# =============================================================================


def logprob_signals(gen: dict) -> dict:
    """Cheapest signal in the study: pure arithmetic on data Stage 3 already saved."""
    import numpy as np

    lps = [float(x) for x in (gen.get("token_logprobs") or [])]
    if not lps:
        return {
            "logprob_mean": None,
            "logprob_min": None,
            "logprob_std": None,
            "logprob_sum": None,
            "answer_n_tokens": 0,
        }
    a = np.array(lps, dtype=float)
    return {
        "logprob_mean": float(a.mean()),
        "logprob_min": float(a.min()),
        "logprob_std": float(a.std()),
        "logprob_sum": float(a.sum()),
        # Length covariate — see module docstring.
        "answer_n_tokens": int(a.size),
    }


# =============================================================================
# NLI plumbing
# =============================================================================


def load_nli(device: str, model_name: str = NLI_MODEL):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        die("sentence-transformers is not installed.", "pip install -r requirements.txt")

    info(f"loading NLI model {model_name} on {device}")
    try:
        return CrossEncoder(model_name, device=device, max_length=256)
    except Exception as e:
        die(
            f"Could not load {model_name}: {type(e).__name__}: {e}",
            "DeBERTa-v3 needs a sentencepiece tokenizer:\n"
            "        pip install sentencepiece protobuf\n"
            "        Also check you have network access for the first download.",
        )


def entailment_index(model) -> int:
    """
    Which output column means "entailment".

    cross-encoder/nli-deberta-v3-base uses ['contradiction', 'entailment',
    'neutral'] — NOT the usual MNLI order. Guessing wrong silently inverts the
    whole signal, so read it from the config and only then fall back.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        cfg = getattr(getattr(model, "model", None), "config", None)
    id2label = getattr(cfg, "id2label", None) or {}
    for i, name in id2label.items():
        if "entail" in str(name).lower():
            return int(i)
    warn("could not read id2label from NLI model — assuming index 1 (documented default)")
    return 1


def nli_entail_probs(model, pairs: list[tuple[str, str]], batch_size: int, ent_idx: int):
    """Entailment probability for each (premise, hypothesis) pair."""
    import numpy as np

    if not pairs:
        return np.array([], dtype=float)

    scores = model.predict(pairs, batch_size=batch_size, convert_to_numpy=True)
    scores = np.asarray(scores, dtype=float)

    if scores.ndim == 1:  # binary/regression head
        return scores

    # predict() returns raw logits on some versions and probabilities on others.
    # Softmax only if the rows are not already normalized.
    row_sums = scores.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        scores = e / e.sum(axis=1, keepdims=True)

    return scores[:, ent_idx]


def grounding_score(
    model, context: str, answer: str, batch_size: int, ent_idx: int
) -> dict:
    """
    Sentence-chunked entailment of `answer` against `context`.

    Returns the strict (min-over-claims) and lenient (mean-over-claims) variants
    plus the naive whole-passage score, so the chunking decision can be
    justified with numbers rather than asserted.
    """
    import numpy as np

    premises = split_sentences(context)
    claims = split_sentences(answer)

    if not premises or not claims:
        return {
            "nli_entail_min": None,
            "nli_entail_mean": None,
            "nli_entail_whole": None,
            "nli_n_claims": len(claims),
            "nli_n_premises": len(premises),
        }

    pairs = [(p, c) for c in claims for p in premises]
    probs = nli_entail_probs(model, pairs, batch_size, ent_idx)
    matrix = probs.reshape(len(claims), len(premises))

    per_claim = matrix.max(axis=1)  # best supporting sentence for each claim
    whole = nli_entail_probs(model, [(context, answer)], batch_size, ent_idx)

    return {
        # Strict: one ungrounded claim drags the whole answer down. This is the
        # variant that matches the annotation guideline.
        "nli_entail_min": float(per_claim.min()),
        "nli_entail_mean": float(per_claim.mean()),
        "nli_entail_whole": float(whole[0]) if len(whole) else None,
        "nli_n_claims": int(len(claims)),
        "nli_n_premises": int(len(premises)),
    }


def consistency_nli(model, samples: list[str], batch_size: int, ent_idx: int) -> dict:
    """
    Mutual entailment across sampled generations.

    Whole-text pairs here (not chunked): samples are short, and the question is
    "do these two answers say the same thing", not "which clause is grounded".
    """
    import numpy as np

    valid = [s for s in samples if s and s.strip()]
    if len(valid) < 2:
        return {"consistency_nli": None, "consistency_nli_min": None}

    pairs, index = [], []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            pairs.append((valid[i], valid[j]))
            pairs.append((valid[j], valid[i]))
            index.append((len(pairs) - 2, len(pairs) - 1))

    probs = nli_entail_probs(model, pairs, batch_size, ent_idx)
    # Mutual entailment = both directions hold. Averaging the two directions
    # keeps a one-way entailment (A more specific than B) from reading as full
    # agreement.
    mutual = [float((probs[a] + probs[b]) / 2.0) for a, b in index]
    return {
        "consistency_nli": float(np.mean(mutual)),
        "consistency_nli_min": float(np.min(mutual)),
    }


# =============================================================================
# Embedding-based signals + baseline
# =============================================================================


def consistency_embedding(encoder, samples: list[str]) -> dict:
    import numpy as np

    valid = [s for s in samples if s and s.strip()]
    if len(valid) < 2:
        return {"consistency_embedding": None, "consistency_embedding_min": None}

    emb = encoder.encode(valid, convert_to_numpy=True, normalize_embeddings=True)
    sims = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            sims.append(float(emb[i] @ emb[j]))
    return {
        "consistency_embedding": float(np.mean(sims)),
        "consistency_embedding_min": float(np.min(sims)),
    }


def rouge_l_recall(answer: str, context: str) -> float:
    """
    LCS-based ROUGE-L recall of the answer against the context.

    Hand-rolled to avoid another dependency. Recall-oriented on purpose: we care
    how much of the ANSWER is covered by the context, not the reverse.
    """
    a = answer.lower().split()
    c = context.lower().split()
    if not a or not c:
        return 0.0

    # Rolling two-row LCS: the full table would be ~250x100 per item, which is
    # fine, but two rows keeps it trivial.
    prev = [0] * (len(c) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(c) + 1)
        ai = a[i - 1]
        for j in range(1, len(c) + 1):
            cur[j] = prev[j - 1] + 1 if ai == c[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[len(c)] / len(a)


def baseline_signals(encoder, answer: str, context: str) -> dict:
    if not answer.strip() or not context.strip():
        return {"baseline_cosine": None, "baseline_rouge_l": None}
    emb = encoder.encode(
        [answer, context], convert_to_numpy=True, normalize_embeddings=True
    )
    return {
        "baseline_cosine": float(emb[0] @ emb[1]),
        "baseline_rouge_l": float(rouge_l_recall(answer, context)),
    }


def free_model(obj) -> None:
    """Release a model and its VRAM. Explicit because 6 GB leaves no slack."""
    del obj
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute log-prob, self-consistency and NLI signals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--generations", default=str(DATA_DIR / "generations.jsonl"))
    ap.add_argument("--out", default=str(DATA_DIR / "signals.jsonl"))
    ap.add_argument(
        "--consistency-method",
        choices=["nli", "embedding", "both"],
        default="nli",
        help="see module docstring for the tradeoff",
    )
    ap.add_argument("--nli-model", default=NLI_MODEL)
    ap.add_argument("--embedding-model", default=EMB_MODEL)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip qa_ids already present in signals.jsonl",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    log_run_config("compute_signals", args)
    require_file(args.generations, "python run_pipeline.py")

    if args.device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                warn("CUDA unavailable — using CPU. (Did you install the CPU torch wheel?)")
                args.device = "cpu"
        except ImportError:
            args.device = "cpu"

    gens = read_jsonl(args.generations)
    if not gens:
        die(f"{args.generations} is empty.", "python run_pipeline.py")

    existing: dict[str, dict] = {}
    if args.resume and Path(args.out).exists():
        existing = {r["qa_id"]: r for r in read_jsonl(args.out)}
        gens = [g for g in gens if g["qa_id"] not in existing]
        info(f"resuming: {len(existing)} already scored, {len(gens)} to go")

    if args.limit:
        gens = gens[: args.limit]
    if not gens:
        info("nothing to compute.")
        return

    rule("Stage 5: compute signals")
    info(f"{len(gens)} generations, consistency={args.consistency_method}, device={args.device}")
    warn("make sure the generator is unloaded from VRAM (ollama stop <model>)")

    from tqdm import tqdm

    results: dict[str, dict] = {
        g["qa_id"]: {"qa_id": g["qa_id"], "timing_ms": {}} for g in gens
    }

    # --- Phase 1: log-probs (no model) --------------------------------------
    rule("Phase 1/3 — log-probability")
    n_missing_lp = 0
    for g in tqdm(gens, unit="item", desc="logprob"):
        with Timer() as t:
            sig = logprob_signals(g)
        if sig["answer_n_tokens"] == 0:
            n_missing_lp += 1
        results[g["qa_id"]].update(sig)
        results[g["qa_id"]]["timing_ms"]["logprob"] = round(t.ms, 3)

    if n_missing_lp:
        warn(
            f"{n_missing_lp}/{len(gens)} generations have NO token_logprobs. "
            f"Signal 1 will be null for them. If this is most of the file, "
            f"Stage 3 ran against an Ollama build without logprob support — "
            f"re-run `python run_pipeline.py --check`."
        )

    # --- Phase 2: embedding model -------------------------------------------
    need_emb = args.consistency_method in ("embedding", "both")
    rule("Phase 2/3 — embeddings (baseline" + (" + consistency)" if need_emb else ")"))

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        die("sentence-transformers is not installed.", "pip install -r requirements.txt")

    encoder = SentenceTransformer(args.embedding_model, device=args.device)
    for g in tqdm(gens, unit="item", desc="embedding"):
        r = results[g["qa_id"]]
        with Timer() as t:
            r.update(baseline_signals(encoder, g.get("answer", ""), g.get("context", "")))
        r["timing_ms"]["baseline"] = round(t.ms, 3)

        if need_emb:
            with Timer() as t:
                r.update(consistency_embedding(encoder, g.get("samples", [])))
            r["timing_ms"]["consistency_embedding"] = round(t.ms, 3)

    free_model(encoder)

    # --- Phase 3: NLI model -------------------------------------------------
    rule("Phase 3/3 — NLI entailment")
    nli = load_nli(args.device, args.nli_model)
    ent_idx = entailment_index(nli)
    info(f"entailment class index = {ent_idx}")

    need_nli_cons = args.consistency_method in ("nli", "both")
    for g in tqdm(gens, unit="item", desc="nli"):
        r = results[g["qa_id"]]

        with Timer() as t:
            r.update(
                grounding_score(
                    nli, g.get("context", ""), g.get("answer", ""),
                    args.batch_size, ent_idx,
                )
            )
        r["timing_ms"]["nli_entailment"] = round(t.ms, 3)

        if need_nli_cons:
            with Timer() as t:
                r.update(
                    consistency_nli(nli, g.get("samples", []), args.batch_size, ent_idx)
                )
            r["timing_ms"]["consistency_nli"] = round(t.ms, 3)

    free_model(nli)

    # --- write --------------------------------------------------------------
    for g in gens:
        r = results[g["qa_id"]]
        # Carried through so evaluate.py can break results down without
        # re-opening generations.jsonl.
        r["intended_condition"] = g.get("intended_condition", "")
        r["distractor_strategy"] = g.get("distractor_strategy", "")
        r["retrieval_hit"] = g.get("retrieval_hit")
        r["model"] = g.get("model", "")
        r["consistency_method"] = args.consistency_method
        r["timestamp_utc"] = utc_now()

    merged = {**existing, **results}
    write_jsonl(args.out, list(merged.values()))

    rule("Done")
    info(f"{len(results)} scored ({len(merged)} total) -> {args.out}")
    _timing_summary(list(results.values()))
    print("\nNext:\n    python evaluate.py\n")


def _timing_summary(rows: list[dict]) -> None:
    """Mean cost per signal — this feeds the paper's cost axis directly."""
    import numpy as np

    rule("Mean latency per item (feeds the cost analysis)")
    keys: set[str] = set()
    for r in rows:
        keys.update(r.get("timing_ms", {}).keys())
    for k in sorted(keys):
        vals = [r["timing_ms"][k] for r in rows if k in r.get("timing_ms", {})]
        if vals:
            print(f"  {k:<24} {np.mean(vals):>9.2f} ms")


if __name__ == "__main__":
    main()
