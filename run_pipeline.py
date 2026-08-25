"""
Stage 3 — Run the RAG pipeline and capture the raw material for all 3 signals.

For every QA pair:
  * resolve context (forced_context_id if set, else top-k retrieval)
  * build the fixed prompt template
  * generate ONE greedy answer WITH token log-probabilities   -> signal 1
  * generate FIVE more samples at temperature 0.7             -> signal 2
  * store the context alongside the answer                    -> signal 3

Everything lands in generations.jsonl, one object per line.

WHY OLLAMA (and not llama-cpp-python)
-------------------------------------
Ollama >= v0.12.11 exposes `logprobs` / `top_logprobs`. Since signal 1 IS token
log-probability, that capability decided it. Beyond that:
  * single Windows installer with CUDA bundled; llama-cpp-python needs a CUDA
    compile on Windows, which is a reliable way to lose an afternoon.
  * `keep_alive: 0` force-unloads the model, which is how this pipeline honours
    "one model in VRAM at a time" on a 6 GB card without any manual juggling.
  * llama-cpp-python needs `logits_all=True` for logprobs (slow and memory-hungry
    against Qwen's ~152k vocab) and has open logprob-correctness bugs.

VRAM (RTX 3060, 6 GB)
---------------------
  qwen2.5:7b-instruct-q4_K_M   ~4.7 GB  fits, little headroom. Use --num-ctx 2048.
  qwen2.5:3b-instruct-q4_K_M   ~2.0 GB  comfortable, ~2.5x faster.
  qwen2.5:0.5b                 ~0.4 GB  smoke tests only.

RUN THE PREFLIGHT FIRST. It verifies that logprobs actually come back in a shape
this script understands, which is the one thing that would silently ruin a
multi-hour run:

    python run_pipeline.py --check --model qwen2.5:0.5b

Then:
    python run_pipeline.py --smoke-test 3 --model qwen2.5:0.5b
    python run_pipeline.py --model qwen2.5:7b-instruct-q4_K_M

Resumable: completed qa_ids are skipped, so just re-run after any crash.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import (
    DATA_DIR,
    Timer,
    append_jsonl,
    completed_ids,
    die,
    info,
    log_run_config,
    read_json,
    require_file,
    rule,
    set_seed,
    truncate,
    utc_now,
    warn,
)

# Versioned so a later prompt change is visible in the data rather than being a
# silent confound between runs.
PROMPT_TEMPLATE_VERSION = "v1"
PROMPT_TEMPLATE = "Context: {context}\nQuestion: {question}\nAnswer:"

# The template is a bare completion, so the model will happily invent a follow-up
# "Question:" turn. Cut it off at the first sign of that.
STOP_SEQUENCES = ["\nQuestion:", "\nContext:", "\n\nQuestion", "\n\nContext"]


# =============================================================================
# Backend
# =============================================================================


class OllamaBackend:
    """Thin HTTP client for Ollama's /api/generate."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        keep_alive: str = "5m",
        timeout: int = 600,
        num_ctx: int = 2048,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.num_ctx = num_ctx

        try:
            import requests  # noqa: F401
        except ImportError:
            die("`requests` is not installed.", "pip install -r requirements.txt")

    # --- preflight ---------------------------------------------------------

    def health_check(self) -> list[str]:
        """Confirm the server is up and return the installed model tags."""
        import requests

        try:
            r = requests.get(f"{self.host}/api/tags", timeout=10)
            r.raise_for_status()
        except Exception as e:
            die(
                f"Cannot reach Ollama at {self.host} ({type(e).__name__}).",
                "Start it with `ollama serve`, or install it from "
                "https://ollama.com/download . Check the port with "
                "`ollama list`.",
            )
        return [m["name"] for m in r.json().get("models", [])]

    def ensure_model(self) -> None:
        tags = self.health_check()
        # Ollama reports "qwen2.5:7b"; users often type the same without a tag.
        if self.model in tags:
            return
        base = self.model.split(":")[0]
        if any(t.split(":")[0] == base for t in tags):
            warn(
                f"exact tag '{self.model}' not found, but '{base}' variants exist: "
                f"{[t for t in tags if t.startswith(base)]}"
            )
        die(
            f"Model '{self.model}' is not installed in Ollama.",
            f"Pull it first:  ollama pull {self.model}\n"
            f"        Installed: {', '.join(tags) if tags else '(none)'}",
        )

    # --- generation --------------------------------------------------------

    def generate(
        self,
        prompt: str,
        temperature: float,
        seed: int,
        num_predict: int = 256,
        want_logprobs: bool = False,
    ) -> dict:
        import requests

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "seed": seed,
                "num_predict": num_predict,
                "num_ctx": self.num_ctx,
                "stop": STOP_SEQUENCES,
            },
        }
        if want_logprobs:
            # Top-level per the Ollama API (Logprobs / TopLogprobs). Harmless if
            # an older server ignores them — the preflight is what catches that.
            payload["logprobs"] = True
            payload["top_logprobs"] = 0

        try:
            r = requests.post(
                f"{self.host}/api/generate", json=payload, timeout=self.timeout
            )
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Ollama generate failed: {type(e).__name__}: {e}") from e

        return r.json()

    def unload(self) -> None:
        """Evict the model from VRAM (keep_alive=0). Frees the card for Stage 5."""
        import requests

        try:
            requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": "", "keep_alive": 0},
                timeout=30,
            )
            info(f"unloaded {self.model} from VRAM")
        except Exception as e:
            warn(f"could not unload model: {e}")


def extract_logprobs(resp: dict) -> tuple[list[str], list[float]]:
    """
    Pull (tokens, logprobs) out of a response, tolerating several shapes.

    Ollama's native field and the OpenAI-compatible field have differed across
    versions, and getting this wrong would silently produce empty signal 1 for a
    whole run. So we accept every shape we know of and let the preflight assert
    that one of them actually matched.
    """
    lp = resp.get("logprobs")

    # Shape A — native Ollama: [{"token": "x", "logprob": -0.1}, ...]
    if isinstance(lp, list) and lp and isinstance(lp[0], dict):
        toks = [d.get("token", "") for d in lp]
        vals = [float(d["logprob"]) for d in lp if d.get("logprob") is not None]
        if len(vals) == len(toks):
            return toks, vals

    # Shape B — OpenAI-style: {"content": [{"token": ..., "logprob": ...}, ...]}
    if isinstance(lp, dict) and isinstance(lp.get("content"), list):
        content = lp["content"]
        toks = [d.get("token", "") for d in content]
        vals = [float(d["logprob"]) for d in content if d.get("logprob") is not None]
        if len(vals) == len(toks):
            return toks, vals

    # Shape C — bare list of floats, tokens reported separately.
    if isinstance(lp, list) and lp and isinstance(lp[0], (int, float)):
        vals = [float(x) for x in lp]
        toks = resp.get("tokens") or [""] * len(vals)
        return list(toks)[: len(vals)], vals

    return [], []


# =============================================================================
# Context assembly
# =============================================================================


def render_context(records: list[dict]) -> str:
    """Join retrieved abstracts into the premise text used by the NLI signal."""
    parts = []
    for r in records:
        title = (r.get("title") or "").strip()
        abstract = (r.get("abstract") or "").strip()
        parts.append(f"{title}\n{abstract}" if title else abstract)
    return "\n\n".join(parts)


def resolve_context(row: dict, index, k: int) -> dict:
    """
    Decide which abstract(s) the model sees.

    forced_context_id wins outright — that is the whole point of the
    poorly_supported condition. Otherwise we retrieve normally and record
    whether the source abstract was actually returned.
    """
    forced = (row.get("forced_context_id") or "").strip()
    source_id = row.get("source_arxiv_id", "")

    if forced:
        rec = index.get_by_id(forced)
        if rec is None:
            raise KeyError(
                f"forced_context_id '{forced}' is not in the index "
                f"(qa_id={row['qa_id']})"
            )
        return {
            "context": render_context([rec]),
            "context_ids": [rec["arxiv_id"]],
            "context_scores": [],
            "retrieval_mode": "forced",
            # A forced distractor is by construction not the source abstract.
            "retrieval_hit": rec["arxiv_id"] == source_id,
        }

    hits = index.retrieve(row["question"], k)
    ids = [h["arxiv_id"] for h in hits]
    return {
        "context": render_context(hits),
        "context_ids": ids,
        "context_scores": [round(h["score"], 6) for h in hits],
        "retrieval_mode": "retrieved",
        # Data-quality guard: with a topically tight corpus the retriever can
        # miss the source abstract, which would silently corrupt the positive
        # class. evaluate.py reports the rate.
        "retrieval_hit": source_id in ids,
    }


# =============================================================================
# Preflight
# =============================================================================


def preflight(backend: OllamaBackend) -> None:
    rule("Preflight")
    backend.ensure_model()
    info(f"Ollama reachable at {backend.host}; model '{backend.model}' present")

    info("testing generation + logprob capture …")
    resp = backend.generate(
        "Context: The sky is blue.\nQuestion: What colour is the sky?\nAnswer:",
        temperature=0.0,
        seed=42,
        num_predict=16,
        want_logprobs=True,
    )
    text = (resp.get("response") or "").strip()
    toks, lps = extract_logprobs(resp)

    print(f"\n  response : {text!r}")
    print(f"  tokens   : {len(toks)}")
    print(f"  logprobs : {len(lps)}")
    if lps:
        print(f"  sample   : {[round(x, 3) for x in lps[:8]]}")

    if not text:
        die(
            "Model returned an empty response.",
            "Try `ollama run <model>` manually to confirm the model works.",
        )

    if not lps:
        die(
            "No token log-probabilities came back — signal 1 cannot be computed.",
            "Ollama >= v0.12.11 is required. Check with `ollama --version` and\n"
            "        upgrade from https://ollama.com/download .\n"
            "        Response keys were: " + ", ".join(sorted(resp.keys())),
        )

    if any(x > 0.0 for x in lps):
        warn("some logprobs are positive — that is not a valid log-probability.")

    rule("Preflight OK")
    info("logprobs verified. Safe to start the real run.")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the RAG pipeline and record generations + logprobs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--qa-pairs", default=str(DATA_DIR / "qa_pairs.csv"))
    ap.add_argument("--index-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default=str(DATA_DIR / "generations.jsonl"))
    ap.add_argument("--backend", choices=["ollama"], default="ollama")
    ap.add_argument("--model", default="qwen2.5:7b-instruct-q4_K_M")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--k", type=int, default=1, help="top-k abstracts to retrieve")
    ap.add_argument("--num-predict", type=int, default=256, help="max answer tokens")
    ap.add_argument("--num-ctx", type=int, default=2048, help="context window")
    ap.add_argument("--n-samples", type=int, default=5, help="extra samples")
    ap.add_argument("--sample-temp", type=float, default=0.7)
    ap.add_argument("--keep-alive", default="5m")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--smoke-test", type=int, default=None, metavar="N", help="only run N pairs"
    )
    ap.add_argument(
        "--check", action="store_true", help="run preflight only, then exit"
    )
    ap.add_argument(
        "--unload-when-done",
        action="store_true",
        help="evict the model from VRAM at the end (do this before Stage 5)",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    backend = OllamaBackend(
        model=args.model,
        host=args.host,
        keep_alive=args.keep_alive,
        timeout=args.timeout,
        num_ctx=args.num_ctx,
    )

    if args.check:
        preflight(backend)
        return

    log_run_config("run_pipeline", args, extra={"prompt_template": PROMPT_TEMPLATE})

    # --- inputs -------------------------------------------------------------
    # The csv the team edits IS the input. Reading a derived copy meant a stale
    # mirror could silently discard hours of labelling with no error, so the
    # mirror was removed rather than patched.
    qa_path = Path(args.qa_pairs)
    if not qa_path.exists():
        die(f"QA pairs not found: {qa_path}", "python generate_qa_pairs.py")

    if qa_path.suffix.lower() == ".csv":
        from generate_qa_pairs import read_rows

        rows = read_rows(qa_path)
    else:
        rows = read_json(qa_path)
    if not rows:
        die(f"{qa_path} is empty.", "python generate_qa_pairs.py")

    from index_corpus import CorpusIndex

    require_file(Path(args.index_dir) / "embeddings.npy", "python index_corpus.py")
    index = CorpusIndex.load(args.index_dir)

    preflight(backend)

    # --- resume -------------------------------------------------------------
    done = completed_ids(args.out, "qa_id")
    todo = [r for r in rows if r["qa_id"] not in done]
    if args.smoke_test:
        todo = todo[: args.smoke_test]

    rule("Stage 3: generation")
    info(f"model={args.model}  k={args.k}  samples={args.n_samples}")
    info(f"{len(rows)} QA pairs, {len(done)} already done, {len(todo)} to run")
    if not todo:
        info("nothing to do — every QA pair already has a generation.")
        return

    total_gens = len(todo) * (1 + args.n_samples)
    info(f"{total_gens} generations this run (1 greedy + {args.n_samples} sampled each)")

    try:
        from tqdm import tqdm
    except ImportError:
        die("tqdm is not installed.", "pip install -r requirements.txt")

    n_ok = n_fail = 0
    bar = tqdm(todo, unit="qa", desc="generating")

    for row in bar:
        qa_id = row["qa_id"]
        bar.set_postfix_str(truncate(qa_id, 28))

        try:
            ctx = resolve_context(row, index, args.k)
        except KeyError as e:
            warn(f"{qa_id}: {e} — skipped")
            n_fail += 1
            continue

        prompt = PROMPT_TEMPLATE.format(context=ctx["context"], question=row["question"])

        try:
            # --- greedy pass: the answer we actually evaluate ---------------
            with Timer() as t_greedy:
                resp = backend.generate(
                    prompt,
                    temperature=0.0,
                    seed=args.seed,
                    num_predict=args.num_predict,
                    want_logprobs=True,
                )
            answer = (resp.get("response") or "").strip()
            tokens, logprobs = extract_logprobs(resp)

            # --- sampled passes: raw material for self-consistency ----------
            samples: list[str] = []
            with Timer() as t_samples:
                for i in range(args.n_samples):
                    s = backend.generate(
                        prompt,
                        temperature=args.sample_temp,
                        seed=args.seed + 1 + i,  # reproducible, but distinct
                        num_predict=args.num_predict,
                        want_logprobs=False,
                    )
                    samples.append((s.get("response") or "").strip())

        except RuntimeError as e:
            # Don't write a partial record: leaving the qa_id absent means the
            # next run retries it automatically.
            warn(f"{qa_id}: {e} — will retry on next run")
            n_fail += 1
            continue

        append_jsonl(
            args.out,
            {
                "qa_id": qa_id,
                "question": row["question"],
                "template_type": row.get("template_type", ""),
                "intended_condition": row.get("intended_condition", ""),
                "source_arxiv_id": row.get("source_arxiv_id", ""),
                "forced_context_id": (row.get("forced_context_id") or "").strip(),
                # Carried so evaluate.py can split poorly_supported into hard
                # (similar) vs easy (dissimilar) negatives.
                "distractor_strategy": (row.get("distractor_strategy") or "").strip(),
                "context": ctx["context"],
                "context_ids": ctx["context_ids"],
                "context_scores": ctx["context_scores"],
                "retrieval_mode": ctx["retrieval_mode"],
                "retrieval_hit": ctx["retrieval_hit"],
                "answer": answer,
                "tokens": tokens,
                "token_logprobs": logprobs,
                "samples": samples,
                "model": args.model,
                "backend": backend.name,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "gen_params": {
                    "k": args.k,
                    "greedy_temperature": 0.0,
                    "sample_temperature": args.sample_temp,
                    "n_samples": args.n_samples,
                    "num_predict": args.num_predict,
                    "num_ctx": args.num_ctx,
                    "seed": args.seed,
                },
                "elapsed_ms_greedy": round(t_greedy.ms, 1),
                "elapsed_ms_samples": round(t_samples.ms, 1),
                "elapsed_ms_total": round(t_greedy.ms + t_samples.ms, 1),
                "timestamp_utc": utc_now(),
            },
        )
        n_ok += 1

    bar.close()

    if args.unload_when_done:
        backend.unload()

    rule("Done")
    info(f"{n_ok} written, {n_fail} failed -> {args.out}")
    if n_fail:
        warn(f"{n_fail} pairs failed. Just re-run the same command to retry them.")
    print(
        "\nNext:\n"
        "    python label_ground_truth.py --annotator annotator1\n"
        "    python compute_signals.py        (unload the generator first:\n"
        "                                      run_pipeline.py --unload-when-done)\n"
    )


if __name__ == "__main__":
    main()
