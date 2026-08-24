"""
Stage 3b (optional) — LLM-as-judge baseline.

This is the EXPENSIVE ARM of the accuracy-vs-cost analysis: the thing the three
cheap signals are being compared against. Without it you have costs but no
ceiling to measure them against.

Unlike Stage 3, this is ONE call per QA pair (~150, not 900) and needs no
logprobs — which is exactly why a free tier can carry it.

    --judge gemini    Gemini 2.5 Flash free tier (250 req/day as of 2026-08).
                      150 calls = one sitting. Needs GEMINI_API_KEY.
    --judge ollama    Local Qwen2.5-7B prompted as a grader. No key, no rate
                      limit, fully reproducible, weaker.

Running BOTH turns the cost axis from a single point into a curve:
    NLI cross-encoder (~184M, local)  ->  7B local judge  ->  frontier judge

The judge emits the same three labels as the human annotators, so evaluate.py
can also report judge-vs-human kappa — which tells you whether the judge is a
trustworthy ceiling in the first place, rather than assuming it.

Usage:
    set GEMINI_API_KEY=...            (PowerShell: $env:GEMINI_API_KEY="...")
    python run_judge_baseline.py --judge gemini
    python run_judge_baseline.py --judge ollama --model qwen2.5:7b-instruct-q4_K_M
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

from common import (
    DATA_DIR,
    LABELS,
    Timer,
    append_jsonl,
    completed_ids,
    die,
    info,
    log_run_config,
    read_jsonl,
    require_file,
    rule,
    set_seed,
    truncate,
    utc_now,
    warn,
)

JUDGE_PROMPT_VERSION = "v1"

# Deliberately mirrors the human annotation guideline in label_ground_truth.py.
# If the two drift apart, judge-vs-human kappa stops measuring agreement and
# starts measuring the difference between two rubrics.
JUDGE_PROMPT = """You are evaluating whether an answer is supported by a context passage.

Context:
\"\"\"
{context}
\"\"\"

Question: {question}

Answer: {answer}

Decide how well the CONTEXT supports the ANSWER. Judge only whether the context
backs up the claims in the answer — not whether the answer is true in general,
and not whether it is well written.

Reply with exactly one of these words and nothing else:

supported            every claim in the answer is stated in or directly follows
                     from the context. An answer that correctly says the context
                     does not contain the information also counts as supported.
partially_supported  some claims are backed by the context and some are not.
unsupported          the central claims are absent from or contradicted by the
                     context.

Your answer:"""


# =============================================================================
# Judges
# =============================================================================


class GeminiJudge:
    """
    Gemini via the REST endpoint (no SDK dependency — just `requests`).

    Thinking is disabled by default. On 2.5-series models an enabled thinking
    budget can consume the whole output allowance and return an empty parts
    list, which looks exactly like a broken prompt.
    """

    name = "gemini"

    def __init__(self, model: str, api_key: str, timeout: int = 60) -> None:
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )

    def check(self) -> None:
        info(f"Gemini judge: {self.model} (key ...{self.api_key[-4:]})")

    def judge(self, prompt: str) -> tuple[str, dict]:
        import requests

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 512,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        r = requests.post(
            self.endpoint,
            json=payload,
            timeout=self.timeout,
            headers={"x-goog-api-key": self.api_key},
        )

        if r.status_code == 429:
            raise RateLimited(r.text[:200])
        if r.status_code == 400 and "thinkingConfig" in r.text:
            # Older/non-thinking models reject the field outright.
            payload["generationConfig"].pop("thinkingConfig", None)
            r = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
                headers={"x-goog-api-key": self.api_key},
            )
        r.raise_for_status()
        data = r.json()

        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            finish = (data.get("candidates") or [{}])[0].get("finishReason", "?")
            raise RuntimeError(f"no text in Gemini response (finishReason={finish})")

        usage = data.get("usageMetadata", {})
        return text, {
            "prompt_tokens": usage.get("promptTokenCount"),
            "completion_tokens": usage.get("candidatesTokenCount"),
        }


class OllamaJudge:
    """Local generator reused as a grader. Reuses Stage 3's backend."""

    name = "ollama"

    def __init__(self, model: str, host: str, timeout: int = 300) -> None:
        from run_pipeline import OllamaBackend

        self.backend = OllamaBackend(
            model=model, host=host, keep_alive="5m", timeout=timeout, num_ctx=4096
        )
        self.model = model

    def check(self) -> None:
        self.backend.ensure_model()
        info(f"Ollama judge: {self.model}")

    def judge(self, prompt: str) -> tuple[str, dict]:
        resp = self.backend.generate(
            prompt, temperature=0.0, seed=42, num_predict=16, want_logprobs=False
        )
        return (resp.get("response") or ""), {
            "prompt_tokens": resp.get("prompt_eval_count"),
            "completion_tokens": resp.get("eval_count"),
        }


class RateLimited(Exception):
    pass


def parse_label(text: str) -> str | None:
    """
    Map free text onto the label vocabulary.

    Checks the compound label first: a plain `in` scan would match "supported"
    inside "partially_supported" and silently mislabel every hedged case.
    """
    t = (text or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not t:
        return None
    for label in ("partially_supported", "unsupported", "supported"):
        if label in t:
            return label
    if t.startswith("partial"):
        return "partially_supported"
    return None


def call_with_backoff(judge, prompt: str, max_retries: int, base_delay: float):
    """
    Retry with exponential backoff + jitter.

    Mandatory on a free tier: a 429 partway through 150 calls should cost a
    pause, not the run.
    """
    delay = base_delay
    for attempt in range(max_retries + 1):
        try:
            return judge.judge(prompt)
        except RateLimited:
            if attempt == max_retries:
                raise
            wait = delay + random.uniform(0, delay * 0.3)
            warn(f"rate limited — sleeping {wait:.1f}s (attempt {attempt + 1})")
            time.sleep(wait)
            delay *= 2
        except Exception as e:
            if attempt == max_retries:
                raise
            wait = delay + random.uniform(0, delay * 0.3)
            warn(f"{type(e).__name__}: {e} — retrying in {wait:.1f}s")
            time.sleep(wait)
            delay *= 2
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="LLM-as-judge baseline for the accuracy-vs-cost analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--generations", default=str(DATA_DIR / "generations.jsonl"))
    ap.add_argument("--out", default=None, help="default: data/judge_<judge>.jsonl")
    ap.add_argument("--judge", choices=["gemini", "ollama"], default="gemini")
    ap.add_argument(
        "--model",
        default=None,
        help="default: gemini-2.5-flash / qwen2.5:7b-instruct-q4_K_M",
    )
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--api-key-env", default="GEMINI_API_KEY")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--base-delay", type=float, default=4.0)
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="pause between calls; raise if you keep hitting per-minute limits",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    if args.model is None:
        args.model = (
            "gemini-2.5-flash" if args.judge == "gemini" else "qwen2.5:7b-instruct-q4_K_M"
        )
    out_path = Path(args.out) if args.out else DATA_DIR / f"judge_{args.judge}.jsonl"

    log_run_config("run_judge_baseline", args, extra={"prompt": JUDGE_PROMPT_VERSION})
    require_file(args.generations, "python run_pipeline.py")

    # --- build judge --------------------------------------------------------
    if args.judge == "gemini":
        key = os.environ.get(args.api_key_env, "").strip()
        if not key:
            die(
                f"${args.api_key_env} is not set.",
                'PowerShell:  $env:GEMINI_API_KEY="your-key-here"\n'
                "        Get a free key at https://aistudio.google.com/apikey\n"
                "        Or skip the frontier judge: --judge ollama",
            )
        judge = GeminiJudge(args.model, key)
    else:
        judge = OllamaJudge(args.model, args.host)

    judge.check()

    gens = read_jsonl(args.generations)
    done = completed_ids(out_path, "qa_id")
    todo = [g for g in gens if g["qa_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    rule(f"Stage 3b: {args.judge} judge")
    info(f"{len(gens)} generations, {len(done)} already judged, {len(todo)} to go")

    if args.judge == "gemini" and len(todo) > 240:
        warn(
            f"{len(todo)} calls vs a ~250/day free-tier cap. Expect to finish "
            f"tomorrow — the run is resumable, so just re-run it then."
        )
    if not todo:
        info("nothing to do.")
        return

    from tqdm import tqdm

    n_ok = n_fail = n_unparsed = 0
    for g in tqdm(todo, unit="qa", desc=f"judging ({args.judge})"):
        prompt = JUDGE_PROMPT.format(
            context=g.get("context", ""),
            question=g.get("question", ""),
            answer=g.get("answer", ""),
        )
        try:
            with Timer() as t:
                raw, usage = call_with_backoff(
                    judge, prompt, args.max_retries, args.base_delay
                )
        except Exception as e:
            warn(f"{g['qa_id']}: {type(e).__name__}: {e} — will retry on next run")
            n_fail += 1
            continue

        label = parse_label(raw)
        if label is None:
            n_unparsed += 1
            warn(f"{g['qa_id']}: unparseable verdict {truncate(raw, 60)!r}")

        append_jsonl(
            out_path,
            {
                "qa_id": g["qa_id"],
                "judge": args.judge,
                "judge_model": args.model,
                "label": label,
                "raw_response": raw.strip()[:500],
                "prompt_version": JUDGE_PROMPT_VERSION,
                "elapsed_ms": round(t.ms, 1),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "timestamp_utc": utc_now(),
            },
        )
        n_ok += 1
        if args.sleep:
            time.sleep(args.sleep)

    rule("Done")
    info(f"{n_ok} judged, {n_fail} failed, {n_unparsed} unparseable -> {out_path}")
    if n_fail:
        warn("re-run the same command to retry failures.")
    print(
        "\nevaluate.py picks this file up automatically and scores it as an\n"
        "extra row alongside the three cheap signals.\n"
    )


if __name__ == "__main__":
    main()
