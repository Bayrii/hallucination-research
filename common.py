"""
Shared utilities for the RAG hallucination-detection pipeline.

Every stage imports from here so JSONL handling, seeding, and run-config logging
behave identically across the pipeline. Keeping them in one place means a
crash-resume bug gets fixed once instead of seven times.

Deliberately dependency-light: no torch / transformers / numpy at module scope.
This module is imported by every script, including ones that must start fast.
Heavy imports are done lazily inside the functions that need them.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

# --- Project layout ---------------------------------------------------------
# Resolved relative to this file so scripts work regardless of cwd.

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = DATA_DIR / "results"

# Canonical label vocabulary. Ordered worst -> best so ordinal distance is just
# an index difference, which weighted kappa in compute_agreement.py relies on.
LABELS = ["unsupported", "partially_supported", "supported"]
LABEL_TO_ORDINAL = {name: i for i, name in enumerate(LABELS)}

CONDITIONS = ["well_supported", "partially_supported", "poorly_supported"]

# Abbreviations whose trailing period is not a sentence boundary. Matched with a
# leading \b (see split_sentences) — without it, short entries like "al." and
# "Ms." match inside ordinary words.
ABBREVIATIONS = [
    "e.g.", "i.e.", "et al.", "cf.", "vs.", "etc.", "Fig.", "Eq.",
    "Sec.", "Tab.", "approx.", "resp.", "Dr.", "Prof.", "Mr.", "Ms.",
    "al.", "No.", "Ref.",
]


# --- Console ----------------------------------------------------------------


def info(msg: str) -> None:
    print(f"[info] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr, flush=True)


def die(msg: str, hint: str | None = None) -> None:
    """
    Fail loudly and early with something actionable.

    The whole pipeline is authored on one machine and run on another, so a
    readable message here saves a debugging round-trip later. Always prefer
    this over letting a missing file surface as a stack trace six frames deep.
    """
    print(f"\n[FATAL] {msg}", file=sys.stderr)
    if hint:
        print(f"[hint]  {hint}", file=sys.stderr)
    print(file=sys.stderr)
    sys.exit(1)


def rule(title: str = "", width: int = 78) -> None:
    if title:
        pad = max(0, width - len(title) - 3)
        print(f"\n== {title} " + "=" * pad, flush=True)
    else:
        print("=" * width, flush=True)


def wrap(text: str, width: int = 78, indent: str = "") -> str:
    """Wrap text for terminal display, preserving paragraph breaks."""
    paragraphs = str(text).split("\n")
    out = []
    for p in paragraphs:
        if not p.strip():
            out.append("")
            continue
        out.append(
            textwrap.fill(
                p, width=width, initial_indent=indent, subsequent_indent=indent
            )
        )
    return "\n".join(out)


# --- Filesystem guards ------------------------------------------------------


def require_file(path: str | Path, produced_by: str) -> Path:
    """
    Assert an input file exists, naming the stage that produces it.

    `produced_by` should be a runnable command, not a description — the person
    hitting this error wants to know exactly what to type next.
    """
    p = Path(path)
    if not p.exists():
        die(
            f"Required input not found: {p}",
            f"This file is produced by:  {produced_by}",
        )
    if p.stat().st_size == 0:
        die(
            f"Required input is empty: {p}",
            f"Re-run:  {produced_by}",
        )
    return p


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --- JSONL ------------------------------------------------------------------
#
# JSONL is used for every incremental output (generations, labels, signals) so
# that a crash costs one line, not the whole run. Readers below tolerate a
# truncated final line for exactly that reason.


def read_jsonl(path: str | Path, quiet: bool = False) -> list[dict[str, Any]]:
    """
    Read a JSONL file into a list of dicts.

    A partially-written final line (killed mid-write) is skipped with a warning
    rather than raising. Any *interior* malformed line is a real corruption and
    is reported, but still skipped so a single bad row can't block a resume.
    """
    p = Path(path)
    if not p.exists():
        return []

    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            is_last = i == len(lines) - 1
            if is_last:
                if not quiet:
                    warn(
                        f"{p.name}: final line is truncated (interrupted write) "
                        f"— skipping it. This is expected after a crash."
                    )
            else:
                warn(f"{p.name}: line {i + 1} is malformed JSON — skipping.")
    return rows


def append_jsonl(path: str | Path, obj: dict[str, Any]) -> None:
    """
    Append one record and force it to disk.

    flush + fsync on every record is deliberate: these runs take hours and are
    resumed after crashes, so losing the OS buffer would lose real GPU time.
    At ~150-900 records the cost is irrelevant.
    """
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write (overwrite) a whole JSONL file at once."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def completed_ids(path: str | Path, key: str = "qa_id") -> set[str]:
    """
    IDs already present in a JSONL file — the basis of every resume in this repo.

    Only rows that parse AND carry `key` count as complete, so a truncated tail
    is naturally re-done on the next run.
    """
    return {r[key] for r in read_jsonl(path, quiet=True) if key in r}


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


# --- Reproducibility --------------------------------------------------------


def set_seed(seed: int) -> None:
    """
    Seed every RNG that can affect this pipeline.

    torch is imported lazily so that stages which never touch it (corpus build,
    QA generation, agreement) don't pay a ~5s import.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_run_config(stage: str, args: Any, extra: dict | None = None) -> Path:
    """
    Persist the resolved config of a run to runs/<timestamp>_<stage>.json.

    Written at start-up, not on success, so a crashed run still leaves evidence
    of what it was trying to do.
    """
    ensure_dir(RUNS_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RUNS_DIR / f"{ts}_{stage}.json"

    payload = {
        "stage": stage,
        "timestamp_utc": utc_now(),
        "args": {k: _jsonable(v) for k, v in vars(args).items()}
        if hasattr(args, "__dict__")
        else _jsonable(args),
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    if extra:
        payload.update({k: _jsonable(v) for k, v in extra.items()})

    write_json(path, payload)
    return path


def _jsonable(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


# --- Timing -----------------------------------------------------------------


class Timer:
    """
    Wall-clock timer for the accuracy-vs-cost analysis.

    Latency is a first-class result in this paper, not incidental logging, so
    signal timings are measured the same way everywhere.

        with Timer() as t:
            ...
        print(t.ms)
    """

    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.ms = (time.perf_counter() - self.t0) * 1000.0

    @property
    def seconds(self) -> float:
        return self.ms / 1000.0


def timed(fn: Callable, *a: Any, **kw: Any) -> tuple[Any, float]:
    """Call fn, return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, (time.perf_counter() - t0) * 1000.0


# --- Text ------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """
    Lightweight sentence splitter for the NLI chunking in compute_signals.py.

    Regex rather than nltk/spacy on purpose: one less model download and one
    less dependency on the offline machine. It handles the common abbreviations
    that appear in arXiv abstracts (e.g., i.e., et al., Fig., vs.), which is
    the failure mode that actually matters here — a bad split silently corrupts
    the entailment matrix.
    """
    import re

    if not text or not text.strip():
        return []

    protected = text
    for i, ab in enumerate(ABBREVIATIONS):
        # The \b is load-bearing. A plain str.replace of "al." also matches
        # inside "retrieval.", "empirical.", "several." — and "Ms." inside
        # "algorithms." — which deletes the sentence boundary and silently
        # merges the whole abstract into one premise.
        protected = re.sub(
            r"\b" + re.escape(ab),
            lambda _m, i=i: f"\x00{i}\x00",
            protected,
        )

    # Mask decimals so "95.4%" does not split. A lambda, not a replacement
    # template — re parses "\x01" in a template as an escape and rejects it.
    protected = re.sub(
        r"(\d)\.(\d)", lambda m: m.group(1) + "\x01" + m.group(2), protected
    )

    parts = re.split(r"(?<=[.!?])\s+", protected)

    out = []
    for p in parts:
        for i, ab in enumerate(ABBREVIATIONS):
            p = p.replace(f"\x00{i}\x00", ab)
        p = p.replace("\x01", ".")
        p = p.strip()
        if p:
            out.append(p)
    return out


def truncate(text: str, n: int = 200) -> str:
    text = str(text).replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def normalize_arxiv_id(arxiv_id: str) -> str:
    """
    Strip the version suffix: '2401.12345v3' -> '2401.12345'.

    arXiv returns revisions as separate search hits, so deduping on the raw ID
    lets the same paper into the corpus twice under different versions.
    """
    import re

    return re.sub(r"v\d+$", "", str(arxiv_id).strip())
