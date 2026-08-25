"""
Stage 2 — Draft candidate QA pairs for manual review.

Produces an EDITABLE csv, not a locked dataset. You are expected to open
data/qa_pairs.csv, fix or delete bad questions, and set `intended_condition`
by hand.

Two modes:

    # 1. draft questions from every abstract
    python generate_qa_pairs.py

    # 2. later, force bad retrieval on a subset (simulates a failed retriever)
    python generate_qa_pairs.py --assign-distractors 50 --strategy similar

Mode 2 reads the csv back, so it never clobbers your manual edits. It writes a
.bak first anyway.

data/qa_pairs.csv is the SINGLE SOURCE OF TRUTH -- run_pipeline.py reads it
directly. An earlier version also wrote a qa_pairs.json mirror and the pipeline
read *that*, so any hand-edit was invisible until the mirror was rebuilt. It
silently discarded review work twice before being removed. Do not reintroduce it.

On distractor strategy
----------------------
Because the corpus is built from several *related* queries, the strategy knob
controls how hard the negatives are, turning a confound into a variable:

    similar     highest-cosine other abstract  -> hard negative. Same subfield,
                plausible-looking context. This is what real retrieval failure
                looks like, and it is where the cheap signals should struggle.
    dissimilar  lowest-cosine other abstract   -> easy negative. Clean signal
                separation; useful as an upper bound.
    random      uniformly sampled other abstract -> mixed difficulty.

Running a mix of `similar` and `dissimilar` gives a difficulty gradient to
report against, which is more informative than one undifferentiated
`poorly_supported` bucket.

Runtime: seconds. The real cost is the manual review afterwards (~1-2 hours).
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

from common import (
    DATA_DIR,
    die,
    info,
    log_run_config,
    read_json,
    require_file,
    rule,
    set_seed,
    truncate,
    warn,
)

# --- Question templates -----------------------------------------------------
#
# Deliberately generic so the same wording applies to every abstract — a fixed
# template set keeps question phrasing from becoming an uncontrolled variable
# across conditions.

TEMPLATES: dict[str, str] = {
    "method": "What method or approach does this paper propose?",
    "dataset": "What dataset(s) or benchmark(s) does this paper use?",
    "finding": "What is the main finding or result reported in this paper?",
    # --- optional extras, enabled with --templates all ---
    #
    # 'limitation' is the useful one: abstracts almost never state limitations,
    # so the model must extrapolate from context that genuinely does not contain
    # the answer. That makes it a natural source of `partially_supported` items
    # WITHOUT having to fake bad retrieval — the retrieval is correct, the
    # context simply underdetermines the answer. Worth including if you want
    # that condition to be well populated.
    "limitation": "What limitations or weaknesses does this paper acknowledge?",
    "metric": "What evaluation metrics does this paper report?",
}

DEFAULT_TEMPLATES = ["method", "dataset", "finding"]

FIELDNAMES = [
    "qa_id",
    "source_arxiv_id",
    "source_title",
    "template_type",
    "question",
    "intended_condition",
    "forced_context_id",
    # Which distractor strategy produced forced_context_id. Carried all the way
    # to evaluate.py so the poorly_supported condition can be split into hard
    # (similar) and easy (dissimilar) negatives — otherwise both collapse into
    # one bucket and the degradation gradient is unrecoverable after Stage 3.
    "distractor_strategy",
    "notes",
]

# Columns that must exist for the file to be usable. Anything else in
# FIELDNAMES is backfilled on read, so a sheet written by an older version of
# this script still loads.
REQUIRED_FIELDNAMES = [
    "qa_id",
    "source_arxiv_id",
    "question",
    "intended_condition",
    "forced_context_id",
]


def make_qa_id(arxiv_id: str, template_type: str) -> str:
    return f"{arxiv_id}__{template_type}"


def draft_pairs(corpus: list[dict], template_keys: list[str]) -> list[dict]:
    rows: list[dict] = []
    for rec in corpus:
        for key in template_keys:
            rows.append(
                {
                    "qa_id": make_qa_id(rec["arxiv_id"], key),
                    "source_arxiv_id": rec["arxiv_id"],
                    "source_title": truncate(rec.get("title", ""), 90),
                    "template_type": key,
                    "question": TEMPLATES[key],
                    # Default assumes correct retrieval. Change by hand during
                    # review, or via --assign-distractors.
                    "intended_condition": "well_supported",
                    "forced_context_id": "",
                    "distractor_strategy": "",
                    "notes": "",
                }
            )
    return rows


# --- csv io -----------------------------------------------------------------


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # The team is told to review this file in a spreadsheet, so extra columns
    # ("reviewer", "comments") are expected. DictWriter raises on any key not in
    # fieldnames, so the header is the union of the canonical fields and
    # whatever else came back — dropping their columns would be worse than the
    # crash it replaces.
    extra = [
        k
        for k in dict.fromkeys(k for r in rows for k in r)
        if k is not None and k not in FIELDNAMES
    ]

    # newline="" is required on Windows or csv writes \r\r\n and Excel shows
    # a blank line between every row.
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES + extra, restval="")
        w.writeheader()
        w.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return rows

    missing = [c for c in REQUIRED_FIELDNAMES if c not in rows[0]]
    if missing:
        die(
            f"{path.name} is missing required column(s): {', '.join(missing)}",
            "Did a spreadsheet editor drop them? Restore from the .bak or re-run "
            "`python generate_qa_pairs.py --force`.",
        )

    # Backfill optional columns so a sheet written before a schema change still
    # works — dying on a newly-added column would throw away manual review work.
    added = [c for c in FIELDNAMES if c not in rows[0]]
    if added:
        info(f"{path.name}: adding new column(s) {', '.join(added)}")
        for r in rows:
            for c in added:
                r.setdefault(c, "")
    return rows


# --- distractor assignment --------------------------------------------------


def assign_distractors(
    rows: list[dict],
    n: int,
    strategy: str,
    index_dir: str,
    seed: int,
    only_condition: str | None = None,
) -> int:
    """
    Point `n` rows at a DIFFERENT abstract, simulating retrieval failure.

    Sets forced_context_id and flips intended_condition to poorly_supported.
    Rows that already carry a forced_context_id are left alone so the function
    is idempotent and safe to re-run.
    """
    rng = random.Random(seed)

    candidates = [
        r
        for r in rows
        if not r.get("forced_context_id", "").strip()
        and (only_condition is None or r.get("intended_condition") == only_condition)
    ]

    # Nothing to do — but the caller still rewrites the csv and json mirror,
    # which is exactly what `--assign-distractors 0` is for. Return before
    # touching the index so refreshing the mirror does not require Stage 1 to
    # have run (or numpy to be installed).
    if n <= 0:
        info("n=0 — refreshing the csv/json mirror only, no distractors assigned.")
        return 0
    if not candidates:
        warn("no eligible rows (all already have forced_context_id).")
        return 0

    if n > len(candidates):
        warn(f"requested {n} but only {len(candidates)} eligible — using all of them.")
        n = len(candidates)

    from index_corpus import CorpusIndex

    idx = CorpusIndex.load(index_dir)  # no encoder load: ID lookups only
    chosen = rng.sample(candidates, n)
    all_ids = [r["arxiv_id"] for r in idx.records]
    changed = 0

    for row in chosen:
        src = row["source_arxiv_id"]
        try:
            if strategy == "random":
                pool = [a for a in all_ids if a != src]
                pick = rng.choice(pool)
            else:
                import numpy as np

                sims = idx.similarity_to_all(src)
                order = np.argsort(-sims) if strategy == "similar" else np.argsort(sims)
                pick = None
                for pos in order:
                    cand = idx.records[int(pos)]["arxiv_id"]
                    if cand != src:
                        pick = cand
                        break
                if pick is None:
                    continue
        except KeyError:
            warn(f"{row['qa_id']}: source {src} not in index — skipped.")
            continue

        row["forced_context_id"] = pick
        row["intended_condition"] = "poorly_supported"
        row["distractor_strategy"] = strategy
        changed += 1

    return changed


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Draft editable QA pairs, and optionally force bad retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--corpus", default=str(DATA_DIR / "corpus.json"))
    ap.add_argument("--out", default=str(DATA_DIR / "qa_pairs.csv"))
    ap.add_argument("--index-dir", default=str(DATA_DIR))
    ap.add_argument(
        "--templates",
        default="default",
        choices=["default", "all"],
        help="'default' = method/dataset/finding; 'all' adds limitation/metric",
    )
    ap.add_argument(
        "--force", action="store_true", help="overwrite an existing qa_pairs.csv"
    )
    ap.add_argument(
        "--assign-distractors",
        type=int,
        metavar="N",
        default=None,
        help="edit-in-place mode: force bad retrieval on N existing rows",
    )
    ap.add_argument(
        "--strategy", choices=["similar", "dissimilar", "random"], default="similar"
    )
    ap.add_argument(
        "--only-condition",
        default="well_supported",
        help=(
            "only convert rows currently in this condition. Defaults to "
            "well_supported so that rows you hand-marked as partially_supported "
            "are never silently overwritten. Pass 'any' to allow all rows."
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    log_run_config("generate_qa_pairs", args)
    out = Path(args.out)

    # --- mode 2: assign distractors to an existing sheet --------------------
    if args.assign_distractors is not None:
        require_file(out, "python generate_qa_pairs.py")
        rows = read_rows(out)

        backup = out.with_suffix(".csv.bak")
        shutil.copy2(out, backup)
        info(f"backed up -> {backup.name}")

        only = None if args.only_condition == "any" else args.only_condition
        rule(f"Assigning {args.assign_distractors} '{args.strategy}' distractors")
        if only is None:
            at_risk = sum(
                1
                for r in rows
                if r.get("intended_condition") not in ("well_supported", "", None)
                and not r.get("forced_context_id", "").strip()
            )
            if at_risk:
                warn(
                    f"--only-condition any: {at_risk} hand-marked row(s) are "
                    f"eligible and may be overwritten to poorly_supported."
                )
        else:
            info(f"only converting rows currently marked '{only}'")

        n = assign_distractors(
            rows,
            args.assign_distractors,
            args.strategy,
            args.index_dir,
            args.seed,
            only,
        )
        write_rows(out, rows)

        info(f"{n} rows now have forced_context_id (condition -> poorly_supported)")
        _summarize(rows)
        return

    # --- mode 1: draft ------------------------------------------------------
    if out.exists() and not args.force:
        die(
            f"{out} already exists — refusing to overwrite your manual edits.",
            "Pass --force to regenerate from scratch, or use "
            "--assign-distractors to edit the existing sheet in place.",
        )

    require_file(args.corpus, "python build_corpus.py")
    corpus = read_json(args.corpus)

    keys = DEFAULT_TEMPLATES if args.templates == "default" else list(TEMPLATES)
    rule("Stage 2: draft QA pairs")
    info(f"{len(corpus)} abstracts x {len(keys)} templates ({', '.join(keys)})")

    rows = draft_pairs(corpus, keys)
    write_rows(out, rows)

    rule("Done")
    info(f"{len(rows)} candidate QA pairs -> {out}")
    _summarize(rows)

    print(
        f"\nNEXT — this stage needs you, not the machine:\n\n"
        f"  1. Open {out} in Excel/LibreOffice.\n"
        f"  2. Delete or rewrite any question that does not fit its abstract.\n"
        f"  3. Set `intended_condition` per row:\n"
        f"       well_supported      answer is fully in the abstract\n"
        f"       partially_supported abstract only hints at the answer\n"
        f"       poorly_supported    answer is absent (usually via a distractor)\n"
        f"  4. For forced bad retrieval, run:\n\n"
        f"       python generate_qa_pairs.py --assign-distractors 50 "
        f"--strategy similar\n\n"
        f"  5. Re-save as CSV, then run:  python run_pipeline.py\n\n"
        f"This csv is the SINGLE SOURCE OF TRUTH. run_pipeline.py reads it\n"
        f"directly, so your edits take effect the moment you save. There is no\n"
        f"sync step to remember.\n"
    )


def _summarize(rows: list[dict]) -> None:
    rule("Condition breakdown")
    counts: dict[str, int] = {}
    for r in rows:
        c = r.get("intended_condition", "?") or "?"
        counts[c] = counts.get(c, 0) + 1
    for c, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {c}")
    forced = sum(1 for r in rows if r.get("forced_context_id", "").strip())
    print(f"  {forced:>4}  rows with a forced_context_id")

    strat: dict[str, int] = {}
    for r in rows:
        s = (r.get("distractor_strategy") or "").strip()
        if s:
            strat[s] = strat.get(s, 0) + 1
    if strat:
        print("\n  distractor strategy (splits the poorly_supported condition):")
        for s, n in sorted(strat.items(), key=lambda kv: -kv[1]):
            hardness = {"similar": "hard negative", "dissimilar": "easy negative",
                        "random": "mixed"}.get(s, "")
            print(f"  {n:>4}  {s:<12} {hardness}")


if __name__ == "__main__":
    main()
