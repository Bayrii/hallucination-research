"""
Stage 0 — Build the arXiv abstract corpus.

Revised from the original single-query draft. Three changes that matter:

  1. MULTIPLE RELATED QUERIES, round-robin merged. A single narrow query returns
     50 near-identical papers, which makes every "irrelevant" distractor a hard
     negative and leaves the retriever unable to separate anything. Several
     related queries give a controllable spread of negative difficulty.

  2. DEDUPE ON THE BASE arXiv ID, not just the lowercased title. arXiv returns
     revisions (v1/v2) as distinct hits, and a retitled revision slips past a
     title-only check.

  3. RECORD THE QUERIES AND FETCH DATE to data/corpus_meta.json. arXiv relevance
     ranking drifts over time, so without this the corpus cannot be rebuilt and
     the experiment is not reproducible.

Runtime: ~1-2 minutes (arXiv asks for a 3s delay between pages).

Usage:
    python build_corpus.py
    python build_corpus.py --target 50 --per-query 30 --out data/corpus.json
"""

from __future__ import annotations

import argparse
import re

from common import (
    DATA_DIR,
    Timer,
    die,
    info,
    log_run_config,
    normalize_arxiv_id,
    rule,
    set_seed,
    truncate,
    utc_now,
    warn,
    write_json,
)

# --- Query set --------------------------------------------------------------
#
# Six related queries spanning the subfield. All cs.CL so everything stays in
# domain, but each targets a different facet — this is what gives distractor
# abstracts a *range* of difficulty instead of all being near-duplicates.
#
# Edit freely; whatever you use is recorded in corpus_meta.json.

SEARCH_QUERIES: list[str] = [
    # 1. Core: hallucination specifically in retrieval-augmented settings.
    'cat:cs.CL AND abs:"hallucination" AND '
    '(abs:"retrieval-augmented" OR abs:"RAG" OR abs:"retrieval augmented")',

    # 2. Faithfulness / groundedness of generated text against a source.
    # NB: bare abs:"grounded" was removed after testing — it matched "visually
    # grounded", "grounded theory" and "grounded language learning", inflating
    # this query from ~1.1k to ~5.6k hits and pushing off-topic clinical
    # summarization papers into the top results.
    'cat:cs.CL AND (abs:"faithfulness" OR abs:"groundedness") '
    'AND abs:"generation"',

    # 3. Evaluation frameworks and benchmarks for RAG systems.
    'cat:cs.CL AND abs:"retrieval-augmented generation" AND '
    '(abs:"evaluation" OR abs:"benchmark")',

    # 4. Factual consistency detection (the NLI-adjacent literature).
    'cat:cs.CL AND (abs:"factual consistency" OR abs:"factuality") AND '
    'abs:"language model"',

    # 5. Attribution / citation / verifiability.
    'cat:cs.CL AND (abs:"attribution" OR abs:"citation") AND '
    'abs:"language model"',

    # 6. Uncertainty & confidence estimation — the signals side of our paper.
    'cat:cs.CL AND (abs:"uncertainty estimation" OR abs:"confidence estimation" '
    'OR abs:"calibration") AND abs:"language model"',
]


# Surveys break the "What method does this paper propose?" template: they don't
# propose one, so the model hedges or invents for reasons unrelated to retrieval
# quality. That contaminates intended_condition, which the whole per-condition
# analysis rests on. Filtered out by default; --keep-surveys overrides.
SURVEY_TITLE = re.compile(
    r"\b(survey|systematic review|literature review|a review of|an overview of"
    r"|tutorial|position paper)\b",
    re.I,
)
SURVEY_ABSTRACT = re.compile(r"\b(this (survey|review)|we (survey|review))\b", re.I)


def is_survey(rec: dict) -> bool:
    if SURVEY_TITLE.search(rec.get("title", "")):
        return True
    # Surveys announce themselves early: "In this survey, we ...".
    return bool(SURVEY_ABSTRACT.search(rec.get("abstract", "")[:400]))


def fetch_query(client, query: str, max_results: int) -> list[dict]:
    """Run one arXiv query and normalize the results into our record shape."""
    import arxiv

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    out: list[dict] = []
    try:
        for r in client.results(search):
            out.append(
                {
                    "arxiv_id": normalize_arxiv_id(r.get_short_id()),
                    "arxiv_id_versioned": r.get_short_id(),
                    "title": r.title.strip().replace("\n", " "),
                    "abstract": r.summary.strip().replace("\n", " "),
                    "authors": [a.name for a in r.authors],
                    "published": r.published.strftime("%Y-%m-%d"),
                    "primary_category": r.primary_category,
                    "url": r.entry_id,
                    "source_query": query,
                }
            )
    except Exception as e:  # network, parse, or rate-limit failures
        warn(f"query failed ({type(e).__name__}: {e}) — continuing with others")
    return out


def round_robin_merge(
    per_query: list[list[dict]],
    target: int,
    min_abstract_chars: int,
    skip_surveys: bool = True,
) -> tuple[list[dict], dict[str, int]]:
    """
    Interleave results so no single query dominates the corpus.

    Taking the top-N of a concatenated list would fill the corpus from whichever
    query happened to return the most hits, which is exactly the topic imbalance
    the multi-query design is meant to avoid.
    """
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    merged: list[dict] = []
    skipped = {"duplicate": 0, "too_short": 0, "survey": 0}

    depth = 0
    max_depth = max((len(q) for q in per_query), default=0)

    while len(merged) < target and depth < max_depth:
        for results in per_query:
            if depth >= len(results):
                continue
            rec = results[depth]

            if rec["arxiv_id"] in seen_ids or rec["title"].lower().strip() in seen_titles:
                skipped["duplicate"] += 1
                continue
            # Too thin to support three distinct questions — a stub abstract
            # makes "what dataset?" unanswerable for reasons unrelated to
            # retrieval, which is the same confound as the survey case.
            if len(rec["abstract"]) < min_abstract_chars:
                skipped["too_short"] += 1
                continue
            if skip_surveys and is_survey(rec):
                skipped["survey"] += 1
                continue

            seen_ids.add(rec["arxiv_id"])
            seen_titles.add(rec["title"].lower().strip())
            merged.append(rec)

            if len(merged) >= target:
                break
        depth += 1

    return merged, skipped


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fetch an arXiv corpus for the hallucination-detection study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--out", default=str(DATA_DIR / "corpus.json"))
    ap.add_argument("--meta-out", default=str(DATA_DIR / "corpus_meta.json"))
    ap.add_argument("--target", type=int, default=50, help="final corpus size")
    ap.add_argument(
        "--per-query", type=int, default=30, help="candidates pulled per query"
    )
    ap.add_argument(
        "--min-abstract-chars",
        type=int,
        default=700,
        help="skip stub abstracts too short to support 3 QA pairs",
    )
    ap.add_argument(
        "--keep-surveys",
        action="store_true",
        help="keep survey/review papers (they break the 'what method?' template)",
    )
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between pages")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    log_run_config("build_corpus", args, extra={"queries": SEARCH_QUERIES})

    try:
        import arxiv
    except ImportError:
        die(
            "The `arxiv` package is not installed.",
            "pip install -r requirements.txt",
        )

    rule("Stage 0: build corpus")
    info(f"{len(SEARCH_QUERIES)} queries x {args.per_query} candidates each")

    client = arxiv.Client(page_size=50, delay_seconds=args.delay, num_retries=3)

    per_query: list[list[dict]] = []
    with Timer() as t:
        for i, q in enumerate(SEARCH_QUERIES, 1):
            info(f"[{i}/{len(SEARCH_QUERIES)}] {truncate(q, 90)}")
            results = fetch_query(client, q, args.per_query)
            info(f"    -> {len(results)} results")
            per_query.append(results)

    total_raw = sum(len(r) for r in per_query)
    if total_raw == 0:
        die(
            "Every arXiv query returned zero results.",
            "Check your network connection, then re-run. If arXiv is reachable, "
            "the query syntax may have been edited into something invalid.",
        )

    corpus, skipped = round_robin_merge(
        per_query,
        args.target,
        args.min_abstract_chars,
        skip_surveys=not args.keep_surveys,
    )

    info(
        f"filtered out: {skipped['duplicate']} duplicate, "
        f"{skipped['too_short']} too short (<{args.min_abstract_chars} chars), "
        f"{skipped['survey']} survey/review"
    )

    if len(corpus) < args.target:
        warn(
            f"Only {len(corpus)} unique abstracts found (target {args.target}). "
            f"Raise --per-query or add queries to SEARCH_QUERIES."
        )

    write_json(args.out, corpus)
    write_json(
        args.meta_out,
        {
            "fetched_utc": utc_now(),
            "queries": SEARCH_QUERIES,
            "per_query_requested": args.per_query,
            "per_query_returned": [len(r) for r in per_query],
            "target_size": args.target,
            "final_size": len(corpus),
            "min_abstract_chars": args.min_abstract_chars,
            "surveys_excluded": not args.keep_surveys,
            "filtered_counts": skipped,
            "seed": args.seed,
            "note": (
                "arXiv relevance ranking drifts over time. Rebuilding with these "
                "queries at a later date will NOT reproduce this exact corpus. "
                "Treat corpus.json itself as the artifact of record."
            ),
        },
    )

    # Per-query contribution: a badly skewed split means one query is doing all
    # the work and the corpus is narrower than it looks.
    rule("Composition")
    counts: dict[str, int] = {}
    for rec in corpus:
        counts[rec["source_query"]] = counts.get(rec["source_query"], 0) + 1
    for q, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>3}  {truncate(q, 68)}")

    rule("Done")
    info(f"{len(corpus)} abstracts -> {args.out}   ({t.seconds:.1f}s)")
    info(f"provenance -> {args.meta_out}")
    print(
        "\nNext: read through corpus.json as a team and delete any off-topic or\n"
        "near-duplicate entries by hand. Then run:\n\n"
        "    python index_corpus.py\n"
    )


if __name__ == "__main__":
    main()
