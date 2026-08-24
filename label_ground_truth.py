"""
Stage 4 — Human ground-truth labelling.

Shows one generation at a time (question, context, answer) and records a label.
Run it once per annotator:

    python label_ground_truth.py --annotator annotator1
    python label_ground_truth.py --annotator annotator2

-> data/labels_annotator1.jsonl, data/labels_annotator2.jsonl

BLIND BY DEFAULT
----------------
`intended_condition` is HIDDEN. If an annotator can see that an item was built
as a `poorly_supported` case, they will label it unsupported — and the ground
truth stops being independent of the experimental manipulation, which is the
one thing it has to be. `--show-condition` exists for spot-checking only; never
use it for a real labelling pass.

Order is shuffled per-annotator (seeded by annotator name) so the two passes are
not correlated by position, and so fatigue does not land on the same items.

Resumable: already-labelled qa_ids are skipped, so quit and resume freely.

Roughly 20-40s per item, so ~1.5 hours for 150 items per annotator.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from common import (
    DATA_DIR,
    append_jsonl,
    completed_ids,
    die,
    info,
    log_run_config,
    read_jsonl,
    require_file,
    rule,
    truncate,
    utc_now,
    wrap,
    write_jsonl,
)

GUIDELINE = """
LABELS — judge only whether the CONTEXT supports the ANSWER's claims.
Not whether the answer is true in the world. Not whether it is well written.

  s  supported            every claim is in the context, or follows directly
  p  partially_supported  some claims backed, some not; or overreaches slightly
  u  unsupported          central claims absent from, or contradicted by, context

  r  refusal              the model correctly declined ("the context does not
                          discuss X"). Recorded as supported + is_abstention.
                          A refusal asserts nothing ungrounded, so it is NOT a
                          hallucination — evaluate.py can remap these later.

  n  add a note      b  back (redo previous)      k  skip      q  save & quit
  ?  show this help
"""


def clear_screen(enabled: bool) -> None:
    if enabled:
        os.system("cls" if os.name == "nt" else "clear")


def render(item: dict, i: int, total: int, show_condition: bool, width: int) -> None:
    rule(f"Item {i}/{total}   [{truncate(item['qa_id'], 40)}]")
    print("\nQUESTION")
    print(wrap(item.get("question", ""), width, "  "))

    print("\nCONTEXT")
    print(wrap(item.get("context", ""), width, "  "))

    print("\nANSWER")
    print(wrap(item.get("answer", "") or "(empty)", width, "  "))

    if show_condition:
        print(f"\n  [!] intended_condition = {item.get('intended_condition')}")
        print(f"  [!] retrieval_hit      = {item.get('retrieval_hit')}")
    print()


def prompt_label(width: int) -> tuple[str, bool, str] | str:
    """
    Returns (label, is_abstention, note), or a control string: back/skip/quit.
    """
    note = ""
    while True:
        try:
            raw = input("  label [s/p/u/r] (n=note, b=back, k=skip, q=quit, ?=help): ")
        except (EOFError, KeyboardInterrupt):
            print()
            return "quit"

        c = raw.strip().lower()
        if c in ("?", "h", "help"):
            print(GUIDELINE)
            continue
        if c == "n":
            note = input("  note: ").strip()
            continue
        if c in ("b", "back"):
            return "back"
        if c in ("k", "skip"):
            return "skip"
        if c in ("q", "quit"):
            return "quit"
        if c in ("s", "supported"):
            return ("supported", False, note)
        if c in ("p", "partial", "partially_supported"):
            return ("partially_supported", False, note)
        if c in ("u", "unsupported"):
            return ("unsupported", False, note)
        if c in ("r", "refusal", "abstain"):
            # Label + flag are stored separately so evaluate.py's --abstention
            # switch can move these between classes without re-annotating.
            return ("supported", True, note)
        print("  ? unrecognised — press ? for help")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Label generations as supported / partially / unsupported.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--generations", default=str(DATA_DIR / "generations.jsonl"))
    ap.add_argument("--annotator", required=True, help="e.g. annotator1")
    ap.add_argument("--out", default=None, help="default: data/labels_<annotator>.jsonl")
    ap.add_argument("--width", type=int, default=78)
    ap.add_argument("--limit", type=int, default=None, help="label at most N items")
    ap.add_argument(
        "--no-shuffle", action="store_true", help="present in file order instead"
    )
    ap.add_argument(
        "--show-condition",
        action="store_true",
        help="SPOT-CHECKING ONLY — biases labels, never use for a real pass",
    )
    ap.add_argument("--no-clear", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else DATA_DIR / f"labels_{args.annotator}.jsonl"
    log_run_config("label_ground_truth", args)
    require_file(args.generations, "python run_pipeline.py")

    gens = read_jsonl(args.generations)
    if not gens:
        die("No generations to label.", "python run_pipeline.py")

    if args.show_condition:
        print(
            "\n  *** --show-condition is ON. Labels from this pass are NOT blind\n"
            "      and must not be used as ground truth. ***\n"
        )

    done = completed_ids(out_path, "qa_id")
    todo = [g for g in gens if g["qa_id"] not in done]

    if not args.no_shuffle:
        # Seeded by annotator name: reproducible, but different per annotator.
        random.Random(args.annotator).shuffle(todo)
    if args.limit:
        todo = todo[: args.limit]

    rule(f"Stage 4: labelling as '{args.annotator}'")
    info(f"{len(gens)} generations, {len(done)} already labelled, {len(todo)} to go")
    if not todo:
        info("nothing left to label.")
        return
    print(GUIDELINE)
    input("  press Enter to begin … ")

    clear = not args.no_clear
    written: list[str] = []  # qa_ids written this session, for `back`
    i = 0

    while i < len(todo):
        item = todo[i]
        clear_screen(clear)
        render(item, i + 1, len(todo), args.show_condition, args.width)

        result = prompt_label(args.width)

        if result == "quit":
            break

        if result == "skip":
            i += 1
            continue

        if result == "back":
            if not written:
                print("  (nothing to go back to)")
                input("  press Enter … ")
                continue
            # Rewrite the file without the last record, then re-present it.
            last_id = written.pop()
            rows = [r for r in read_jsonl(out_path) if r["qa_id"] != last_id]
            write_jsonl(out_path, rows)
            i = max(0, i - 1)
            info(f"removed label for {last_id} — re-labelling it")
            continue

        label, is_abstention, note = result
        append_jsonl(
            out_path,
            {
                "qa_id": item["qa_id"],
                "annotator": args.annotator,
                "label": label,
                "is_abstention": is_abstention,
                "note": note,
                "blind": not args.show_condition,
                "timestamp_utc": utc_now(),
            },
        )
        written.append(item["qa_id"])
        i += 1

    total_done = len(completed_ids(out_path, "qa_id"))
    rule("Saved")
    info(f"{total_done}/{len(gens)} labelled -> {out_path}")
    if total_done < len(gens):
        info("re-run the same command to continue where you left off.")
    else:
        print(
            "\nOnce BOTH annotators are done:\n\n"
            "    python compute_agreement.py\n"
        )


if __name__ == "__main__":
    main()
