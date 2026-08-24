"""
Stage 4b — Inter-annotator agreement, and consensus label construction.

Computes Cohen's kappa between two annotators and prints every disagreement so
the team can resolve them.

WHY WEIGHTED KAPPA TOO
----------------------
Plain Cohen's kappa is NOMINAL: it treats supported-vs-partially_supported as
exactly as wrong as supported-vs-unsupported. Our labels are ORDINAL
(unsupported < partially_supported < supported), so nominal kappa understates
real agreement — near-misses are penalised like opposites. Report the linear-
weighted figure as the headline and the nominal one for comparability with
papers that only give nominal.

    python compute_agreement.py
    python compute_agreement.py --export-disagreements data/disagreements.csv
    python compute_agreement.py --resolve          # build labels_final.jsonl

--resolve writes labels_final.jsonl: agreed items pass straight through, and it
prompts you for each disagreement. That file is what evaluate.py consumes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import (
    DATA_DIR,
    LABELS,
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
    warn,
    wrap,
)


def load_labels(path: str | Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    out: dict[str, dict] = {}
    for r in rows:
        # Later rows win, so a re-labelled item (via `back`) supersedes.
        out[r["qa_id"]] = r
    return out


def kappa_report(y1: list[str], y2: list[str]) -> dict[str, float]:
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        die("scikit-learn is not installed.", "pip install -r requirements.txt")

    return {
        "nominal": cohen_kappa_score(y1, y2, labels=LABELS),
        "linear": cohen_kappa_score(y1, y2, labels=LABELS, weights="linear"),
        "quadratic": cohen_kappa_score(y1, y2, labels=LABELS, weights="quadratic"),
    }


def interpret(k: float) -> str:
    """Landis & Koch (1977) bands — conventional, and what reviewers expect."""
    if k < 0:
        return "worse than chance"
    if k < 0.21:
        return "slight"
    if k < 0.41:
        return "fair"
    if k < 0.61:
        return "moderate"
    if k < 0.81:
        return "substantial"
    return "almost perfect"


def print_confusion(y1: list[str], y2: list[str], n1: str, n2: str) -> None:
    rule("Confusion matrix")
    counts = {(a, b): 0 for a in LABELS for b in LABELS}
    for a, b in zip(y1, y2):
        counts[(a, b)] += 1

    short = {"unsupported": "unsup", "partially_supported": "part", "supported": "supp"}
    print(f"\n  rows = {n1}, cols = {n2}\n")
    header = " " * 10 + "".join(f"{short[b]:>8}" for b in LABELS)
    print(header)
    for a in LABELS:
        row = f"  {short[a]:<8}" + "".join(f"{counts[(a, b)]:>8}" for b in LABELS)
        print(row)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cohen's kappa between two annotators; optionally resolve.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--a1", default=str(DATA_DIR / "labels_annotator1.jsonl"))
    ap.add_argument("--a2", default=str(DATA_DIR / "labels_annotator2.jsonl"))
    ap.add_argument("--generations", default=str(DATA_DIR / "generations.jsonl"))
    ap.add_argument("--export-disagreements", default=None, metavar="CSV")
    ap.add_argument(
        "--resolve",
        action="store_true",
        help="interactively build data/labels_final.jsonl",
    )
    ap.add_argument("--final-out", default=str(DATA_DIR / "labels_final.jsonl"))
    ap.add_argument("--width", type=int, default=78)
    args = ap.parse_args()

    log_run_config("compute_agreement", args)
    require_file(args.a1, "python label_ground_truth.py --annotator annotator1")
    require_file(args.a2, "python label_ground_truth.py --annotator annotator2")

    l1, l2 = load_labels(args.a1), load_labels(args.a2)
    n1 = Path(args.a1).stem.replace("labels_", "")
    n2 = Path(args.a2).stem.replace("labels_", "")

    shared = sorted(set(l1) & set(l2))
    if not shared:
        die(
            "The two annotators have no qa_ids in common.",
            "Both passes must run over the same generations.jsonl.",
        )

    only1, only2 = set(l1) - set(l2), set(l2) - set(l1)
    if only1 or only2:
        warn(
            f"{len(only1)} item(s) labelled only by {n1}, "
            f"{len(only2)} only by {n2} — kappa uses the {len(shared)} in common."
        )

    y1 = [l1[q]["label"] for q in shared]
    y2 = [l2[q]["label"] for q in shared]

    rule("Stage 4b: inter-annotator agreement")
    info(f"{len(shared)} items labelled by both")

    n_agree = sum(1 for a, b in zip(y1, y2) if a == b)
    ks = kappa_report(y1, y2)

    print(f"\n  raw agreement      : {n_agree}/{len(shared)} "
          f"({100.0 * n_agree / len(shared):.1f}%)")
    print(f"  Cohen's kappa      : {ks['nominal']:.3f}   ({interpret(ks['nominal'])})")
    print(f"  weighted (linear)  : {ks['linear']:.3f}   ({interpret(ks['linear'])})"
          "   <- report this one")
    print(f"  weighted (quadr.)  : {ks['quadratic']:.3f}   ({interpret(ks['quadratic'])})")

    # Abstention flags are a separate judgement and can disagree independently.
    a1_abs = [bool(l1[q].get("is_abstention")) for q in shared]
    a2_abs = [bool(l2[q].get("is_abstention")) for q in shared]
    if any(a1_abs) or any(a2_abs):
        n_abs_agree = sum(1 for a, b in zip(a1_abs, a2_abs) if a == b)
        print(
            f"\n  abstention flag    : {n_abs_agree}/{len(shared)} agree "
            f"({sum(a1_abs)} by {n1}, {sum(a2_abs)} by {n2})"
        )

    print_confusion(y1, y2, n1, n2)

    # --- disagreements ------------------------------------------------------
    disagreements = [q for q in shared if l1[q]["label"] != l2[q]["label"]]
    gens = {g["qa_id"]: g for g in read_jsonl(args.generations)} if Path(args.generations).exists() else {}

    rule(f"Disagreements ({len(disagreements)})")
    for q in disagreements:
        g = gens.get(q, {})
        print(f"\n  {q}")
        print(f"    {n1}: {l1[q]['label']:<20}  {n2}: {l2[q]['label']}")
        if g:
            print(f"    Q: {truncate(g.get('question', ''), 70)}")
            print(f"    A: {truncate(g.get('answer', ''), 70)}")
        for who, lab in ((n1, l1[q]), (n2, l2[q])):
            if lab.get("note"):
                print(f"    note[{who}]: {truncate(lab['note'], 60)}")

    if args.export_disagreements:
        p = Path(args.export_disagreements)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "qa_id", f"label_{n1}", f"label_{n2}", "question", "answer",
                "resolved_label",
            ])
            for q in disagreements:
                g = gens.get(q, {})
                w.writerow([
                    q, l1[q]["label"], l2[q]["label"],
                    g.get("question", ""), g.get("answer", ""), "",
                ])
        info(f"\nexported -> {p}  (fill in resolved_label together)")

    # --- resolve ------------------------------------------------------------
    if args.resolve:
        resolve(shared, l1, l2, gens, n1, n2, args)
    else:
        print(
            f"\nNext: resolve the {len(disagreements)} disagreement(s) together, then\n"
            f"build the consensus file:\n\n"
            f"    python compute_agreement.py --resolve\n"
        )


def resolve(shared, l1, l2, gens, n1, n2, args) -> None:
    """Write labels_final.jsonl: agreements pass through, disagreements prompted."""
    out = Path(args.final_out)
    already = completed_ids(out, "qa_id")

    rule("Building consensus labels")
    if already:
        info(f"{len(already)} already resolved in {out.name} — skipping those")

    n_auto = n_manual = 0
    for q in shared:
        if q in already:
            continue

        lab1, lab2 = l1[q]["label"], l2[q]["label"]
        abst = bool(l1[q].get("is_abstention")) or bool(l2[q].get("is_abstention"))

        if lab1 == lab2:
            final, how = lab1, "agreed"
            n_auto += 1
        else:
            g = gens.get(q, {})
            print("\n" + "-" * 78)
            print(f"  {q}")
            if g:
                print("\n  QUESTION"); print(wrap(g.get("question", ""), args.width, "    "))
                print("\n  CONTEXT");  print(wrap(truncate(g.get("context", ""), 1200), args.width, "    "))
                print("\n  ANSWER");   print(wrap(g.get("answer", ""), args.width, "    "))
            print(f"\n  {n1} said: {lab1}")
            print(f"  {n2} said: {lab2}")

            final = None
            while final is None:
                try:
                    c = input("  consensus [s/p/u] (Enter to skip): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\n  stopping — progress is saved.")
                    return
                if c == "":
                    break
                final = {"s": "supported", "p": "partially_supported",
                         "u": "unsupported"}.get(c)
                if final is None:
                    print("  ? use s, p, or u")
            if final is None:
                continue
            how = "resolved"
            n_manual += 1

        append_jsonl(out, {
            "qa_id": q,
            "label": final,
            "is_abstention": abst,
            "resolution": how,
            f"label_{n1}": lab1,
            f"label_{n2}": lab2,
            "timestamp_utc": utc_now(),
        })

    rule("Done")
    info(f"{n_auto} auto-agreed, {n_manual} manually resolved -> {out}")
    print("\nNext:\n    python compute_signals.py\n    python evaluate.py\n")


if __name__ == "__main__":
    main()
