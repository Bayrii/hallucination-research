"""
Stage 6 — Evaluate the signals against consensus ground truth.

Merges signals.jsonl with labels_final.jsonl and answers the paper's question:
how well does each cheap signal detect an unsupported answer, and at what cost?

DETECTION DIRECTION
-------------------
The positive class is UNSUPPORTED — that is the thing we are trying to catch.
Every signal is oriented so that HIGHER = MORE SUPPORTED (high log-prob, high
consistency, high entailment), so the detection score is the NEGATED signal.
Getting this backwards yields AUROC ~= 1 - AUROC and looks like a great result,
so it is done in exactly one place: `detection_score()`.

WHY AUROC IS THE HEADLINE, NOT BEST-THRESHOLD F1
------------------------------------------------
Sweeping thresholds on the full dataset and reporting the maximum F1 tunes a
parameter on the test set. It is optimistically biased and a reviewer will say
so. Here:

  * AUROC / AUPRC are reported as the primary metrics — threshold-free, with
    1000-resample bootstrap confidence intervals. With n~150 those intervals
    are wide, and showing them is the honest move.
  * F1 comes from stratified K-fold CV: the threshold is chosen on training
    folds and F1 measured on held-out folds. Reported as mean +/- std.

Usage:
    python evaluate.py
    python evaluate.py --partial supported --abstention exclude
    python evaluate.py --combined
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import (
    CONDITIONS,
    DATA_DIR,
    RESULTS_DIR,
    die,
    info,
    log_run_config,
    read_jsonl,
    require_file,
    rule,
    set_seed,
    warn,
)

# (key in signals.jsonl, display name, signal family)
# Family groups variants of the same underlying idea so the chart doesn't imply
# there are nine independent signals when there are really three plus a baseline.
SIGNAL_SPECS: list[tuple[str, str, str]] = [
    ("logprob_mean",          "LogProb (mean)",       "logprob"),
    ("logprob_min",           "LogProb (min)",        "logprob"),
    ("consistency_nli",       "SelfConsist (NLI)",    "consistency"),
    ("consistency_embedding", "SelfConsist (embed)",  "consistency"),
    ("nli_entail_min",        "NLI entail (min)",     "nli"),
    ("nli_entail_mean",       "NLI entail (mean)",    "nli"),
    ("nli_entail_whole",      "NLI entail (whole)",   "nli"),
    ("baseline_cosine",       "Baseline (cosine)",    "baseline"),
    ("baseline_rouge_l",      "Baseline (ROUGE-L)",   "baseline"),
]

# One representative per family, for the headline figure and combined model.
PRIMARY = ["logprob_mean", "consistency_nli", "nli_entail_min", "baseline_cosine"]

TIMING_FOR_SIGNAL = {
    "logprob_mean": "logprob",
    "logprob_min": "logprob",
    "consistency_nli": "consistency_nli",
    "consistency_embedding": "consistency_embedding",
    "nli_entail_min": "nli_entailment",
    "nli_entail_mean": "nli_entailment",
    "nli_entail_whole": "nli_entailment",
    "baseline_cosine": "baseline",
    "baseline_rouge_l": "baseline",
}


# =============================================================================
# Label handling
# =============================================================================


def binarize(label: str, is_abstention: bool, partial: str, abstention: str):
    """
    Map a 3-way label to binary 'is unsupported'. None means "drop this item".

    Abstention is checked FIRST: a refusal asserts nothing ungrounded, so which
    class it belongs to is a research decision, not a property of the label.
    """
    if is_abstention:
        if abstention == "exclude":
            return None
        return 1 if abstention == "unsupported" else 0

    if label == "supported":
        return 0
    if label == "unsupported":
        return 1
    if label == "partially_supported":
        if partial == "exclude":
            return None
        return 1 if partial == "unsupported" else 0
    return None


def detection_score(value: float) -> float:
    """Signals are oriented high=supported; detection targets unsupported."""
    return -float(value)


# =============================================================================
# Metrics
# =============================================================================


def bootstrap_auroc(y, s, n_boot: int, seed: int) -> tuple[float, float, float, float]:
    """AUROC, AUPRC, and the 95% bootstrap CI for AUROC."""
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y)
    s = np.asarray(s, dtype=float)

    auroc = float(roc_auc_score(y, s))
    auprc = float(average_precision_score(y, s))

    rng = np.random.default_rng(seed)
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        # A resample can be single-class, where AUROC is undefined — skip it
        # rather than letting it distort the interval.
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], s[idx]))

    if len(boots) < n_boot * 0.5:
        return auroc, auprc, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auroc, auprc, float(lo), float(hi)


def best_threshold(y, s):
    """Threshold maximizing F1 on the data given. Used INSIDE training folds only."""
    import numpy as np
    from sklearn.metrics import f1_score

    cands = np.unique(s)
    if len(cands) > 200:  # keep the sweep cheap on continuous signals
        cands = np.quantile(s, np.linspace(0, 1, 200))

    best_f1, best_t = -1.0, float(cands[0])
    for t in cands:
        f1 = f1_score(y, (s >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def cv_f1(y, s, n_splits: int, seed: int) -> dict:
    """
    Cross-validated precision/recall/F1 with the threshold chosen on train folds.

    This is the whole point of the stage: an honest F1 that was never allowed to
    see the data it is scored on.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, precision_score, recall_score

    y = np.asarray(y)
    s = np.asarray(s, dtype=float)

    n_min = int(min(np.sum(y == 0), np.sum(y == 1)))
    if n_min < 2:
        return {"f1": float("nan"), "f1_std": float("nan"), "precision": float("nan"),
                "recall": float("nan"), "threshold": float("nan")}
    splits = max(2, min(n_splits, n_min))

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    f1s, ps, rs, ts = [], [], [], []
    for tr, te in skf.split(s.reshape(-1, 1), y):
        t, _ = best_threshold(y[tr], s[tr])
        pred = (s[te] >= t).astype(int)
        f1s.append(f1_score(y[te], pred, zero_division=0))
        ps.append(precision_score(y[te], pred, zero_division=0))
        rs.append(recall_score(y[te], pred, zero_division=0))
        ts.append(t)

    return {
        "f1": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "precision": float(np.mean(ps)),
        "recall": float(np.mean(rs)),
        "threshold": float(np.mean(ts)),
    }


# =============================================================================
# Main analysis
# =============================================================================


def evaluate_signal(items, key, n_boot, n_splits, seed) -> dict | None:
    import numpy as np

    usable = [it for it in items if it["signals"].get(key) is not None]
    if len(usable) < 10:
        return None

    y = [it["y"] for it in usable]
    s = [detection_score(it["signals"][key]) for it in usable]
    if len(set(y)) < 2:
        return None

    auroc, auprc, lo, hi = bootstrap_auroc(y, s, n_boot, seed)
    out = {"n": len(usable), "n_pos": int(sum(y)), "auroc": auroc, "auprc": auprc,
           "ci_lo": lo, "ci_hi": hi}
    out.update(cv_f1(y, s, n_splits, seed))

    tkey = TIMING_FOR_SIGNAL.get(key)
    lat = [it["timing"].get(tkey) for it in usable if it["timing"].get(tkey) is not None]
    out["latency_ms"] = float(np.mean(lat)) if lat else float("nan")
    return out


def evaluate_judge(items, judge_rows, partial, abstention) -> dict | None:
    """
    Score a judge's hard labels. No threshold to sweep, so no CV is needed —
    but that also means no AUROC, which is why it is reported separately.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    import numpy as np

    by_id = {r["qa_id"]: r for r in judge_rows if r.get("label")}
    y_true, y_pred, lat = [], [], []
    for it in items:
        jr = by_id.get(it["qa_id"])
        if jr is None:
            continue
        pred = binarize(jr["label"], False, partial, abstention)
        if pred is None:
            continue
        y_true.append(it["y"])
        y_pred.append(pred)
        if jr.get("elapsed_ms") is not None:
            lat.append(jr["elapsed_ms"])

    if len(y_true) < 10 or len(set(y_true)) < 2:
        return None

    agree = float(np.mean([a == b for a, b in zip(y_true, y_pred)]))
    return {
        "n": len(y_true), "n_pos": int(sum(y_true)),
        "auroc": float("nan"), "auprc": float("nan"),
        "ci_lo": float("nan"), "ci_hi": float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_std": float("nan"),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "threshold": float("nan"),
        "latency_ms": float(np.mean(lat)) if lat else float("nan"),
        "binary_agreement": agree,
    }


def combined_model(items, n_splits, seed) -> dict | None:
    """Cross-validated logistic regression over the primary signals."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    feats = [k for k in PRIMARY if any(it["signals"].get(k) is not None for it in items)]
    usable = [it for it in items if all(it["signals"].get(k) is not None for k in feats)]
    if len(usable) < 20 or not feats:
        return None

    X = np.array([[detection_score(it["signals"][k]) for k in feats] for it in usable])
    y = np.array([it["y"] for it in usable])
    if len(set(y)) < 2:
        return None

    n_min = int(min(np.sum(y == 0), np.sum(y == 1)))
    splits = max(2, min(n_splits, n_min))
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    auroc, auprc, lo, hi = bootstrap_auroc(y, proba, 1000, seed)
    out = {"n": len(usable), "n_pos": int(y.sum()), "auroc": auroc, "auprc": auprc,
           "ci_lo": lo, "ci_hi": hi, "latency_ms": float("nan"), "features": feats}
    out.update(cv_f1(y, proba, splits, seed))
    return out


# =============================================================================
# Reporting
# =============================================================================


def print_table(title: str, rows: list[tuple[str, dict]]) -> None:
    rule(title)
    print(
        f"\n  {'signal':<22}{'n':>5}{'pos':>5}{'AUROC':>8}"
        f"{'95% CI':>16}{'AUPRC':>8}{'F1(cv)':>9}{'P':>7}{'R':>7}{'ms':>10}"
    )
    print("  " + "-" * 97)
    for name, m in rows:
        ci = (
            f"[{m['ci_lo']:.2f},{m['ci_hi']:.2f}]"
            if m["ci_lo"] == m["ci_lo"]
            else "  --"
        )
        auroc = f"{m['auroc']:.3f}" if m["auroc"] == m["auroc"] else "  --"
        auprc = f"{m['auprc']:.3f}" if m["auprc"] == m["auprc"] else "  --"
        f1 = f"{m['f1']:.3f}" if m["f1"] == m["f1"] else "  --"
        lat = f"{m['latency_ms']:.1f}" if m["latency_ms"] == m["latency_ms"] else "--"
        print(
            f"  {name:<22}{m['n']:>5}{m['n_pos']:>5}{auroc:>8}{ci:>16}"
            f"{auprc:>8}{f1:>9}{m['precision']:>7.3f}{m['recall']:>7.3f}{lat:>10}"
        )
    print()


def make_plots(overall, per_condition, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless: never try to open a window on a server
    import matplotlib.pyplot as plt
    import numpy as np

    names = [n for n, _ in overall]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    # --- 1. AUROC with bootstrap CI ----------------------------------------
    ax = axes[0]
    vals = [m["auroc"] for _, m in overall]
    errs = [
        [max(0, m["auroc"] - m["ci_lo"]) for _, m in overall],
        [max(0, m["ci_hi"] - m["auroc"]) for _, m in overall],
    ]
    ok = [v == v for v in vals]
    ax.barh(
        [n for n, k in zip(names, ok) if k],
        [v for v, k in zip(vals, ok) if k],
        xerr=[[e for e, k in zip(errs[0], ok) if k], [e for e, k in zip(errs[1], ok) if k]],
        color="#4C72B0", capsize=3,
    )
    ax.axvline(0.5, color="grey", ls="--", lw=1, label="chance")
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUROC (detecting unsupported)")
    ax.set_title("Detection performance\n(95% bootstrap CI)")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()

    # --- 2. F1 by intended condition ---------------------------------------
    ax = axes[1]
    conds = [c for c in CONDITIONS if c in per_condition]
    if conds:
        sig_names = list(dict.fromkeys(n for c in conds for n, _ in per_condition[c]))
        width = 0.8 / max(1, len(conds))
        x = np.arange(len(sig_names))
        for i, c in enumerate(conds):
            lookup = dict(per_condition[c])
            ys = [lookup[n]["f1"] if n in lookup else np.nan for n in sig_names]
            ax.bar(x + i * width, ys, width, label=c.replace("_", " "))
        ax.set_xticks(x + width * (len(conds) - 1) / 2)
        ax.set_xticklabels(sig_names, rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=8)
    ax.set_ylabel("F1 (cross-validated)")
    ax.set_ylim(0, 1)
    ax.set_title("Degradation by retrieval quality")

    # --- 3. the paper's headline: accuracy vs cost -------------------------
    ax = axes[2]
    for n, m in overall:
        if m["auroc"] != m["auroc"] or m["latency_ms"] != m["latency_ms"]:
            continue
        ax.scatter(max(m["latency_ms"], 1e-3), m["auroc"], s=70)
        ax.annotate(
            n, (max(m["latency_ms"], 1e-3), m["auroc"]),
            textcoords="offset points", xytext=(6, 4), fontsize=8,
        )
    ax.set_xscale("log")
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xlabel("mean latency per query (ms, log scale)")
    ax.set_ylabel("AUROC")
    ax.set_title("Accuracy vs. cost")
    ax.set_ylim(0, 1)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    info(f"chart -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate detection signals against consensus ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--signals", default=str(DATA_DIR / "signals.jsonl"))
    ap.add_argument("--labels", default=str(DATA_DIR / "labels_final.jsonl"))
    ap.add_argument("--generations", default=str(DATA_DIR / "generations.jsonl"))
    ap.add_argument("--judge-files", nargs="*", default=None,
                    help="default: auto-discover data/judge_*.jsonl")
    ap.add_argument("--partial", choices=["supported", "unsupported", "exclude"],
                    default="unsupported", help="how to treat partially_supported")
    ap.add_argument("--abstention", choices=["supported", "unsupported", "exclude"],
                    default="supported", help="how to treat correct refusals")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--combined", action="store_true",
                    help="also fit a CV'd logistic regression over the signals")
    ap.add_argument("--out-csv", default=str(RESULTS_DIR / "summary.csv"))
    ap.add_argument("--out-png", default=str(RESULTS_DIR / "signals_comparison.png"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    log_run_config("evaluate", args)
    require_file(args.signals, "python compute_signals.py")
    require_file(args.labels, "python compute_agreement.py --resolve")

    signals = {r["qa_id"]: r for r in read_jsonl(args.signals)}
    labels = {r["qa_id"]: r for r in read_jsonl(args.labels)}
    gens = (
        {g["qa_id"]: g for g in read_jsonl(args.generations)}
        if Path(args.generations).exists() else {}
    )

    # --- merge --------------------------------------------------------------
    items, n_dropped = [], 0
    for qa_id, lab in labels.items():
        sig = signals.get(qa_id)
        if sig is None:
            continue
        y = binarize(lab["label"], bool(lab.get("is_abstention")),
                     args.partial, args.abstention)
        if y is None:
            n_dropped += 1
            continue
        items.append({
            "qa_id": qa_id,
            "y": y,
            "signals": sig,
            "timing": sig.get("timing_ms", {}),
            "condition": sig.get("intended_condition")
            or gens.get(qa_id, {}).get("intended_condition", ""),
            "strategy": sig.get("distractor_strategy")
            or gens.get(qa_id, {}).get("distractor_strategy", ""),
            "retrieval_hit": sig.get("retrieval_hit"),
        })

    rule("Stage 6: evaluation")
    info(f"partial -> {args.partial};  abstention -> {args.abstention}")
    if not items:
        die("No items left after merging signals with labels.",
            "Check that qa_ids match between signals.jsonl and labels_final.jsonl.")

    n_pos = sum(it["y"] for it in items)
    info(f"{len(items)} items  ({n_pos} unsupported, {len(items) - n_pos} supported)"
         + (f", {n_dropped} excluded by config" if n_dropped else ""))

    if n_pos < 10 or (len(items) - n_pos) < 10:
        warn("one class has <10 items — every metric below will be very unstable.")

    hits = [it["retrieval_hit"] for it in items if it["retrieval_hit"] is not None]
    if hits:
        info(f"retrieval hit rate: {100.0 * sum(hits) / len(hits):.1f}% "
             f"(source abstract present in retrieved context)")

    # --- overall ------------------------------------------------------------
    overall = []
    for key, name, _fam in SIGNAL_SPECS:
        m = evaluate_signal(items, key, args.n_boot, args.cv_folds, args.seed)
        if m:
            overall.append((name, m))
    if not overall:
        die("No signal had enough usable data to evaluate.",
            "Check that signals.jsonl is populated (compute_signals.py).")

    # --- judges -------------------------------------------------------------
    judge_files = args.judge_files
    if judge_files is None:
        judge_files = sorted(str(p) for p in DATA_DIR.glob("judge_*.jsonl"))
    for jf in judge_files:
        if not Path(jf).exists():
            continue
        rows = read_jsonl(jf)
        m = evaluate_judge(items, rows, args.partial, args.abstention)
        if m:
            label = f"JUDGE {Path(jf).stem.replace('judge_', '')}"
            overall.append((label, m))
            info(f"{label}: binary agreement with humans = {m['binary_agreement']:.3f}")

    if args.combined:
        cm = combined_model(items, args.cv_folds, args.seed)
        if cm:
            overall.append(("COMBINED (LR)", cm))
            info(f"combined model features: {', '.join(cm['features'])}")

    overall.sort(key=lambda kv: (-(kv[1]["auroc"] if kv[1]["auroc"] == kv[1]["auroc"] else -1)))
    print_table("Overall — detecting UNSUPPORTED answers", overall)

    # --- per condition ------------------------------------------------------
    per_condition: dict[str, list] = {}
    for cond in CONDITIONS:
        subset = [it for it in items if it["condition"] == cond]
        if len(subset) < 10:
            continue
        ys = [it["y"] for it in subset]
        if len(set(ys)) < 2:
            info(f"condition '{cond}': {len(subset)} items but only one class "
                 f"({'all unsupported' if ys[0] else 'all supported'}) — "
                 f"threshold metrics undefined, skipping.")
            continue
        rows = []
        for key, name, _fam in SIGNAL_SPECS:
            m = evaluate_signal(subset, key, args.n_boot, args.cv_folds, args.seed)
            if m:
                rows.append((name, m))
        if rows:
            per_condition[cond] = rows
            print_table(f"Condition: {cond}  (n={len(subset)})", rows)

    # --- hard vs easy negatives --------------------------------------------
    # Splits poorly_supported by how the distractor was chosen. `similar` is a
    # same-subfield hard negative (what real retrieval failure looks like);
    # `dissimilar` is obviously unrelated. Collapsed together they hide the
    # degradation gradient, which is the point of the condition in the first place.
    per_strategy: dict[str, list] = {}
    for strat in sorted({it["strategy"] for it in items if it["strategy"]}):
        subset = [it for it in items if it["strategy"] == strat]
        if len(subset) < 10:
            info(f"distractor '{strat}': only {len(subset)} items — skipped.")
            continue
        ys = [it["y"] for it in subset]
        if len(set(ys)) < 2:
            info(
                f"distractor '{strat}': {len(subset)} items but a single class "
                f"— threshold metrics undefined, skipped."
            )
            continue
        rows = []
        for key, name, _fam in SIGNAL_SPECS:
            m = evaluate_signal(subset, key, args.n_boot, args.cv_folds, args.seed)
            if m:
                rows.append((name, m))
        if rows:
            hardness = {
                "similar": "hard negatives",
                "dissimilar": "easy negatives",
                "random": "mixed negatives",
            }.get(strat, strat)
            per_strategy[f"distractor_{strat}"] = rows
            print_table(f"Distractor '{strat}' — {hardness}  (n={len(subset)})", rows)

    # --- csv ----------------------------------------------------------------
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["scope", "signal", "n", "n_pos", "auroc", "auroc_ci_lo", "auroc_ci_hi",
            "auprc", "f1_cv", "f1_std", "precision", "recall", "threshold", "latency_ms"]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        scopes = (
            [("overall", overall)]
            + list(per_condition.items())
            + list(per_strategy.items())
        )
        for scope, rows in scopes:
            for name, m in rows:
                w.writerow([
                    scope, name, m["n"], m["n_pos"],
                    f"{m['auroc']:.4f}", f"{m['ci_lo']:.4f}", f"{m['ci_hi']:.4f}",
                    f"{m['auprc']:.4f}", f"{m['f1']:.4f}", f"{m['f1_std']:.4f}",
                    f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                    f"{m['threshold']:.4f}", f"{m['latency_ms']:.3f}",
                ])
    info(f"summary -> {out_csv}")

    make_plots(overall, per_condition, Path(args.out_png))

    rule("Interpretation notes")
    print(
        "\n  * AUROC is the headline. Best-threshold F1 on the full set would be\n"
        "    optimistically biased; F1 here is cross-validated.\n"
        "  * Check whether the CIs overlap before claiming one signal beats another.\n"
        "  * The NLI signal is partly circular: the annotation guideline asks the\n"
        "    same question NLI computes. Frame it as cost, not as discovery.\n"
        "  * Re-run with --partial supported to check the conclusion is not an\n"
        "    artifact of how partially_supported was binarized.\n"
    )


if __name__ == "__main__":
    main()
