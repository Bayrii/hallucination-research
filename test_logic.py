"""Exercise the dependency-free logic. No installs, no models, no network."""
import sys, tempfile, os
sys.path.insert(0, r"c:\Dev\Hallucination research")

fails = []
def check(name, got, want):
    if got != want:
        fails.append(f"{name}\n     got : {got!r}\n     want: {want!r}")
    else:
        print(f"  ok  {name}")

# --- common.split_sentences -------------------------------------------------
from common import split_sentences, normalize_arxiv_id, read_jsonl, append_jsonl

print("\n[split_sentences]")
check("abbrev e.g. not split",
      split_sentences("We use RAG, e.g. dense retrieval. It works."),
      ["We use RAG, e.g. dense retrieval.", "It works."])
check("decimal not split",
      split_sentences("Accuracy was 95.4% overall. Good."),
      ["Accuracy was 95.4% overall.", "Good."])
check("et al. not split",
      split_sentences("Following Lewis et al. we test this. Done."),
      ["Following Lewis et al. we test this.", "Done."])
check("empty", split_sentences(""), [])
check("no terminal punct", split_sentences("one sentence no period"),
      ["one sentence no period"])
# Regression: "al." used to match inside "retrieval.", merging the abstract.
check("'al.' not matched inside 'retrieval.'",
      split_sentences("We improve retrieval. It works."),
      ["We improve retrieval.", "It works."])
check("'Ms.' not matched inside 'algorithms.'",
      split_sentences("We compare algorithms. Results follow."),
      ["We compare algorithms.", "Results follow."])
check("'al.' not matched inside 'empirical.'",
      split_sentences("The study is empirical. We conclude."),
      ["The study is empirical.", "We conclude."])
check("real 'et al.' still protected",
      split_sentences("Lewis et al. proposed RAG. We extend it."),
      ["Lewis et al. proposed RAG.", "We extend it."])
check("multi-sentence abstract",
      len(split_sentences(
          "RAG combines retrieval and generation. We evaluate on 3 datasets. "
          "Accuracy reaches 91.2% overall. See Fig. 2 for details.")),
      4)

print("\n[normalize_arxiv_id]")
check("v3 stripped", normalize_arxiv_id("2401.12345v3"), "2401.12345")
check("no version", normalize_arxiv_id("2401.12345"), "2401.12345")
check("old style", normalize_arxiv_id("cs/0501001v1"), "cs/0501001")

# --- jsonl round trip + truncated tail --------------------------------------
print("\n[jsonl]")
d = tempfile.mkdtemp()
p = os.path.join(d, "t.jsonl")
append_jsonl(p, {"qa_id": "a", "v": 1})
append_jsonl(p, {"qa_id": "b", "v": 2})
check("round trip", [r["qa_id"] for r in read_jsonl(p)], ["a", "b"])
with open(p, "a", encoding="utf-8") as f:
    f.write('{"qa_id": "c", "v":')          # simulate a crash mid-write
check("truncated tail skipped", [r["qa_id"] for r in read_jsonl(p, quiet=True)], ["a", "b"])

from common import completed_ids
check("completed_ids", completed_ids(p, "qa_id"), {"a", "b"})

# --- ROUGE-L ----------------------------------------------------------------
print("\n[rouge_l_recall]")
from compute_signals import rouge_l_recall
check("identical -> 1.0", rouge_l_recall("the cat sat", "the cat sat"), 1.0)
check("disjoint -> 0.0", rouge_l_recall("alpha beta", "gamma delta"), 0.0)
check("half covered", rouge_l_recall("the cat", "the dog"), 0.5)
check("empty answer", rouge_l_recall("", "the cat"), 0.0)
sub = rouge_l_recall("cat sat", "the cat sat on the mat")
check("answer fully inside context", sub, 1.0)

# --- binarize ---------------------------------------------------------------
print("\n[binarize]")
from evaluate import binarize, detection_score
check("supported", binarize("supported", False, "unsupported", "supported"), 0)
check("unsupported", binarize("unsupported", False, "unsupported", "supported"), 1)
check("partial->unsup (default)",
      binarize("partially_supported", False, "unsupported", "supported"), 1)
check("partial->sup", binarize("partially_supported", False, "supported", "supported"), 0)
check("partial->exclude", binarize("partially_supported", False, "exclude", "supported"), None)
check("abstention beats label",
      binarize("supported", True, "unsupported", "unsupported"), 1)
check("abstention->exclude", binarize("unsupported", True, "unsupported", "exclude"), None)
check("abstention->supported", binarize("unsupported", True, "unsupported", "supported"), 0)
check("detection_score negates", detection_score(0.8), -0.8)

# --- judge label parsing ----------------------------------------------------
print("\n[parse_label]")
from run_judge_baseline import parse_label
check("plain", parse_label("supported"), "supported")
check("compound not swallowed", parse_label("partially_supported"), "partially_supported")
check("hyphen form", parse_label("partially-supported"), "partially_supported")
check("space form", parse_label("partially supported"), "partially_supported")
check("unsupported not read as supported", parse_label("unsupported"), "unsupported")
check("sentence wrapper", parse_label("The answer is unsupported."), "unsupported")
check("trailing prose partial",
      parse_label("partially supported, since one claim is missing"),
      "partially_supported")
check("empty -> None", parse_label(""), None)
check("garbage -> None", parse_label("i don't know"), None)

# --- logprob extraction shapes ---------------------------------------------
print("\n[extract_logprobs]")
from run_pipeline import extract_logprobs
check("shape A (native)",
      extract_logprobs({"logprobs": [{"token": "a", "logprob": -0.1},
                                     {"token": "b", "logprob": -0.5}]}),
      (["a", "b"], [-0.1, -0.5]))
check("shape B (openai)",
      extract_logprobs({"logprobs": {"content": [{"token": "x", "logprob": -0.2}]}}),
      (["x"], [-0.2]))
check("shape C (bare floats)",
      extract_logprobs({"logprobs": [-0.1, -0.2], "tokens": ["a", "b"]}),
      (["a", "b"], [-0.1, -0.2]))
check("absent", extract_logprobs({"response": "hi"}), ([], []))

# --- logprob signals --------------------------------------------------------
print("\n[logprob_signals]")
try:
    import numpy  # noqa: F401
    from compute_signals import logprob_signals
    s = logprob_signals({"token_logprobs": [-1.0, -2.0, -3.0]})
    check("mean", s["logprob_mean"], -2.0)
    check("min", s["logprob_min"], -3.0)
    check("count", s["answer_n_tokens"], 3)
    check("empty -> nulls",
          logprob_signals({"token_logprobs": []})["logprob_mean"], None)
except ImportError:
    print("  -- skipped (numpy not installed on this machine)")

# --- qa_pairs csv round trip ------------------------------------------------
# Regression: the team edits this file in Excel. An added column used to crash
# DictWriter; extra columns must now survive the round trip intact.
print("\n[qa_pairs csv round trip]")
from generate_qa_pairs import write_rows, read_rows, FIELDNAMES
import pathlib

row = {k: "" for k in FIELDNAMES}
row.update({"qa_id": "2401.1__method", "source_arxiv_id": "2401.1",
            "question": "Q?", "intended_condition": "well_supported",
            "reviewer": "bayram", "comments": "looks fine"})
p = pathlib.Path(d) / "qa_pairs.csv"
write_rows(p, [row])
back = read_rows(p)
check("row survives", back[0]["qa_id"], "2401.1__method")
check("extra column 'reviewer' preserved", back[0].get("reviewer"), "bayram")
check("extra column 'comments' preserved", back[0].get("comments"), "looks fine")
check("canonical columns intact", all(c in back[0] for c in FIELDNAMES), True)

print("\n" + "=" * 60)
if fails:
    print(f"{len(fails)} FAILURE(S):\n")
    for f in fails:
        print("  X " + f)
    sys.exit(1)
print("all logic checks passed")
