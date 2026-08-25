# Detecting Hallucinations in RAG with Lightweight Inference-Time Signals

Experimental pipeline for the ICRR submission.

**Question.** Can three cheap, inference-time signals detect when a RAG answer is
not supported by its retrieved context — without an expensive judge model?

**Signals.** Token log-probability · self-consistency across sampled generations ·
NLI entailment against the retrieved context.

The headline result is an **accuracy-vs-cost tradeoff**, not a single accuracy
number. Latency is measured per signal and treated as a primary axis.

---

## START HERE — current state (updated 2026-08-25)

Stages 0–2 are **done** and committed. Nothing has been run on a GPU yet; no
model has produced a single token in this project so far.

| Artifact | State |
|---|---|
| `data/corpus.json` | ✅ 50 abstracts, 6 related arXiv queries, surveys + stub abstracts filtered out |
| `data/qa_pairs.csv` | ✅ 150 QA pairs drafted. **Manual review in progress** — 124 `well_supported`, 26 `partially_supported`, 0 distractors assigned |
| Everything from Stage 3 on | ❌ not started |

**Next actions, in order** (steps 2–5 need the RTX 3060):

1. **Finish reviewing `data/qa_pairs.csv`** — set `intended_condition` to
   `partially_supported` wherever the abstract can't answer its question.
   `data/prescreen_flags.csv` lists candidates; the `dataset` flags are reliable,
   the `method`/`finding` flags are mostly false alarms. Target 30–60 total.
2. **Install** (see Setup below), then `python index_corpus.py`.
3. **Assign distractors** — converts 50 `well_supported` rows into
   `poorly_supported` by pointing them at the wrong abstract:
   ```
   python generate_qa_pairs.py --assign-distractors 25 --strategy similar
   python generate_qa_pairs.py --assign-distractors 25 --strategy dissimilar
   ```
   The arithmetic is `well_supported = 150 − partially − distractors`. At 26
   `partially_supported` this leaves 74 `well_supported`, which is healthy —
   keep 25/25. Only reduce the distractor count if `partially_supported` ends
   up *large* (say 80+), which is what would squeeze `well_supported`.
4. **Preflight, then generate** (~50 min):
   ```
   python run_pipeline.py --check --model qwen2.5:0.5b
   python run_pipeline.py --model qwen2.5:7b-instruct-q4_K_M --unload-when-done
   ```
5. **Label, score, evaluate** — Stages 4–6. These need no GPU.

### Things that already went wrong once — don't repeat them

- **`data/qa_pairs.csv` is the single source of truth.** There was a
  `qa_pairs.json` mirror that the pipeline read instead; it went stale twice and
  silently discarded hand-labelled rows with no error. It has been deleted and
  `run_pipeline.py` now reads the CSV directly. **Do not reintroduce a mirror.**
- **`--assign-distractors` overwrites `intended_condition`.** It defaults to
  `--only-condition well_supported` so hand-marked `partially_supported` rows are
  protected. Don't pass `--only-condition any` unless you mean it.
- **Always run `run_pipeline.py --check` first.** It verifies Ollama actually
  returns token log-probabilities (needs **v0.12.11+**). Signal 1 *is* logprobs,
  so without that verification a multi-hour run produces unusable data.
- **Unload the generator before Stage 5** (`--unload-when-done`, or
  `ollama stop <model>`). 6 GB will not hold the generator and the NLI model.

---

## Setup

Run everything on the **RTX 3060 machine**. Only Stage 3 needs the GPU, but it is
simplest to keep one environment.

```bash
python -m venv .venv
.venv\Scripts\activate            # PowerShell: .venv\Scripts\Activate.ps1

# CUDA build FIRST — the default PyPI wheel is CPU-only and will silently
# give you ~20x slower NLI scoring.
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

Install [Ollama](https://ollama.com/download) (**v0.12.11 or newer** — earlier
builds cannot return logprobs, which kills signal 1), then:

```bash
ollama --version
ollama pull qwen2.5:7b-instruct-q4_K_M     # ~4.7 GB, the main generator
ollama pull qwen2.5:0.5b                   # ~0.4 GB, smoke tests only
```

After the first successful install, freeze exact versions for the paper:

```bash
pip freeze > requirements.lock.txt
```

---

## Run order

| # | Command | Time | Needs GPU |
|---|---|---|---|
| 0 | `python build_corpus.py` | ~2 min | no |
| — | *manually prune `data/corpus.json`* | ~30 min | — |
| 1 | `python index_corpus.py` | ~10 sec | no |
| 2 | `python generate_qa_pairs.py` | seconds | no |
| — | *manually review `data/qa_pairs.csv`* | 1–2 hrs | — |
| 2b | `python generate_qa_pairs.py --assign-distractors 50 --strategy similar` | seconds | no |
| 3 | `python run_pipeline.py --check` then the real run | **~50 min** | **yes** |
| 3b | `python run_judge_baseline.py --judge gemini` *(optional)* | ~10 min | no |
| 4 | `python label_ground_truth.py --annotator annotator1` (×2 people) | ~1.5 hrs each | no |
| 4b | `python compute_agreement.py --resolve` | ~20 min | no |
| 5 | `python compute_signals.py` | 20–40 min | helps |
| 6 | `python evaluate.py --combined` | seconds | no |

### Stage 0–2 — corpus, index, questions

```bash
python build_corpus.py
python index_corpus.py
python index_corpus.py --query "how is faithfulness measured?" --k 5   # sanity check
python generate_qa_pairs.py
```

Open `data/qa_pairs.csv`, fix the questions, and set `intended_condition` per row.
Then force bad retrieval on a subset:

```bash
python generate_qa_pairs.py --assign-distractors 50 --strategy similar
```

`--strategy` controls how hard the negatives are: `similar` = same-subfield hard
negative (realistic), `dissimilar` = easy negative (upper bound), `random` = mixed.
Running a mix gives a difficulty gradient to report against.

> `data/qa_pairs.csv` is read directly by `run_pipeline.py`. Your edits take
> effect as soon as you save — there is no sync step.

### Stage 3 — generation

**Always run the preflight first.** It verifies that logprobs actually come back
in a shape the parser understands — the one failure that would silently ruin a
multi-hour run:

```bash
python run_pipeline.py --check --model qwen2.5:0.5b
python run_pipeline.py --smoke-test 3 --model qwen2.5:0.5b
python run_pipeline.py --model qwen2.5:7b-instruct-q4_K_M --unload-when-done
```

Resumable — re-run the same command after any crash and it picks up where it
stopped. 150 QA pairs × 6 generations = 900 generations.

### Stage 4 — labelling

```bash
python label_ground_truth.py --annotator annotator1
python label_ground_truth.py --annotator annotator2      # different person
python compute_agreement.py --resolve
```

Labelling is **blind**: `intended_condition` is hidden, and item order is shuffled
per annotator. Do not pass `--show-condition` for a real pass — it makes the
ground truth dependent on the manipulation it is supposed to validate.

`--resolve` writes `data/labels_final.jsonl`, which Stage 6 consumes.

### Stage 5–6 — signals and evaluation

```bash
ollama stop qwen2.5:7b-instruct-q4_K_M      # free VRAM first
python compute_signals.py --device cuda
python evaluate.py --combined
```

---

## VRAM budget (6 GB)

Load **one model at a time**. Stage 5 frees each model before loading the next.

| Model | Size | Stage |
|---|---|---|
| `qwen2.5:7b-instruct-q4_K_M` | ~4.7 GB | 3 — tight; use `--num-ctx 2048` |
| `qwen2.5:3b-instruct-q4_K_M` | ~2.0 GB | 3 — comfortable, ~2.5× faster |
| `all-MiniLM-L6-v2` | ~0.4 GB | 1, 5 |
| `nli-deberta-v3-base` | ~0.8 GB | 5 |

If Stage 5 OOMs, the generator is still resident. `ollama stop <model>` or run
Stage 3 with `--unload-when-done`.

---

## Method notes

Decisions a reviewer will ask about, and where they live in the code.

**AUROC is the headline, not best-threshold F1.** Sweeping thresholds on the full
set and reporting the max tunes a parameter on the test set. `evaluate.py` reports
AUROC/AUPRC with 1000-resample bootstrap CIs, and derives F1 from stratified K-fold
CV with the threshold chosen on training folds only.

**Abstention is not hallucination.** When the context is irrelevant, an
instruction-tuned model often correctly refuses. That answer asserts nothing
ungrounded. Refusals are flagged at label time (`r` key) and mapped at eval time
via `--abstention {supported,unsupported,exclude}`, so the choice is a reported
decision rather than a silent one.

**NLI uses sentence-level chunking.** These cross-encoders are trained on short
sentence pairs; a 200-word abstract as premise degrades them. `compute_signals.py`
builds a (context sentence × answer claim) entailment matrix and aggregates max
over premises, then min over claims — SummaC/FactCC style. The naive whole-passage
score is also stored so the choice can be justified with numbers.

**The NLI signal is partly circular.** The annotation guideline asks the same
question NLI computes. Frame the contribution as *cost* — a 184M cross-encoder
approximating a judge — not as a discovery.

**`retrieval_hit` guards the positive class.** With a topically tight corpus the
retriever can miss the source abstract for a `well_supported` question, silently
corrupting ground truth. Stage 3 records it; Stage 6 reports the rate.

**Expect self-consistency to be condition-dependent.** When context is present all
5 samples anchor to it and agree regardless of correctness; when it is irrelevant
they diverge. The per-condition breakdown is designed to surface exactly this.

---

## Files

```
build_corpus.py        0   fetch arXiv abstracts (6 related queries)
index_corpus.py        1   embed + retrieve; imported by run_pipeline
generate_qa_pairs.py   2   draft editable QA pairs; assign distractors
run_pipeline.py        3   generate answers + logprobs + 5 samples   [GPU]
run_judge_baseline.py  3b  LLM-as-judge baseline (optional)
label_ground_truth.py  4   blind human annotation CLI
compute_agreement.py   4b  Cohen's + weighted kappa; build consensus
compute_signals.py     5   the three signals + free baseline
evaluate.py            6   AUROC/AUPRC, CV'd F1, plots, CSV
common.py              —   jsonl io, seeding, timing, run-config logging
data/                  —   all artifacts
runs/                  —   resolved config of every run, for reproducibility
```

Every incremental output is JSONL so a crash costs one line, not the run. Every
stage is independently runnable and takes `--help`.

---

## Troubleshooting

**`No token log-probabilities came back`** — Ollama is older than v0.12.11.
Check `ollama --version` and upgrade.

**`Could not load cross-encoder/nli-deberta-v3-base`** — missing tokenizer deps:
`pip install sentencepiece protobuf`.

**`torch.cuda.is_available()` is False** — you have the CPU wheel. Reinstall with
the CUDA `--index-url` above.

**Stage 5 OOM** — the generator is still in VRAM. `ollama stop <model>`.

**Gemini judge returns empty responses** — thinking budget consumed the output
allowance. `run_judge_baseline.py` already sets `thinkingBudget: 0`.

**Everything is slower than the table says** — check you are on the 3060 and not
a CPU-only box. On a 4-core laptop CPU, Stage 3 with the 7B is ~12 hours instead
of ~50 minutes; drop to `qwen2.5:3b-instruct-q4_K_M` (~5 h) or `1.5b` (~2.5 h).
