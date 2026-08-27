# How to label — instructions for Bayram and valiyyaddin

We need to decide, for each answer the model wrote, whether it is actually
supported by the text it was given. This is the answer key the whole paper is
measured against.

**We both label all 150 items.** Agreement is then computed over the full set
rather than a sample, which gives a tighter and more defensible reliability
estimate. Budget about 1.5–2 hours each.

---

## Bayram

```powershell
cd "E:\Coding\Hallucination research"
.venv\Scripts\python.exe label_ground_truth.py --annotator annotator1
```

## valiyyaddin

```powershell
cd "E:\Coding\Hallucination research"
.venv\Scripts\python.exe label_ground_truth.py --annotator annotator2
```

### If valiyyaddin works on a different computer

```powershell
git clone https://github.com/Bayrii/hallucination-research.git
cd hallucination-research
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe label_ground_truth.py --annotator annotator2
```

No GPU needed. When finished, push the labels so Bayram can pull them:

```powershell
git add data/labels_annotator2.jsonl
git commit -m "annotator2 labels"
git push
```

> **Already started with `--subset 50`? Do not delete anything.** Just run the
> command without the flag. The tool resumes from your labels file, not from the
> flag, so your existing labels are kept and it simply continues through the
> remaining items. A label is the same judgement regardless of which pool it was
> drawn from.
>
> The `--subset N` option still exists if a future study wants a
> sampled reliability pass; it is just not what we are doing here.

---

## What you do for each item

The screen shows three things:

- **QUESTION** — what was asked
- **CONTEXT** — the paper abstract the model was given
- **ANSWER** — what the model wrote

Ask yourself one question:

> **Is everything the answer says actually in the context above?**

Not "is it true in real life." Not "is it well written." Not "did it answer the
question." Only: **is it in there.**

Then press one key:

| Key | When to press it |
|-----|------------------|
| `s` | Everything the answer says is in the context |
| `p` | Some of it is in the context, some is not |
| `u` | The main things it says are not there, or contradict it |
| `r` | The model said it cannot answer from this context |

Other keys: `n` add a note · `b` go back and redo the last one · `k` skip ·
`q` save and quit · `?` show the guide again

Press `q` whenever you want. Running the command again continues exactly where
you stopped — nothing is lost.

Roughly 30 seconds per item, so about **1.5–2 hours** for Bayram and
**30–40 minutes** for valiyyaddin.

---

## Two rules while labelling

1. **Do not discuss the items with each other.** The point is two independent
   opinions. Comparing notes makes the agreement score meaningless.
2. **Item order is different for each of you.** That is intentional, not a bug.

---

## The confusing cases

Some items were deliberately given the **wrong paper**. The model often notices.
It reacts in three ways, and they get different keys:

**1. It refuses.** *"This does not relate to the context, I cannot answer."*
→ press `r`

**2. It refuses, then guesses anyway.** *"Not the right context, but that paper
seems to be about attention-based auditing…"*
→ press `p` or `u`. It is guessing from the paper **title in the question**, not
from the context, so those claims are not supported. A sentence or two of
guessing → `p`. The whole answer → `u`.

**3. It describes the wrong paper correctly.** *"The context is for paper Y, not
paper X. Paper Y introduces ETRAG, which does…"*
→ press `s`, then `n` and type `answered about context paper`

Case 3 feels wrong, but it is right: everything the model said really is in the
context. It just answered a different question. The note is how we count those
separately.

---

## When you are both finished

Bayram runs:

```powershell
.venv\Scripts\python.exe compute_agreement.py
```

This prints how often you two agreed and lists every item where you disagreed.

Then, **together**, go through the disagreements and pick a final answer for each:

```powershell
.venv\Scripts\python.exe compute_agreement.py --resolve
```

This writes `data/labels_final.jsonl`, which is what produces the paper's
results. After that the rest is automatic:

```powershell
.venv\Scripts\python.exe compute_signals.py --device cuda
.venv\Scripts\python.exe evaluate.py --combined
```

---

## You can start now

You do not need to wait for the generation run to finish. The tool reads
whatever is ready and skips what you have already done, so label some tonight
and the rest whenever.
