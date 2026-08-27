# How to label — instructions for Bayram and valiyyaddin

We need to decide, for each answer the model wrote, whether it is actually
supported by the text it was given. This is the answer key the whole paper is
measured against.

**We both label all 150 items.** Agreement is computed over the full set rather
than a sample, which gives a tighter and more defensible reliability estimate.

> ### SECOND PASS - read this first
>
> Our first pass reached only kappa = 0.26 ("fair"). The cause was not
> carelessness: the guideline did not cover the most common case in the data, so
> we labelled it in opposite directions. Specifically, **41 items where the model
> answered about the requested paper from its own training data** were marked
> `supported` by one of us and `unsupported` by the other. The second reading is
> the correct one (see case 4 below), and applying it changes the ground truth
> from 4 unsupported items to 45 - the difference between an unusable dataset and
> a workable one.
>
> The guideline below now covers that case. We are both re-labelling all 150
> items independently with it. The first-pass labels are archived in
> `data/pass1/` so we can report both figures honestly.
>
> Budget about 1-1.5 hours each - faster than the first pass, since the items are
> familiar.

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

## The one mistake to avoid

**Do not press `s` because the answer looks CORRECT.**

An answer can be completely true and still be **unsupported** — because the
context in front of you does not say it. "Supported" means *this context states
this*, nothing else.

If you catch yourself thinking *"yes, that is what that paper does"* — stop and
ask instead: *"can I point at the sentence in THIS context that says so?"* If you
cannot, it is not supported, however right it sounds.

This is what went wrong in our first pass, so it is worth re-reading.

## The confusing cases

Some items were deliberately given the **wrong paper**. The model usually
notices, then does one of four things. **Cases 3 and 4 look alike and are
opposites.**

**1. It refuses.** *"This does not relate to the context, I cannot answer."*
→ `r`

**2. It refuses, then guesses from the title.** *"Not the right context, but that
paper seems to be about attention-based auditing…"*
→ `p` or `u` — it is inventing from the **title in the question**, not reading
the context. A sentence or two → `p`. A whole answer of it → `u`.

**3. It describes the paper it was GIVEN.** → **`s`**
*"The context is for paper Y, not paper X. Paper Y introduces ETRAG, which…"*
→ press `s`, then `n` and type `answered about context paper`

Every claim really is in the context. It answered the wrong question — a real
failure, but not an ungrounded one.

**4. It answers about the REQUESTED paper from memory.** → **`u`**
*"Paper X does not relate to this context. Paper X proposes a layered oversight
method that…"*
→ press `u`, then `n` and type `answered from parametric knowledge`

The model recognised paper X from its training data and described it. Those
claims are **not in the context**, so they are not supported — even if every word
is factually right.

**This is the single most important case in the whole study.** It is the model
ignoring the document it was given and falling back on what it memorised —
exactly the failure we are trying to detect. Marking these `s` because they are
accurate would erase the thing we are measuring.

> **3 vs 4 in one line:** does the answer describe **the paper in front of you**
> (3 → `s`) or **the paper named in the question** (4 → `u`)?

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
