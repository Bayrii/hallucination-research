# Detecting Hallucinations in RAG Pipelines: Can Lightweight Confidence Signals Catch Unsupported Claims?

*[Author names removed for anonymized review]*

---

## Abstract
*[Draft after Results are in — 250 words max, 4-6 keywords]*

**Keywords:** [e.g., retrieval-augmented generation, hallucination detection, natural language inference, self-consistency, low-resource evaluation]

---

## 1. Introduction

*[DRAFT — refine wording, this is a starting point]*

Retrieval-Augmented Generation (RAG) has become the standard architecture for grounding large language models (LLMs) in external, up-to-date knowledge rather than relying solely on parametric memory. Despite this, RAG systems still frequently produce hallucinations — confident, fluent claims that are not actually supported by the retrieved context. This remains one of the most commonly cited obstacles to deploying LLM-based systems in production, particularly for teams without the resources to run a large judge model or fine-tune a dedicated verifier.

Most existing hallucination-detection approaches either require a second, larger LLM to act as a judge, or require additional training data and fine-tuning. Both options are costly and often impractical for smaller teams or resource-constrained deployments. This raises a practical question: **how much can be achieved using only signals that are already available at inference time, from models that fit on modest hardware?**

This paper evaluates three lightweight, inference-time signals for flagging unsupported claims in a RAG pipeline: (1) token-level output probability, (2) self-consistency across repeated generations, and (3) NLI-based entailment scoring between generated claims and retrieved context. We build a small-scale RAG system over a corpus of scientific paper abstracts drawn from several related queries within one ML/NLP area (hallucination detection, RAG evaluation, faithfulness, attribution, and uncertainty estimation), and compare the three signals' detection accuracy and computational cost — all on resource-constrained hardware (a single consumer GPU) — to assess which signal, if any, offers a practical, low-cost hallucination flag for teams who cannot rely on larger judge models.

Crucially, we do not treat detection accuracy as the sole outcome. Because the premise of the paper is that a *cheap* signal may be good enough, per-signal latency is measured under identical conditions and reported as a first-class result alongside accuracy.

**Contributions:**
1. A direct, controlled comparison of three inference-time hallucination-detection signals under identical retrieval and generation conditions, against both a near-free lexical/embedding baseline and an LLM-judge reference point.
2. A resource-constrained experimental setup (single 6GB-VRAM GPU) demonstrating what is achievable without large-scale infrastructure.
3. An accuracy-versus-cost analysis intended to guide practical signal choice for small teams building RAG systems.
4. An analysis of how each signal *degrades* as retrieval quality drops, by evaluating separately under three controlled retrieval conditions — showing that the signals fail in different places rather than simply ranking against one another.

---

## 2. Related Work

*[DRAFT — expand with your own paraphrased summaries once you've reread each source; keep each source to 2-4 sentences, no direct quotes]*

**Hallucination detection via self-consistency.** SelfCheckGPT (Manakul et al., 2023) introduced a family of zero-resource, black-box hallucination-detection methods that sample multiple generations for the same prompt and measure agreement across samples, under the assumption that hallucinated content is less consistent across samples than grounded content is. Several variants were proposed (BERTScore, n-gram, NLI, MQAG, LLM-prompt), with the NLI variant generally offering a strong accuracy-to-cost balance.

**NLI-based factual consistency.** A separate line of work frames hallucination detection as a natural language inference problem: a generated claim is checked for entailment, neutrality, or contradiction against the retrieved/source context. This approach requires only a lightweight, pre-trained NLI model rather than an additional LLM call, making it attractive for resource-constrained settings.

**Uncertainty/probability-based signals.** Token-level output probabilities have also been explored as a low-cost proxy for hallucination risk, on the intuition that a model is less confident (lower probability) when generating content it has not actually retrieved support for. This is the cheapest of the three signal types, requiring no extra inference calls.

**Gap.** Prior work largely evaluates these signals independently, or compares them against large LLM-judge baselines that assume access to expensive frontier models. Little work directly compares all three lightweight signal types against each other under a shared, resource-constrained setup — which is the gap this paper addresses.

*(Add 1-2 more sources here — the RAG architecture papers you read — for context on the RAG pipeline itself, not just hallucination detection.)*

---

## 3. Methodology

*[DRAFT — fill in exact numbers once decided as a team]*

**3.1 Task and domain.** Claim verification / closed-domain QA against a corpus of ~50 scientific paper abstracts. Rather than a single narrow query, the corpus is assembled from six related arXiv queries within one area (hallucination in RAG, faithfulness/groundedness, RAG evaluation benchmarks, factual consistency, attribution/citation, and uncertainty estimation), merged round-robin so no one query dominates. This is deliberate: a single-query corpus yields ~50 near-interchangeable abstracts, which both collapses retrieval discrimination and makes every distractor an equally hard negative. Drawing from related-but-distinct facets instead yields a *range* of distractor difficulty, which §3.3 exploits as a controlled variable.

Questions ask about specific reported details in a given paper's abstract (e.g., reported metric, dataset used, proposed method) — chosen because claims are narrow, factual, and largely unambiguous to verify, and because the team already has strong domain familiarity here.

**3.2 RAG pipeline.**
- **Retriever:** embedding-based retrieval using `sentence-transformers/all-MiniLM-L6-v2`, cosine similarity over L2-normalised embeddings, top-k selection. Each abstract is embedded once as title + abstract; the title measurably helps short questions latch onto the right paper.
- **Generator:** Qwen2.5-7B-Instruct, 4-bit quantised (Q4_K_M, ~4.7 GB) served locally via Ollama. Ollama was chosen over `llama-cpp-python` for one decisive reason: it exposes per-token log-probabilities through its API (v0.12.11+), and signal 1 *is* token log-probability. Secondary reasons: CUDA ships with the installer rather than requiring a local compile, and its `keep_alive` control allows the generator to be evicted from VRAM on demand — necessary to honour the one-model-at-a-time constraint of §4.
- **No API generator.** An API model was considered for contrast and rejected. Generation requires 900 calls (150 pairs × 6 samples), which exceeds the free-tier daily quotas we had access to, and free-tier logprob support is inconsistent across providers. Routing generation through a hosted API would also weaken the paper's own claim of a cheap, local, reproducible setup. An API model is instead used only as a *judge* baseline (§3.6), where the call budget is 150 rather than 900 and no logprobs are needed.
- **Question templates:** three fixed templates per abstract (proposed method, dataset used, main finding). Each **names the target paper**, e.g. *"What method or approach does the paper "X" propose?"*. This is not cosmetic. An initial version used the deictic form *"...does this paper propose?"*, which is retrievable only by chance: measured over the 100 items that use live retrieval, top-1 hit rate was **2/100 at mean cosine 0.263**, so almost every nominally well-supported item was scored against the wrong abstract. Naming the paper raises this to **97/100 at mean cosine 0.683**. Template structure is otherwise identical across all rows, so question phrasing remains controlled — only the title varies.
- **Prompt template:** fixed template of the form `Context: {retrieved_chunks}\nQuestion: {query}\nAnswer:` — held constant across all conditions to avoid confounds, and versioned in the output records so any change is visible in the data rather than silent.

**3.3 Inducing hallucination conditions.** Query-context pairs are constructed under three retrieval conditions:
- **Well-supported** — retrieved chunks directly contain the answer.
- **Partially supported** — retrieved chunks are topically related but missing the specific fact.
- **Poorly supported** — retrieval is deliberately overridden with a different abstract, simulating retriever failure.

For the poorly-supported condition the distractor abstract is selected by one of three strategies, making negative difficulty an explicit variable rather than an accident of corpus construction: **similar** (highest-cosine other abstract — a same-subfield hard negative, the realistic failure mode), **dissimilar** (lowest-cosine — an easy negative, an effective upper bound), and **random**.

We additionally record, for every non-overridden item, whether the source abstract actually appeared in the retrieved top-k (`retrieval_hit`). Because the corpus is topically tight, the retriever can miss the source abstract for a nominally well-supported question, which would silently contaminate the positive class. The hit rate is reported in §5.

**3.4 Ground-truth labeling.** Two team members independently label each generated answer against its retrieved context on a three-point scale — *supported*, *partially supported*, *unsupported* — rather than a binary. Collapsing to binary is deferred to evaluation time (§3.6) and treated as a reported analysis choice rather than a labelling decision.

Labelling is **blind**: the intended retrieval condition is hidden from annotators, and item order is shuffled independently per annotator. If annotators could see that an item was constructed as poorly-supported they would label it unsupported, and the ground truth would cease to be independent of the manipulation it exists to validate.

**Abstentions are recorded separately.** When context is irrelevant, an instruction-tuned model frequently responds that the context does not contain the information. Such an answer asserts nothing ungrounded and is therefore not a hallucination, despite arising from a poorly-supported item. Annotators flag these explicitly, and their treatment (as supported, as unsupported, or excluded) is a reported configuration choice in §3.6. Abstention rate per condition is itself reported as a secondary result.

Inter-annotator agreement is reported as Cohen's kappa and, as the headline figure, **linearly-weighted** kappa: the label scale is ordinal, so unweighted kappa penalises a supported/partially-supported near-miss as heavily as a supported/unsupported inversion and understates real agreement. Disagreements are resolved by discussion into a consensus label set.

**3.5 Detection signals.** All signals are oriented so that a higher value means *more supported*; the detection score for the unsupported class is therefore the negated signal.

1. **Log-probability** — mean, minimum and standard deviation of token log-probability over the greedy generation, captured directly from the generator at no additional inference cost. We also record answer token count as a covariate: mean log-probability is length-dependent, and without the covariate a confidence effect cannot be distinguished from a length effect.

2. **Self-consistency** — N = 5 additional samples per query at temperature 0.7. Agreement is scored two ways, reported as a deliberate cost/fidelity contrast. The **NLI** variant computes mutual entailment across all 10 sample pairs in both directions (averaging the two directions, so that one-way entailment from a more specific answer does not read as full agreement). The **embedding** variant takes mean pairwise cosine similarity of sentence embeddings — substantially cheaper (the exact ratio is measured and reported in §5), but blind to negation, since *"X improves Y"* and *"X does not improve Y"* embed almost identically.

3. **NLI entailment** — a lightweight pretrained cross-encoder (`cross-encoder/nli-deberta-v3-base`, ~184M parameters) scores entailment between the generated answer and its retrieved context. Rather than scoring the answer against the whole abstract, we apply **sentence-level chunking**: the context is split into sentences and the answer into claims, an entailment matrix is computed over all (context sentence × answer claim) pairs, and per-claim scores are taken as the *max over context sentences* (is this claim supported anywhere?). This follows the SummaC/FactCC decomposition and is motivated by the fact that these cross-encoders are trained on short sentence pairs; supplying a 200-word abstract as premise degrades them.

   We aggregate across claims by **mean**, not by the stricter *min*. The min variant — the natural SummaC-style choice, and our initial default — proved unusable: it saturates near zero for *every* item, correctly-answered ones included, and therefore carries no information. The cause is claim granularity rather than a defect in the model. Sentence-level "claims" from an LLM are compound, mixing supported content with unsupported elaboration, and NLI (correctly) labels the whole sentence non-entailed if any part of it is unsupported. For example, against the premise *"RGB divides the instances … into 4 separate testbeds …"*, the claim *"The RGB divides the instances … into four separate testbeds, allowing for a systematic investigation of how well the models perform in each area"* scores neutral 1.000 / entailment 0.000, whereas the same claim truncated before the trailing clause scores entailment 0.997. Decomposition into atomic facts rather than sentences is the principled fix and is left to future work; the min variant is retained in the results table so this collapse is visible rather than hidden.

4. **NLI non-contradiction** — the same forward passes, reading the *contradiction* column instead. Entailment alone conflates two distinct failures that the three-point label scheme separates: content that is merely unsupported (NLI *neutral*, corresponding to *partially supported*) and content the context actively denies (NLI *contradiction*, corresponding to *unsupported*). We record 1 − max contradiction over claims, oriented like every other signal so that higher means more supported. This costs nothing beyond the entailment computation.

5. **Near-free baseline** — embedding cosine similarity and ROUGE-L recall between answer and context. This costs essentially nothing and exists to establish whether the more expensive signals earn their cost. If it matches the NLI signal, that is a reportable finding.

**3.6 Evaluation.** The three-point labels are collapsed to a binary *supported / unsupported* target, with the mapping of *partially supported* (default: unsupported) and of abstentions (default: supported) exposed as analysis flags; headline results are re-run under alternative mappings as a robustness check.

**Threshold-free metrics are primary.** We report AUROC and average precision (AUPRC) per signal with 95% confidence intervals from 1000-resample bootstrapping. Sweeping thresholds over the full dataset and reporting the maximum F1 selects a parameter on the evaluation data and is optimistically biased; we therefore report F1 from **stratified K-fold cross-validation**, choosing the threshold on training folds and measuring on held-out folds (mean ± standard deviation). Given the dataset size, confidence intervals are wide, and overlapping intervals are treated as failure to distinguish two signals rather than as a ranking.

Results are additionally broken down **by retrieval condition**, to characterise how each signal degrades as retrieval quality falls rather than reporting a single aggregate.

**Cost.** Per-signal wall-clock latency is measured under identical conditions and reported alongside accuracy; the primary figure plots AUROC against mean per-query latency on a log scale.

**Reference points.** Two additional arms bound the comparison: a cross-validated logistic regression over all three signals (does combining them beat any one alone?), and an LLM-as-judge baseline — the same three-point rubric given to the local 7B generator, and optionally to a frontier model via a free-tier API. The judge arms supply the expensive end of the cost curve. Judge-versus-human agreement is reported, since a judge baseline is only a meaningful ceiling if it agrees with human annotators in the first place; §6 shows that on one label category it does not.

The judge is run under **two prompt variants** that differ only in how prominently the abstention rule is stated. This is not prompt tuning to maximise a score: both are reported, and the comparison is itself a result (§6). Selecting whichever prompt scored better and reporting only that would convert a finding about judge reliability into a hidden researcher degree of freedom.

---

## 4. Experimental Setup

*[DRAFT — confirm final counts once the run completes]*

| Component | Choice | VRAM |
|---|---|---|
| Hardware | Single NVIDIA RTX 3060, 6 GB VRAM | — |
| Retriever | `sentence-transformers/all-MiniLM-L6-v2` | ~0.4 GB |
| Generator | Qwen2.5-7B-Instruct, Q4_K_M 4-bit, via Ollama | ~4.7 GB |
| NLI model | `cross-encoder/nli-deberta-v3-base` (~184M params) | ~0.8 GB |
| Judge baseline | Gemini 2.5 Flash (free tier) + local Qwen2.5-7B | — |
| Corpus | ~50 arXiv abstracts, 6 related queries, `cat:cs.CL` | — |
| QA pairs | ~150 (3 question templates × 50 abstracts) | — |
| Conditions | Well-supported / partially supported / poorly supported | — |
| Self-consistency samples | N = 5 at temperature 0.7 | — |
| Total generations | 900 (150 × [1 greedy + 5 sampled]) | — |

**Throughput caveat.** The laptop dGPU used for these experiments is firmware-limited to a 20 W board power budget against an 80 W default, holding the SM clock at 210 MHz (of 2100 MHz) even at 100% utilisation. A raw fp16 matrix-multiply benchmark measures 2.96 TFLOPS against the ~15–20 TFLOPS expected of this part. Generation therefore runs at roughly 310 s per query (six generations each) rather than the ~50 s the hardware should support. This affects wall-clock cost only, not any reported accuracy, and the per-signal latencies in §5 are measured on the same machine so their *relative* ordering is unaffected. Absolute latencies should be read as an upper bound.

**Memory discipline.** The 6 GB budget cannot hold the generator and the scoring models simultaneously. Models are therefore loaded strictly one at a time: the generator is evicted from VRAM (via Ollama's `keep_alive`) before the embedding and NLI models are loaded, and each scoring model is explicitly freed before the next. This constraint shapes the pipeline into discrete stages that communicate through files on disk, which has the incidental benefit of making every stage independently re-runnable.

**Reproducibility.** Random seeds are fixed throughout; the greedy pass is decoded at temperature 0 and the five sampled passes use fixed, distinct seeds. The resolved configuration of every run is written to disk, as are the exact arXiv queries and fetch date — arXiv relevance ranking drifts over time, so the corpus file itself, not the query, is the artifact of record. All intermediate outputs are stored as line-delimited JSON so that an interrupted run loses at most one record and can resume.

---

## 5. Results
*[Fill in after experiments — tables/charts, minimal prose]*

Planned reporting, in order:
1. **Dataset composition** — final QA-pair count, class balance after binarisation, retrieval hit rate (computed over genuinely retrieved rows only; forced-distractor rows are excluded, since their miss is a manipulation rather than a retriever failure), and abstention rate per condition.
2. **Inter-annotator agreement** — raw agreement, Cohen's kappa, linearly-weighted kappa, confusion matrix.
3. **Main table** — AUROC and AUPRC per signal with bootstrap CIs, cross-validated F1/precision/recall, and mean per-query latency. Includes the near-free baseline, the combined logistic-regression arm, and both judge arms.
4. **Per-condition breakdown** — the same metrics computed within well-, partially-, and poorly-supported subsets.
5. **Headline figure** — AUROC against per-query latency (log scale).

## 6. Discussion
*[Fill in after Results — which signal wins, accuracy-cost tradeoff, connection to prior work]*

Two results are worth interpreting carefully regardless of how the numbers land:

**Self-consistency should be expected to be condition-dependent, not uniformly weak.** When relevant context is present, all five samples are conditioned on the same passage and will agree whether or not the answer overreaches — so consistency carries little information. When context is irrelevant, the model must invent, and inventions diverge across samples. If the per-condition breakdown shows this pattern, self-consistency is not a weak signal but a *narrow* one, useful specifically for detecting retrieval failure. That is a more useful conclusion for practitioners than an aggregate ranking.

**Log-probability measures fluency, not grounding.** A model can be confidently wrong, and copied text scores high regardless of whether copying was appropriate. We expect this signal to underperform, and report it as an informative baseline rather than a contender.

**Retrieval failure mostly produces refusal, not hallucination.** In a preliminary scan of generated answers, roughly 69% of poorly-supported items elicited an explicit mismatch response — the model observing that the supplied context concerns a different paper — against 0% of well-supported items. This is a direct consequence of naming the target paper in the question (§3.2): the model can compare the requested title against the context and detect the substitution, which the earlier deictic phrasing made impossible. The consequence is that our strongest manipulation induces *appropriate abstention* far more often than fabrication, which compresses the positive class and is the main reason abstention handling is an explicit analysis flag rather than a fixed convention.

Within those mismatch responses we observe three distinct behaviours, which the labelling guideline treats differently: clean refusal; refusal accompanied by speculation extrapolated from the title given in the question rather than from the context; and — most awkward for this study — an accurate summary of the *supplied* paper instead of the requested one. The last is fully grounded in its context and will be scored as such by every signal here, yet it does not answer the question at all. Grounding-based detection is by construction blind to this failure mode, which suggests that faithfulness metrics alone are an incomplete safety story for RAG.

**The LLM judge does not follow its own rubric on abstentions.** Our judge prompt states explicitly that an answer which correctly declines to answer counts as *supported*, since it asserts nothing the context fails to back. Across the 40 refusal-style answers in our data, the judge applied that rule **zero times**, returning *unsupported* on 39 and *partially supported* on one. To separate a genuine property of the judge from an artefact of our prompt, we re-ran the full set with a second prompt in which the rubric is unchanged but the abstention rule is promoted out of the label definition into its own paragraph, given three worked examples, and stated as an explicit prohibition. Compliance moved from 0/40 to 1/40. Verdicts on the 110 non-refusal answers were essentially unchanged (79 → 78 *supported*), confirming that the second prompt did not broadly alter behaviour but specifically failed to move the cases its added rule governs.

We therefore treat this as a measured limitation of LLM-as-judge rather than a prompt-engineering shortfall: on a category where the rubric and the model's prior disagree, the prior wins, and additional instruction does not recover it. This has a direct bearing on the paper's framing. The usual argument for cheap signals is cost; this suggests a second axis. A prompted judge can misread a rubric, and does so silently and systematically, whereas an NLI cross-encoder has no rubric to misread — it emits an entailment probability, and the mapping from that probability to a decision is ours, inspectable and adjustable after the fact. Where a judge's failure is invisible without exactly the kind of audit we performed here, a threshold on a mechanical signal is not.

## 7. Limitations

**The NLI signal is partly circular.** Our annotation guideline asks whether the context supports the answer's claims — definitionally close to what an NLI model computes. A strong result for this signal should therefore be framed as a *cost* finding (a 184M-parameter cross-encoder approximating a judge at a fraction of the cost) rather than as a discovery that entailment predicts groundedness. It also means the NLI signal and the human labels are not fully independent measurements.

**Dataset size.** With roughly 150 QA pairs, confidence intervals on AUROC are wide and small differences between signals are not resolvable. We report intervals rather than point estimates for this reason, and treat overlapping intervals as inconclusive.

**Single domain, single task format.** All items are closed-domain QA over scientific abstracts in one ML/NLP area, with three fixed question templates. Abstracts are short, self-contained and unusually well-structured compared with the messy documents typical of production RAG; results may not transfer to long or heterogeneous sources.

**Single generator.** Signals are measured against one 7B model at one quantisation level. Signal behaviour plausibly depends on model scale and calibration — a smaller model hallucinates more freely, a larger one more convincingly — and we do not vary this.

**Ground truth is a two-annotator consensus.** Both annotators are authors, and both are familiar with the corpus domain. Agreement statistics are reported as a rigour check, but consensus among two non-independent annotators is not an external gold standard.

**Binarisation is a choice, not a fact.** *Partially supported* is a genuinely intermediate category, and mapping it to either class is defensible. We default to treating it as unsupported and report the alternative mapping as a robustness check, but any headline number depends on this decision.

**Abstention handling.** Correct refusals are counted as supported by default on the grounds that they assert nothing ungrounded. This is a judgement, and it interacts with the poorly-supported condition specifically: a model that refuses often will appear to hallucinate less without being any better at using context.

**Latency is measured on one machine, and that machine is power-limited.** Cost comparisons reflect a single consumer GPU running at roughly a fifth of its rated throughput (§4). Relative ordering across signals is unaffected, since all are measured under identical conditions, but absolute latencies should be treated as upper bounds rather than representative figures.

**Claim granularity limits the entailment signal.** We decompose answers into sentences, but LLM sentences are compound and NLI marks an entire sentence non-entailed if any clause is unsupported. This is why the strict min-over-claims aggregation collapses (§3.5) and why even the mean variant is likely pessimistic on correct answers. Atomic-fact decomposition would give a cleaner measurement; we did not implement it.

**The poorly-supported condition induces less hallucination than intended.** Because the model can detect the paper substitution, that condition yields a high rate of appropriate refusal rather than fabrication (§6). Our positive class is therefore smaller and less adversarial than the design anticipated, and results should not be read as characterising signal behaviour under *undetectable* retrieval failure — a harder and arguably more realistic setting.

## 8. Conclusion & Future Work
*[Fill in]*

## References

*Style: APA 7th. Author lists, venues and years below were checked against the
ACL Anthology / publisher records. Items marked ⚠ still need page numbers or a
DOI confirmed before submission — do not copy those fields from memory.*

He, P., Liu, X., Gao, J., & Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with
disentangled attention. *International Conference on Learning Representations
(ICLR 2021).*

Laban, P., Schnabel, T., Bennett, P. N., & Hearst, M. A. (2022). SummaC:
Re-visiting NLI-based models for inconsistency detection in summarization.
*Transactions of the Association for Computational Linguistics, 10.* ⚠ *pages*

Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for
categorical data. *Biometrics, 33*(1). ⚠ *pages — verify before submission*

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D.
(2020). Retrieval-augmented generation for knowledge-intensive NLP tasks.
*Advances in Neural Information Processing Systems, 33.* ⚠ *pages*

Manakul, P., Liusie, A., & Gales, M. J. F. (2023). SelfCheckGPT: Zero-resource
black-box hallucination detection for generative large language models.
*Proceedings of the 2023 Conference on Empirical Methods in Natural Language
Processing*, 9004–9017. Singapore.

Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P. W., Iyyer, M.,
Zettlemoyer, L., & Hajishirzi, H. (2023). FActScore: Fine-grained atomic
evaluation of factual precision in long form text generation. *Proceedings of
the 2023 Conference on Empirical Methods in Natural Language Processing.*
⚠ *pages; confirm full author list*

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using
Siamese BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods
in Natural Language Processing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP)*, 3982–3992. Hong Kong, China.

⚠ *Still to add: a citation for Qwen2.5 (the generator) and for the
`cross-encoder/nli-deberta-v3-base` checkpoint — cite the model card or its
associated report, not just the HuggingFace URL.*