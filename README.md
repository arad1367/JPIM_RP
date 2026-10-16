[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pejman-ebrahimi-4a60151a7/)
[![HuggingFace](https://img.shields.io/badge/🤗_Hugging_Face-FFD21E?style=for-the-badge)](https://huggingface.co/arad1367)
[![University](https://img.shields.io/badge/University-00205B?style=for-the-badge&logo=academia&logoColor=white)](https://www.uni.li/pejman.ebrahimi?set_language=en)

# JPIM_RP — Guardrails, Refusal Rhetoric, and User Backlash

Replication package for the field study of how users respond to refusals by large
language models, based on
[LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m).

```bash
git clone https://github.com/arad1367/JPIM_RP.git
cd JPIM_RP
```

> **Status: manuscript under review.** This repository contains the analysis
> code. Result tables, figures and the manuscript are withheld until the paper
> is accepted and will be added here on publication.

---

## The study in one paragraph

One million human–LLM conversations were screened for refusals, yielding 12,170
verified refusal episodes across 25 deployed systems and 95 languages. For each
episode we code three rhetorical features of the refusal — whether it declares
the system to be an AI, whether it justifies the boundary in moral terms, and
whether it apologises — and whether the user's next turn resists the refusal.
The analysis asks whether declaring AI identity provokes resistance, and whether
that depends on the rhetoric surrounding it.

---

## Data availability

**The coded datasets are not hosted here.** The source corpus was deliberately
left uncleaned by its authors, so the request and follow-up fields contain
unsafe, offensive and explicit language — these are precisely the exchanges in
which guardrails trigger, and removing them would destroy the phenomenon under
study. GitHub's content filters block a commit of files in this form.

The source corpus is public and gated by its authors:

**https://huggingface.co/datasets/lmsys/lmsys-chat-1m**

With that access the pipeline below regenerates every derived dataset from
scratch. Every stage is deterministic, so the reconstruction is exact. The
derived datasets are also available directly from the corresponding author on
request.

---

## Pipeline

Run in order. Scripts 1 to 4 build the data; 5 to 7 produce the results.

| # | Script | Reads | Writes |
|---|---|---|---|
| 1 | `01_build_corpus.py` | LMSYS-Chat-1M (HuggingFace) | `Wasko_Final_Dataset_FINAL_CLEAN.csv` |
| 2 | `02_score_sentiment_toxicity.py` | output of 1 | `Wasko_Final_Dataset_ENRICHED.csv` |
| 3 | `03_recode_residual_cases.py` | output of 2 | `Wasko_Final_Dataset_RECODED.csv` |
| 4 | `04_score_severity.py` | output of 3 | `Wasko_Final_Dataset_ANALYSIS.csv` |
| 5 | `05_analysis.py` | output of 4 | Tables 1–5, Figure 2 margins |
| 6 | `06_supplementary.py` | output of 4 | identification diagnostics |
| 7 | `07_figure2.py` | margins from 5 | Figure 2 (PNG, PDF, TIFF) |

What each stage does:

1. **Build corpus.** Screens all one million conversations against the refusal
   marker dictionary, assembles refusal episodes, codes fourteen rhetorical
   features by pattern matching, classifies refusal validity, request category
   and user follow-up behaviour with Qwen2.5-3B-Instruct, and applies quality
   control.
2. **Sentiment and toxicity.** Scores follow-up sentiment and request toxicity
   with pretrained multilingual classifiers, and builds the control variables.
   Also exports stratified coding-validation samples.
3. **Residual cases.** Applies the verified re-coding of the 163 episodes the
   second-stage classifier could not label, and removes the episodes containing
   an automated platform moderation notice in the user-turn position.
4. **Severity.** Scores how harmful compliance would be, using four independent
   yes–no threshold questions put to Qwen2.5-14B-Instruct. Runs self-tests and a
   piloted quality gate before scoring the full sample, and writes nothing if
   either fails.
5. **Analysis.** Descriptives, the focal logistic regressions, predictive
   margins, the robustness models, and follow-up sentiment as a secondary
   outcome.
6. **Supplementary.** Per-system cell counts, restricted samples, GEE, cluster
   bootstrap, leave-one-system-out, language splits, alternative outcome
   definitions and alternative severity specifications.
7. **Figure.** Draws Figure 2 from the margins written by step 5.

Paths default to the layout below; override any of them with `--input` and
`--outdir`.

```
Wasko_JPIM/
├── data/      produced by steps 1–4
└── output/
    ├── results/
    ├── results_supplementary/
    └── figures/
```

---

## Requirements

Python 3.10–3.13.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Steps 1, 2 and 4 require a CUDA GPU and were run on an NVIDIA A100. Steps 5 to 7
need no GPU and complete in a few minutes. Step 1 additionally requires
`datasets` and `huggingface_hub`, plus a token with access to the gated corpus.

Most users should start at step 5 with the derived dataset obtained from the
corresponding author, since steps 1 to 4 require gated corpus access and several
hours of GPU time.

---

## Reproducibility

Every stage is deterministic and reproduces exactly on rerun:

- Rhetorical features are coded by plain case-insensitive substring matching.
- The classification model uses greedy decoding.
- Severity is scored by reading the model's probability over the two answer
  tokens at a single forward pass, with no sampling. Each question is asked in
  both answer orders and the probabilities averaged, so option position cannot
  drive the rating.
- Sentiment and toxicity use pretrained classifiers in inference mode.
- The cluster bootstrap and all sampling use a fixed seed.

Step 4 verifies itself before doing any work: it checks that scores are
invariant to batching, that it recovers known answers including three cases
designed to catch confounds, and that it passes three pre-specified quality
gates on a stratified pilot. It exits without writing if any check fails.

---

## Known data issues

1. **`is_actual_refusal` is stored as text.** Two of the 51,186 screened rows
   carry a malformed value. Neither is in the analysis sample.
2. **Fifteen episodes contain an automated platform moderation notice** in the
   user-turn position rather than a user message. They record no user behaviour
   and are excluded, giving an analysis sample of 12,170.
3. **`prompt_category` and `refusal_type` are deterministically related.**
   `value_violation` corresponds exactly to `value_based_refusal`, so the two
   never enter the same model.
4. **`value_violation` is a residual category.** It is the mapped form of the
   classifier's `not_instrumental` output and consists largely of ordinary
   conversational exchanges, not of requests judged to violate a value.
5. **The screening dictionary overlaps two coded features.** `as an ai`,
   `i apologize` and `i'm sorry` are both screening markers and feature markers,
   so the corpus is selected partly on the AI-identity disclaimer and the
   apology. Reported base rates for those two features are conditional on the
   screen and are not population prevalence estimates. Comparisons between
   refusals with and without a feature are unaffected.

---

## Ethics and privacy

Person names in the source corpus were replaced with placeholders (`NAME_1`,
`NAME_2`, …) by the corpus authors before release. No additional
de-identification was applied. All users of the original platform consented to
research use of their conversations through the platform's terms of use. This
study analyses only publicly released data and involved no interaction with
human participants.

---

## Citation

Citation details will be added on publication.

Source corpus:

> Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Li, Z.,
> Lin, Z., Xing, E. P., Gonzalez, J. E., Stoica, I., & Zhang, H. (2023).
> LMSYS-Chat-1M: A large-scale real-world LLM conversation dataset.
> arXiv:2309.11998.

Classification and severity scoring:

> Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., et al. (2024).
> Qwen2.5 technical report. arXiv:2412.15115.

---

## Contact

**Pejman Ebrahimi**

- pejman.ebrahimi@uni.li
- pejman.ebrahimi77@gmail.com
- GitHub: [arad1367](https://github.com/arad1367)
- Hugging Face: [arad1367](https://huggingface.co/arad1367)
