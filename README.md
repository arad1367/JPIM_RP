[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pejman-ebrahimi-4a60151a7/)
[![HuggingFace](https://img.shields.io/badge/🤗_Hugging_Face-FFD21E?style=for-the-badge)](https://huggingface.co/arad1367)
[![University](https://img.shields.io/badge/University-00205B?style=for-the-badge&logo=academia&logoColor=white)](https://www.uni.li/pejman.ebrahimi?set_language=en)

# JPIM_RP — Guardrails, Refusal Rhetoric, and User Backlash

Replication package for the field study of user responses to LLM refusals, based
on [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m).

```bash
git clone https://github.com/arad1367/JPIM_RP.git
cd JPIM_RP
```

> **Status: manuscript under review.** This repository contains the analysis code
> only. Result tables, figures, and the manuscript are withheld until the paper
> is accepted and will be added here on publication.

---

## Data availability

**The coded datasets are not hosted in this repository.** The source corpus was
deliberately left uncleaned by its authors, so the request and follow-up fields
contain unsafe, offensive, and explicit language — these are precisely the
exchanges in which guardrails trigger, and removing them would destroy the
phenomenon under study. GitHub's content filters block a commit of files in this
form, so the derived datasets are distributed on request instead.

The source corpus is publicly available and can be obtained directly:

**https://huggingface.co/datasets/lmsys/lmsys-chat-1m**

Access is gated by the corpus authors and requires accepting their terms. With
that access, `00_build_corpus.py` in this repository regenerates the base dataset
from scratch, and `01_build_dataset.py` regenerates the analysis dataset from it.
Every stage is deterministic, so the reconstruction is exact.

The two derived files — `Wasko_Final_Dataset_FINAL_CLEAN_1507.csv` (51,186
episodes, 38 variables) and `Wasko_Final_Dataset_ENRICHED.csv` (the same rows
with sentiment, toxicity, and control variables appended, 52 variables) — are
available to reviewers and editors on request. Please contact the corresponding
author using the addresses below.

The analysis sample is the 12,185 episodes where `is_actual_refusal` is `TRUE`.

---

## Contents

| File | Description |
|---|---|
| `00_build_corpus.py` | Screens the 1M-conversation corpus for refusal episodes, codes the fourteen rhetorical features, runs the classification tasks, and applies quality control. Produces the base dataset. |
| `01_build_dataset.py` | Adds follow-up sentiment and request toxicity, builds the control variables, and exports coding-validation samples. |
| `02_analysis.py` | Descriptives, the focal logistic regression models, predictive margins, and the robustness models with additional controls. |
| `03_supplementary.py` | Identification diagnostics: per-model cell counts, restricted samples, random-intercept and GEE estimators, cluster bootstrap, leave-one-model-out, and alternative outcome definitions. |
| `requirements.txt` | Python dependencies for scripts 01–03. |

---

## Requirements

Python 3.10–3.13.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

On Python 3.13, upgrade `setuptools` before installing — 3.13 no longer bundles
it, and several packages fail to build without it.

`00_build_corpus.py` additionally requires `datasets` and `huggingface_hub`, plus
a CUDA GPU. It was run on an NVIDIA A100.

Scripts 01 and 03 benefit from a GPU but run on CPU. Script 02 needs no GPU and
completes in under a minute.

---

## Reproducing the results

**Step 1 — build the corpus from LMSYS-Chat-1M**

```bash
python 00_build_corpus.py
```

Set `TARGET_DIR` at the top of the file first. Requires a HuggingFace token with
access to `lmsys/lmsys-chat-1m`. Several hours on a GPU; each stage checkpoints,
so an interrupted run resumes. Output: `Wasko_Final_Dataset_FINAL_CLEAN.csv` and
`Wasko_Final_Dataset_FINAL_FLAGGED.csv` (the rows removed by quality control).

**Step 2 — add sentiment, toxicity, and controls**

```bash
python 01_build_dataset.py \
    --input Wasko_Final_Dataset_FINAL_CLEAN.csv \
    --outdir output
```

Roughly 15–30 minutes on a GPU. Use `--scope confirmed` to score only the
analysis sample when running on CPU. Output: `Wasko_Final_Dataset_ENRICHED.csv`,
coding-validation samples, and a data-integrity report.

**Step 3 — run the analysis**

```bash
python 02_analysis.py \
    --input output/Wasko_Final_Dataset_ENRICHED.csv \
    --outdir results

python 03_supplementary.py \
    --input output/Wasko_Final_Dataset_ENRICHED.csv \
    --outdir results_supplementary
```

Script 02 verifies its own output against the published coefficients and prints a
pass/fail line for each. Script 03's cluster bootstrap defaults to 1,000
replications; change it with `--boot`.

---

## Determinism

Every stage is deterministic and reproduces exactly on rerun:

- Pattern matching is plain case-insensitive substring matching.
- The classifier (`Qwen2.5-3B-Instruct`) uses greedy decoding.
- Sentiment and toxicity models run in inference mode with fixed weights.
- Sampling for validation files and the cluster bootstrap uses a fixed seed.

Applying the feature dictionaries in `00_build_corpus.py` to the derived dataset
reproduces the stored flags for all fourteen features across the 12,185 analysis
episodes, with a single exception arising from an apostrophe variant.

---

## Known data issues

1. **`is_actual_refusal` is stored as text.** Two of the 51,186 rows carry a
   malformed value: one is `X`, one contains leaked generation text. Neither is
   in the analysis sample, so no reported result is affected. Both are listed in
   the data-integrity report produced by `01_build_dataset.py`.

2. **`prompt_category` and `refusal_type` are deterministically related.**
   `value_violation` corresponds exactly to `value_based_refusal`. The two can
   never enter the same model.

3. **`value_violation` is a residual category.** It is the mapped form of the
   classifier's `not_instrumental` output, not a substantive judgment that the
   request violated a value.

4. **`uncategorized` is a parsing residual.** It is assigned when the
   classifier's output matches none of the permitted labels, not when the
   follow-up was itself unclassifiable in substance.

5. **The screening dictionary overlaps two coded features.** `as an ai`,
   `i apologize`, and `i'm sorry` are both screening markers and feature
   markers, so the corpus is selected partly on the AI-identity disclaimer and
   the apology. Reported base rates for these two features are conditional on
   the screen and are not population prevalence estimates. Comparisons between
   refusals with and without a feature are unaffected.

---

## Ethics and privacy

Person names in the source corpus were replaced with placeholders (`NAME_1`,
`NAME_2`, …) by the corpus authors before release. No additional
de-identification was applied. All users of the original platform consented to
research use of their conversations through the platform's terms of use. This
study analyzes only publicly released data and involved no interaction with
human participants.

---

## Citation

Citation details will be added on publication.

Source corpus:

> Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Li, Z.,
> Lin, Z., Xing, E. P., Gonzalez, J. E., Stoica, I., & Zhang, H. (2023).
> LMSYS-Chat-1M: A large-scale real-world LLM conversation dataset.
> arXiv:2309.11998.

Classifier:

> Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., et al. (2024).
> Qwen2.5 technical report. arXiv:2412.15115.

---

## Contact

**Pejman Ebrahimi**

- pejman.ebrahimi@uni.li
- pejman.ebrahimi77@gmail.com
- GitHub: [arad1367](https://github.com/arad1367)
- Hugging Face: [arad1367](https://huggingface.co/arad1367)

Requests for the derived datasets, and questions about the code, are welcome at
either address.
