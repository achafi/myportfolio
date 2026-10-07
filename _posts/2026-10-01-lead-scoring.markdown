---
title : "Lead scoring with XGBoost and SHAP"
date : 2026-10-01
tags : [machine learning, XGBoost, SHAP, Python, scikit-learn]
header :
  image : ""
excerpt : "Machine learning, XGBoost, SHAP"
---
[Source code](https://github.com/achafi/lead_scoring)

# Lead scoring: predicting conversion and explaining why
*The objective of this project is to predict the probability that a marketing lead will convert, turn that probability into a prioritization score, group the leads into Low / Medium / High propensity segments, and explain every prediction with SHAP. It is a complete applied ML workflow — data, EDA, preprocessing, model comparison, cross-validation, explainability, segmentation and a scoring script — on a real public dataset.*

## Introduction

Sales teams do not need a model that is 2% more accurate in the abstract; they need a ranked list of leads so the team calls the right people first. That framing changes the whole project: the metric that matters is how well the model *ranks* converted leads (PR-AUC), the output is a score plus segments, and the explanations matter as much as the predictions because someone has to defend the prioritization to the business.

## Dataset

The X Education lead dataset from Kaggle: **9,240 rows and 37 columns**, with `Converted` as the binary target. `Prospect ID` and `Lead Number` are unique identifiers and there are no exact duplicate rows. Blank cells and `Select` placeholders are treated as missing / not selected — an important detail, because in this dataset "Select" means the user never answered, not the literal string. Several fields carry substantial missingness.

## Workflow

**Data → EDA → Preprocessing → Model training → Model evaluation → Cross-validation → SHAP → Lead scoring → Segmentation**

## Model comparison

Six candidates were compared: a majority-class baseline, logistic regression, random forest, histogram gradient boosting, XGBoost and LightGBM. **XGBoost** was selected for the highest mean PR-AUC in five-fold stratified cross-validation on the training split (0.897 versus 0.895 for LightGBM).

The README is refreshingly honest about how close it was: the difference is small, and LightGBM had slightly lower fold variation and marginally higher test PR-AUC, recall and F1. The selection rule was fixed in advance (mean CV PR-AUC), which is exactly how it should be settled.

## Results

Final test set:

| Metric | Result |
|---|---:|
| Precision | 0.759 |
| Recall | 0.837 |
| F1 | 0.796 |
| ROC-AUC | 0.914 |
| PR-AUC | 0.871 |

Five-fold stratified cross-validation:

| Metric | Mean | Std |
|---|---:|---:|
| ROC-AUC | 0.928 | 0.005 |
| PR-AUC | 0.897 | 0.012 |

Precision is the share of leads flagged by the model that converted; recall is the share of converted leads the model identified; ROC-AUC measures separation across thresholds, while PR-AUC focuses on how well the model ranks the converted leads — the number that matters for a scoring use case.

## Explainability with SHAP

SHAP explains the selected XGBoost model in raw-score (log-odds) units; positive contributions push the output toward conversion. The largest overall contributors were **Last Activity**, **Total Time Spent on Website** and **Lead Origin**, followed by the Asymmetrique activity score, course preference and occupation. For one high-score lead, `Lead Origin_Lead Add Form`, website time and the activity score pushed the prediction toward conversion.

SHAP describes model associations, not causes — an important line to keep in mind before anyone in a sales meeting starts talking about what "drives" conversions.

## Lead scoring and segmentation

The model's `final_score` is its estimated conversion probability: a score of 0.72 means a model estimate of 72% conversion probability, not a guarantee. Test leads were split into segments at the one-third and two-thirds quantiles of their scores (cutoffs ≈ **0.126** and **0.723**):

| Segment | Leads | Share | Avg predicted prob. | Actual conversion rate |
|---|---:|---:|---:|---:|
| Low | 625 | 33.8% | 4.5% | 3.4% |
| Medium | 607 | 32.8% | 37.4% | 29.0% |
| High | 616 | 33.3% | 90.5% | 83.6% |

Observed conversion rates increase monotonically from Low to High on the held-out set — the practical check that the scoring actually separates leads. Operationally: high propensity gets priority follow-up, medium gets normal follow-up, low gets nurturing, and the team picks the real cutoffs based on its capacity.

## Project structure

```text
lead_scoring/
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Modeling.ipynb
├── reports/
│   └── lead_scoring_project_report.md
├── models/
│   ├── final_model.joblib
│   └── model_metadata.json
├── src/
│   └── scoring.py
└── data/                  # dataset downloaded from Kaggle (not committed)
```

Scoring new leads without retraining:

```bash
.venv/bin/python src/scoring.py --input new_leads.csv --output scored_leads.csv
```

The output preserves the input columns and adds `final_score` and `lead_segment` (`--no-segmentation` returns scores only). The script loads the saved pipeline — it does not train.

## Limitations

- Results depend on this public dataset and may not transfer to another business or to future leads.
- Predicted probabilities are model estimates; calibration should be checked before treating scores as precise probabilities.
- Quantile-based segments are relative to the test-score distribution and may need re-cutting on future data.
- The timing of `Last Activity` and the asymmetrique fields is not verified — they must be confirmed available at the intended scoring point, or the model is leaking information it would not have in production.

## Conclusion

The full pipeline — model comparison with a pre-declared selection metric, cross-validation rather than a single split, SHAP explanations, and quantile-based segmentation validated against actual conversion rates — makes this a realistic blueprint for a lead-prioritization system, with the caveats written down instead of hidden.

[Source code](https://github.com/achafi/lead_scoring)
