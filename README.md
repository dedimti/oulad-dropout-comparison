# Comparative Analysis of Deep Learning and Gradient Boosting for Student Dropout Prediction (OULAD)

Reproducible code for the study comparing five machine learning models — Random Forest,
XGBoost, a standalone CNN, a standalone LSTM, and a hybrid CNN-LSTM — for student dropout
prediction on the **Open University Learning Analytics Dataset (OULAD)**, with SHAP-based
interpretability and a three-tier Early Warning System (EWS).

> **Key finding:** On the full real OULAD dataset (32,593 students), XGBoost achieved the
> highest performance (Accuracy 86.69%, AUC-ROC 0.9440) and **significantly outperformed**
> the more complex hybrid CNN-LSTM (Accuracy 84.14%, AUC-ROC 0.9250; paired t-test p=0.033).
> SHAP analysis identified the temporal span of LMS activity (`active_span`) as the dominant
> predictor. For tabular educational data, simpler gradient boosting models can match or
> exceed deep learning while remaining more interpretable.

This repository accompanies the manuscript submitted to *JUITA: Jurnal Informatika*.

---

## Repository structure

```
.
├── src/
│   ├── 01_download_oulad.py      # download & verify the OULAD dataset
│   ├── 02_run_experiment.py      # full pipeline: features → 5 models → significance → SHAP → EWS
│   └── 03_generate_figures.py    # regenerate publication-quality figures from results
├── figures/                      # generated figures (Fig. 2–7)
├── results/                      # output tables/JSON (created after running the pipeline)
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

## Quickstart

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. download the real OULAD dataset (CC-BY 4.0)
python src/01_download_oulad.py
#    if the automatic download fails, see "Dataset" below for manual steps

# 3. run the full experiment (training + significance test + SHAP + EWS)
python src/02_run_experiment.py

# 4. (optional) regenerate the publication-quality figures
python src/03_generate_figures.py
```

All outputs (tables as CSV, a JSON summary, and a paste-ready `RESULTS_FOR_PAPER.md`)
are written to `results/` (or `outputs/` depending on the script's working directory).

## Methodology summary

- **Dataset:** real OULAD, 32,593 student-presentations, 21 engineered features
  (7 demographic + 4 academic + 10 LMS behavioral).
- **Label:** `final_result == "Withdrawn"` → dropout (positive class).
- **Validation:** 5-fold stratified cross-validation, `random_state=42`.
- **Imbalance handling:** SMOTE applied **only within each training fold** to prevent leakage.
- **Models:** Random Forest, XGBoost, CNN, LSTM, CNN-LSTM (identical preprocessing & folds).
- **Significance:** paired t-test + Wilcoxon signed-rank test on per-fold AUC, with 95% CI.
- **Interpretability:** SHAP KernelExplainer (model-agnostic).
- **EWS:** three-tier risk stratification (low / medium / high) at P=0.30 and P=0.60.

## Dataset

The OULAD dataset is **not redistributed** in this repository. It is publicly available
under a CC-BY 4.0 license:

- Official: https://analyse.kmi.open.ac.uk/open_dataset
- Mirror (UCI): https://archive.ics.uci.edu/dataset/349/

After downloading, place the seven CSV files (`studentInfo.csv`, `studentVle.csv`,
`studentAssessment.csv`, `assessments.csv`, `studentRegistration.csv`, `courses.csv`,
`vle.csv`) inside a `data/` folder next to the scripts.

**Dataset citation:**

> J. Kuzilek, M. Hlosta, and Z. Zdrahal, "Open University Learning Analytics dataset,"
> *Scientific Data*, vol. 4, art. 170171, 2017, doi: 10.1038/sdata.2017.171.

## Reproducibility note

Deep-learning results may vary slightly across runs and hardware due to non-deterministic
GPU/CPU operations, even with a fixed random seed. The reported figures reflect a single
5-fold cross-validation run; minor deviations (typically < 0.5% in AUC) are expected and do
not affect the qualitative conclusions.

## Citation

If you use this code, please cite the manuscript (details to be updated upon publication):

> D. Irawan and Sudarmaji, "Comparative Analysis of Deep Learning and Gradient Boosting
> Models for Student Dropout Prediction Using the OULAD Behavioral Dataset," *JUITA: Jurnal
> Informatika*, 2026 (submitted).

## License

Code released under the MIT License (see `LICENSE`). The OULAD dataset is licensed
separately under CC-BY 4.0 by its original authors.

## Authors

- **Dedi Irawan** (corresponding) — Faculty of Computer Science, Universitas Muhammadiyah Metro — dedimti@ummetro.ac.id
- **Sudarmaji** — Faculty of Computer Science, Universitas Muhammadiyah Metro — majidarma5022@gmail.com
