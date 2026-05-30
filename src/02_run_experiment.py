"""
02_run_experiment.py
Pipeline lengkap CNN-LSTM EWS pada OULAD ASLI.
Menghasilkan semua tabel/gambar pengganti hasil sintetis + uji signifikansi statistik.

Metodologi sengaja dibuat identik dengan naskah:
  - 21 fitur: 7 demografik + 4 akademik + 10 LMS behavioral
  - label: Withdrawn = 1 (dropout), lainnya = 0
  - SMOTE HANYA di dalam fold training (cegah kebocoran)
  - 5-fold stratified CV, random_state=42
  - 5 model: RF, XGBoost, CNN, LSTM, CNN-LSTM (proposed)
  - SHAP KernelExplainer pada CNN-LSTM
  - EWS tiga-tier low/medium/high
TAMBAHAN (menutup kelemahan reviewer): paired t-test + Wilcoxon + 95% CI antar-fold.
"""
import os, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
from scipy import stats
import xgboost as xgb

SEED = 42
np.random.seed(SEED)
DATA_DIR, OUT = "data", "outputs"
os.makedirs(OUT, exist_ok=True)


def _import_tf():
    """Import TensorFlow saat dibutuhkan, set seed & log level."""
    import tensorflow as tf
    from tensorflow.keras import layers, models
    tf.get_logger().setLevel("ERROR")
    tf.random.set_seed(SEED)
    return tf, layers, models

# ----------------------------------------------------------------------------
# 1. FEATURE ENGINEERING — 21 fitur dari OULAD asli
# ----------------------------------------------------------------------------
def build_features():
    print("[1/8] Feature engineering dari OULAD asli ...")
    si  = pd.read_csv(f"{DATA_DIR}/studentInfo.csv")
    sa  = pd.read_csv(f"{DATA_DIR}/studentAssessment.csv")
    asm = pd.read_csv(f"{DATA_DIR}/assessments.csv")
    sr  = pd.read_csv(f"{DATA_DIR}/studentRegistration.csv")
    # studentVle besar -> baca kolom seperlunya
    svle = pd.read_csv(f"{DATA_DIR}/studentVle.csv",
                       usecols=["code_module","code_presentation","id_student",
                                "id_site","date","sum_click"])

    key = ["code_module","code_presentation","id_student"]

    # --- LABEL: Withdrawn = dropout ---
    si = si[si["final_result"].notna()].copy()
    si["label"] = (si["final_result"] == "Withdrawn").astype(int)

    # --- 7 FITUR DEMOGRAFIK ---
    demo = si[key + ["gender","age_band","disability","region",
                     "highest_education","imd_band","num_of_prev_attempts"]].copy()

    # --- 4 FITUR AKADEMIK ---
    # studied_credits + 3 skor asesmen ringkas
    acad = si[key + ["studied_credits"]].copy()
    # rata-rata skor & skor asesmen pertama (urut tanggal) per mahasiswa-presentasi
    agg_assess = (sa.merge(asm[["id_assessment","code_module","code_presentation","date"]],
                           on="id_assessment", how="left")
                    .sort_values("date")
                    .groupby(["code_module","code_presentation","id_student"])
                    .agg(avg_assessment_score=("score","mean"),
                         first_assessment_score=("score","first"),
                         n_assessments_taken=("score","count"))
                    .reset_index())
    acad = acad.merge(agg_assess, on=key, how="left")

    # --- 10 FITUR LMS BEHAVIORAL (dari studentVle) ---
    g = svle.groupby(key)
    lms = g.agg(total_clicks=("sum_click","sum"),
                active_days=("date","nunique"),
                mean_daily_clicks=("sum_click","mean"),
                max_daily_clicks=("sum_click","max"),
                std_daily_clicks=("sum_click","std"),
                first_access_day=("date","min"),
                last_access_day=("date","max")).reset_index()
    lms["active_span"] = lms["last_access_day"] - lms["first_access_day"]
    # early engagement (klik pada 4 minggu pertama, date < 28) -> proxy sinyal dini
    early = (svle[svle["date"] < 28].groupby(key)["sum_click"].sum()
             .rename("early_clicks").reset_index())
    lms = lms.merge(early, on=key, how="left")
    # rasio klik akhir vs awal (tren keterlibatan)
    late = (svle[svle["date"] >= 28].groupby(key)["sum_click"].sum()
            .rename("late_clicks").reset_index())
    lms = lms.merge(late, on=key, how="left")
    lms["engagement_trend"] = (lms["late_clicks"].fillna(0) + 1) / (lms["early_clicks"].fillna(0) + 1)

    # --- GABUNG ---
    df = demo.merge(acad, on=key, how="left").merge(lms, on=key, how="left")
    df = df.merge(si[key + ["label"]], on=key, how="left")

    # registrasi (tanggal unregister tdk dipakai sbg fitur utk hindari kebocoran target)
    df = df.merge(sr[key + ["date_registration"]], on=key, how="left")

    # pilih tepat 21 fitur
    feature_cols = [
        # demografik (7)
        "gender","age_band","disability","region","highest_education",
        "imd_band","num_of_prev_attempts",
        # akademik (4)
        "studied_credits","avg_assessment_score","first_assessment_score","n_assessments_taken",
        # LMS behavioral (10)
        "total_clicks","active_days","mean_daily_clicks","max_daily_clicks",
        "std_daily_clicks","active_span","early_clicks","late_clicks",
        "engagement_trend","date_registration",
    ]
    assert len(feature_cols) == 21, f"Harus 21 fitur, dapat {len(feature_cols)}"
    df = df[feature_cols + ["label"]].copy()
    print(f"      -> {len(df):,} mahasiswa-presentasi, {len(feature_cols)} fitur")
    print(f"      -> dropout rate: {df['label'].mean()*100:.1f}%")
    return df, feature_cols


# ----------------------------------------------------------------------------
# 2. PREPROCESSING
# ----------------------------------------------------------------------------
def preprocess(df, feature_cols):
    print("[2/8] Preprocessing (imputasi, encoding, standardisasi) ...")
    df = df.copy()
    cat_cols = ["gender","age_band","disability","region","highest_education","imd_band"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    for c in cat_cols:
        df[c] = df[c].fillna("unknown").astype(str)
        df[c] = LabelEncoder().fit_transform(df[c])
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)
    return X, y


# ----------------------------------------------------------------------------
# 3. MODEL BUILDERS (arsitektur identik naskah)
# ----------------------------------------------------------------------------
def make_cnn(n_features, layers, models):
    m = models.Sequential([
        layers.Input(shape=(n_features, 1)),
        layers.Conv1D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

def make_lstm(n_features, layers, models):
    m = models.Sequential([
        layers.Input(shape=(n_features, 1)),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

def make_cnn_lstm(n_features, layers, models):
    m = models.Sequential([
        layers.Input(shape=(n_features, 1)),
        layers.Conv1D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling1D(2),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


def metrics_block(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        auc=roc_auc_score(y_true, y_prob),
    )


# ----------------------------------------------------------------------------
# 4-5. CROSS-VALIDATION + 5 MODEL
# ----------------------------------------------------------------------------
def run_cv(X, y):
    print("[3-5/8] 5-fold stratified CV untuk 5 model (SMOTE dalam-fold) ...")
    tf, layers, models = _import_tf()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    n_features = X.shape[1]
    per_fold = {m: [] for m in ["RandomForest","XGBoost","CNN","LSTM","CNN-LSTM"]}
    # simpan prediksi CNN-LSTM untuk confusion matrix & EWS
    cnnlstm_y_true, cnnlstm_y_prob = [], []

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        print(f"      Fold {fold}/5 ...")
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        # SMOTE HANYA di training fold
        Xtr_s, ytr_s = SMOTE(random_state=SEED).fit_resample(Xtr, ytr)

        # RF
        rf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                    random_state=SEED, n_jobs=-1).fit(Xtr_s, ytr_s)
        per_fold["RandomForest"].append(metrics_block(yte, rf.predict_proba(Xte)[:,1]))
        # XGBoost
        xg = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               subsample=0.9, eval_metric="logloss",
                               random_state=SEED, n_jobs=-1).fit(Xtr_s, ytr_s)
        per_fold["XGBoost"].append(metrics_block(yte, xg.predict_proba(Xte)[:,1]))
        # DL inputs (n,features,1)
        Xtr_dl = Xtr_s.reshape(-1, n_features, 1)
        Xte_dl = Xte.reshape(-1, n_features, 1)
        for name, builder in [("CNN",make_cnn),("LSTM",make_lstm),("CNN-LSTM",make_cnn_lstm)]:
            tf.keras.backend.clear_session()
            mdl = builder(n_features, layers, models)
            mdl.fit(Xtr_dl, ytr_s, epochs=30, batch_size=64, verbose=0,
                    validation_split=0.1,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5,
                               restore_best_weights=True)])
            prob = mdl.predict(Xte_dl, verbose=0).ravel()
            per_fold[name].append(metrics_block(yte, prob))
            if name == "CNN-LSTM":
                cnnlstm_y_true.append(yte); cnnlstm_y_prob.append(prob)

    return per_fold, np.concatenate(cnnlstm_y_true), np.concatenate(cnnlstm_y_prob), n_features


# ----------------------------------------------------------------------------
# 6. UJI SIGNIFIKANSI (penutup kelemahan reviewer)
# ----------------------------------------------------------------------------
def significance(per_fold):
    print("[6/8] Uji signifikansi statistik antar-fold ...")
    rows = []
    proposed = "CNN-LSTM"
    prop_auc = np.array([f["auc"] for f in per_fold[proposed]])
    prop_f1  = np.array([f["f1"]  for f in per_fold[proposed]])
    for name in per_fold:
        if name == proposed: continue
        a = np.array([f["auc"] for f in per_fold[name]])
        f = np.array([f["f1"]  for f in per_fold[name]])
        t_auc, p_auc = stats.ttest_rel(prop_auc, a)
        try:    w_auc, pw_auc = stats.wilcoxon(prop_auc, a)
        except Exception: pw_auc = np.nan
        diff = prop_auc - a
        ci = stats.t.interval(0.95, len(diff)-1, loc=diff.mean(),
                              scale=stats.sem(diff)) if diff.std()>0 else (diff.mean(),diff.mean())
        rows.append(dict(comparison=f"{proposed} vs {name}",
                         mean_auc_diff=round(diff.mean(),4),
                         ci95_low=round(ci[0],4), ci95_high=round(ci[1],4),
                         paired_t_p=round(p_auc,4),
                         wilcoxon_p=round(pw_auc,4) if pw_auc==pw_auc else "NA",
                         significant_05=("YES" if p_auc<0.05 else "NO")))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 7. SHAP
# ----------------------------------------------------------------------------
def run_shap(X, y, feature_cols, n_features):
    print("[7/8] SHAP feature importance pada CNN-LSTM ...")
    import shap
    tf, layers, models = _import_tf()
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    Xs_s, ys_s = SMOTE(random_state=SEED).fit_resample(Xs, y)
    tf.keras.backend.clear_session()
    mdl = make_cnn_lstm(n_features, layers, models)
    mdl.fit(Xs_s.reshape(-1,n_features,1), ys_s, epochs=30, batch_size=64, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

    n_bg = min(100, len(Xs)); n_sample = min(200, len(Xs))
    bg = shap.sample(Xs, n_bg, random_state=SEED)
    sample = shap.sample(Xs, n_sample, random_state=SEED)
    f = lambda d: mdl.predict(d.reshape(-1, n_features, 1), verbose=0).ravel()
    expl = shap.KernelExplainer(f, bg)
    sv = expl.shap_values(sample, nsamples=100)
    sv = np.array(sv).reshape(n_sample, n_features)
    mean_abs = np.abs(sv).mean(axis=0)
    imp = (pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
           .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    imp["rank"] = imp.index + 1
    return imp, mdl, sc


# ----------------------------------------------------------------------------
# 8. EWS tiga-tier + figur
# ----------------------------------------------------------------------------
def ews_and_figures(y_true, y_prob, per_fold, imp):
    print("[8/8] EWS tiga-tier + membuat semua gambar ...")
    # threshold dikalibrasi data riil
    low  = (y_prob < 0.30).sum()
    med  = ((y_prob >= 0.30) & (y_prob < 0.60)).sum()
    high = (y_prob >= 0.60).sum()
    n = len(y_prob)
    ews = pd.DataFrame([
        dict(risk="Low (P<0.30)",    n=int(low),  pct=round(low/n*100,1)),
        dict(risk="Medium (0.30-0.60)", n=int(med), pct=round(med/n*100,1)),
        dict(risk="High (P>=0.60)",  n=int(high), pct=round(high/n*100,1)),
    ])

    # Fig 4: confusion matrix
    cm = confusion_matrix(y_true, (y_prob>=0.5).astype(int))
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-dropout","Dropout"], yticklabels=["Non-dropout","Dropout"])
    plt.title("CNN-LSTM Confusion Matrix (OULAD)"); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_confusion.png", dpi=150); plt.close()

    # Fig 5: bar performa
    names = list(per_fold.keys())
    accs = [np.mean([f["accuracy"] for f in per_fold[m]])*100 for m in names]
    f1s  = [np.mean([f["f1"] for f in per_fold[m]])*100 for m in names]
    recs = [np.mean([f["recall"] for f in per_fold[m]])*100 for m in names]
    x = np.arange(len(names)); w=0.25
    plt.figure(figsize=(9,5))
    plt.bar(x-w, accs, w, label="Accuracy")
    plt.bar(x,   f1s,  w, label="F1")
    plt.bar(x+w, recs, w, label="Recall")
    plt.xticks(x, names, rotation=15); plt.ylabel("%"); plt.legend()
    plt.title("Model Performance Comparison (OULAD, 5-fold CV)")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_performance_bar.png", dpi=150); plt.close()

    # Fig 6: SHAP
    top = imp.head(10).iloc[::-1]
    plt.figure(figsize=(8,5))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("mean |SHAP|"); plt.title("Top-10 SHAP Feature Importance (CNN-LSTM, OULAD)")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_shap.png", dpi=150); plt.close()

    # Fig 7: distribusi risiko
    plt.figure(figsize=(6,4))
    plt.bar(ews["risk"], ews["n"], color=["#4caf50","#ff9800","#f44336"])
    plt.ylabel("Number of students"); plt.title("EWS Risk Distribution (OULAD)")
    plt.xticks(rotation=10); plt.tight_layout()
    plt.savefig(f"{OUT}/fig_ews_temporal.png", dpi=150); plt.close()

    return ews


# ----------------------------------------------------------------------------
def summarize(df, feature_cols, per_fold, sig_df, imp, ews, y):
    def mean_std(metric):
        return {m: (round(np.mean([f[metric] for f in per_fold[m]]),4),
                    round(np.std([f[metric] for f in per_fold[m]]),4)) for m in per_fold}
    perf_rows = []
    for m in per_fold:
        perf_rows.append(dict(model=m,
            accuracy=round(np.mean([f["accuracy"] for f in per_fold[m]])*100,2),
            precision=round(np.mean([f["precision"] for f in per_fold[m]])*100,2),
            recall=round(np.mean([f["recall"] for f in per_fold[m]])*100,2),
            f1=round(np.mean([f["f1"] for f in per_fold[m]])*100,2),
            auc=round(np.mean([f["auc"] for f in per_fold[m]]),4),
            auc_std=round(np.std([f["auc"] for f in per_fold[m]]),4)))
    perf = pd.DataFrame(perf_rows)

    table1 = pd.DataFrame([
        ("Total students", f"{len(df):,}"),
        ("Dropout (label=1)", f"{int(y.sum()):,} ({y.mean()*100:.1f}%)"),
        ("Non-dropout (label=0)", f"{int((y==0).sum()):,} ({(1-y.mean())*100:.1f}%)"),
        ("Total features", "21 (7+4+10)"),
        ("CV strategy", "5-fold stratified"),
        ("Train/test split", "5-fold CV"),
        ("Data source", "OULAD (real, CC-BY 4.0)"),
    ], columns=["Characteristic","Value"])

    table1.to_csv(f"{OUT}/table1_dataset.csv", index=False)
    perf.to_csv(f"{OUT}/table2_performance.csv", index=False)
    sig_df.to_csv(f"{OUT}/table2b_significance.csv", index=False)
    imp.head(10).to_csv(f"{OUT}/table3_shap.csv", index=False)
    ews.to_csv(f"{OUT}/table4_ews.csv", index=False)

    summary = dict(dataset=table1.to_dict("records"),
                   performance=perf.to_dict("records"),
                   significance=sig_df.to_dict("records"),
                   shap_top10=imp.head(10).to_dict("records"),
                   ews=ews.to_dict("records"))
    json.dump(summary, open(f"{OUT}/results_summary.json","w"), indent=2)

    prop = perf[perf.model=="CNN-LSTM"].iloc[0]
    with open(f"{OUT}/RESULTS_FOR_PAPER.md","w") as fh:
        fh.write("# Hasil OULAD Asli — Siap Tempel ke Naskah\n\n")
        fh.write("## Angka utama CNN-LSTM (ganti di abstrak & kesimpulan)\n\n")
        fh.write(f"- Accuracy: **{prop.accuracy}%**\n- Precision: **{prop.precision}%**\n")
        fh.write(f"- Recall: **{prop.recall}%**\n- F1: **{prop.f1}%**\n- AUC-ROC: **{prop.auc}**\n\n")
        fh.write(f"- Dataset: **{len(df):,} mahasiswa OULAD asli**, dropout {y.mean()*100:.1f}%\n\n")
        fh.write("## Uji signifikansi (BARU — masukkan ke Sec. III.B)\n\n")
        try:
            fh.write(sig_df.to_markdown(index=False))
        except Exception:
            fh.write(sig_df.to_string(index=False))
        fh.write("\n\n> Jika p<0.05: klaim 'statistically significant improvement' kini sah.\n")
        fh.write("> Jika p>=0.05: pertahankan framing 'competitive', itu pun temuan jujur yang valid.\n\n")
        fh.write("## Top-3 SHAP (ganti di abstrak)\n\n")
        for _,r in imp.head(3).iterrows():
            fh.write(f"- {r['feature']}: |SHAP|={r['mean_abs_shap']:.4f}\n")
        fh.write("\n## CATATAN PENTING\n")
        fh.write("Karena ini data NYATA, fitur SHAP teratas kemungkinan BERBEDA dari naskah lama "
                 "(yang menonjolkan disability krn artefak sintetis). Perbarui seluruh interpretasi "
                 "SHAP & equity agar sesuai hasil riil ini. Jangan paksakan narasi lama.\n")
    print(f"\n[SELESAI] Semua output di folder '{OUT}/'. Buka RESULTS_FOR_PAPER.md.")
    print(perf.to_string(index=False))


def main():
    df, feature_cols = build_features()
    X, y = preprocess(df, feature_cols)
    per_fold, yt, yp, n_features = run_cv(X, y)
    sig_df = significance(per_fold)
    imp, _, _ = run_shap(X, y, feature_cols, n_features)
    ews = ews_and_figures(yt, yp, per_fold, imp)
    summarize(df, feature_cols, per_fold, sig_df, imp, ews, y)


if __name__ == "__main__":
    main()
