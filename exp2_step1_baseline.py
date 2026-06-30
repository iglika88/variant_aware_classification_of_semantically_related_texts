"""
EXP 2 — STEP 1: BASELINE
========================
LinearSVC + TF-IDF (Word + Char n-grams), window=2, leakage-free.
This is the reference baseline for all subsequent steps.

Run from: /Users/oussamaelmasri/Documents/finalVersion/
Output:   results/exp2_step1_baseline.csv
"""

import os, warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH    = "dataset.csv"
WINDOW      = 2
N_FOLDS     = 5
SEED        = 42
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH).reset_index(drop=True)
le = LabelEncoder()
df["label"] = le.fit_transform(df["version"])
CLASSES = list(le.classes_)
print(f"Classes:   {CLASSES}")
print(f"Sentences: {len(df)}")

# ── Window builder (leakage-free) ─────────────────────────────────────────────
def build_windows(df, indices, window):
    """Build windows from sentence indices, discarding cross-boundary windows."""
    idx_set = set(indices)
    records = []
    for version, grp in df.groupby("version"):
        grp = grp.loc[grp.index.isin(idx_set)].sort_index()
        sentences    = grp["sentence"].tolist()
        sent_indices = grp.index.tolist()
        label        = grp["label"].iloc[0]
        for i in range(len(sentences) - window + 1):
            w_idx = sent_indices[i : i + window]
            if w_idx[-1] - w_idx[0] == window - 1:
                text = " ".join(sentences[i : i + window])
                records.append({"text": text, "label": label, "version": version})
    return records

# ── Run baseline ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1 — BASELINE: LinearSVC + TF-IDF (Word + Char), window=2")
print("=" * 60)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
f1s, accs, all_preds, all_labels = [], [], [], []

for fold, (tr_idx, va_idx) in enumerate(
    skf.split(df.index.values, df["label"].values), 1
):
    tr_recs = build_windows(df, df.index.values[tr_idx], WINDOW)
    va_recs = build_windows(df, df.index.values[va_idx],  WINDOW)
    if not va_recs: continue

    X_tr = [r["text"]  for r in tr_recs]
    X_va = [r["text"]  for r in va_recs]
    y_tr = [r["label"] for r in tr_recs]
    y_va = [r["label"] for r in va_recs]

    # Word TF-IDF
    wv = TfidfVectorizer(analyzer="word", ngram_range=(1,1),
                         max_features=50000, sublinear_tf=True)
    Xw_tr = wv.fit_transform(X_tr)
    Xw_va = wv.transform(X_va)

    # Char TF-IDF
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4),
                         max_features=50000, sublinear_tf=True)
    Xc_tr = cv.fit_transform(X_tr)
    Xc_va = cv.transform(X_va)

    Xtr = hstack([Xw_tr, Xc_tr])
    Xva = hstack([Xw_va, Xc_va])

    clf = LinearSVC(class_weight="balanced", max_iter=2000, C=1.0)
    clf.fit(Xtr, y_tr)
    preds = clf.predict(Xva)

    f1  = f1_score(y_va, preds, average="macro")
    acc = np.mean(np.array(preds) == np.array(y_va))
    print(f"  Fold {fold}: n_val={len(y_va):3d}  F1={f1:.4f}  Acc={acc:.4f}")
    f1s.append(f1); accs.append(acc)
    all_preds.extend(preds); all_labels.extend(y_va)

mean_f1, std_f1 = np.mean(f1s), np.std(f1s)
print(f"\n  → Mean F1  = {mean_f1:.4f} ± {std_f1:.4f}")
print(f"  → Mean Acc = {np.mean(accs):.4f}")

print("\nClassification report (pooled folds):")
print(classification_report(all_labels, all_preds, target_names=CLASSES))

# ── Save ──────────────────────────────────────────────────────────────────────
pd.DataFrame([{
    "step":       "1_baseline",
    "config":     "LinearSVC + TF-IDF (Word + Char)",
    "n_features": "TF-IDF only",
    "f1_mean":    mean_f1,
    "f1_std":     std_f1,
    "acc_mean":   np.mean(accs),
}]).to_csv(f"{RESULTS_DIR}/exp2_step1_baseline.csv", index=False)
print(f"\nSaved to {RESULTS_DIR}/exp2_step1_baseline.csv")
