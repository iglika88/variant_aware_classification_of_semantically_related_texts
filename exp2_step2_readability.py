"""
EXP 2 — STEP 2: BASELINE + READABILITY FEATURES
================================================
LinearSVC + TF-IDF + 5 readability/stylometric features per sentence.

Readability features (5):
  1. sentence_length      — number of tokens
  2. avg_word_length      — average number of characters per word
  3. TTR                  — type-token ratio (vocabulary richness)
  4. character_length     — total number of characters
  5. punctuation_ratio    — punctuation marks / total tokens

Run from: /Users/oussamaelmasri/Documents/finalVersion/
Output:   results/exp2_step2_readability.csv
"""

import os, warnings, string
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

# ── Readability features ──────────────────────────────────────────────────────
PUNCT_SET = set(string.punctuation)

def compute_readability_features(text):
    """
    Compute 5 readability features for a single text (sentence or window).
    Returns a list of 5 floats.
    """
    tokens = text.split()
    n_tokens = max(len(tokens), 1)

    # 1. sentence_length (number of tokens)
    sentence_length = len(tokens)

    # 2. avg_word_length
    avg_word_length = np.mean([len(t) for t in tokens]) if tokens else 0.0

    # 3. TTR (type-token ratio)
    ttr = len(set(t.lower() for t in tokens)) / n_tokens

    # 4. character_length (total chars)
    character_length = len(text)

    # 5. punctuation_ratio
    punct_count = sum(1 for c in text if c in PUNCT_SET)
    punctuation_ratio = punct_count / n_tokens

    return [sentence_length, avg_word_length, ttr,
            character_length, punctuation_ratio]

print("\nComputing readability features per sentence...")
read_features = np.array([compute_readability_features(s) for s in df["sentence"]])
print(f"  Shape: {read_features.shape}  (5 features per sentence)")
print(f"  Sample (first sentence): {read_features[0].round(3)}")

# ── Window builder ────────────────────────────────────────────────────────────
def build_windows(df, read_features, indices, window):
    """Average readability features over the window sentences."""
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
                rf   = read_features[w_idx].mean(axis=0)
                records.append({
                    "text":  text,
                    "label": label,
                    "read":  rf,
                })
    return records

# ── Run ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — BASELINE + READABILITY")
print("=" * 60)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
f1s, accs, all_preds, all_labels = [], [], [], []

for fold, (tr_idx, va_idx) in enumerate(
    skf.split(df.index.values, df["label"].values), 1
):
    tr_recs = build_windows(df, read_features, df.index.values[tr_idx], WINDOW)
    va_recs = build_windows(df, read_features, df.index.values[va_idx],  WINDOW)
    if not va_recs: continue

    X_tr = [r["text"]  for r in tr_recs]
    X_va = [r["text"]  for r in va_recs]
    y_tr = [r["label"] for r in tr_recs]
    y_va = [r["label"] for r in va_recs]
    R_tr = np.array([r["read"] for r in tr_recs])
    R_va = np.array([r["read"] for r in va_recs])

    # Word TF-IDF
    wv = TfidfVectorizer(analyzer="word", ngram_range=(1,1),
                         max_features=50000, sublinear_tf=True)
    Xw_tr = wv.fit_transform(X_tr); Xw_va = wv.transform(X_va)

    # Char TF-IDF
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4),
                         max_features=50000, sublinear_tf=True)
    Xc_tr = cv.fit_transform(X_tr); Xc_va = cv.transform(X_va)

    # Scale readability
    sc = StandardScaler()
    R_tr_s = sc.fit_transform(R_tr)
    R_va_s = sc.transform(R_va)

    Xtr = hstack([Xw_tr, Xc_tr, sp.csr_matrix(R_tr_s)])
    Xva = hstack([Xw_va, Xc_va, sp.csr_matrix(R_va_s)])

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
    "step":       "2_readability",
    "config":     "LinearSVC + TF-IDF + Readability (5)",
    "n_features": "TF-IDF + 5 readability",
    "f1_mean":    mean_f1,
    "f1_std":     std_f1,
    "acc_mean":   np.mean(accs),
}]).to_csv(f"{RESULTS_DIR}/exp2_step2_readability.csv", index=False)
print(f"\nSaved to {RESULTS_DIR}/exp2_step2_readability.csv")
