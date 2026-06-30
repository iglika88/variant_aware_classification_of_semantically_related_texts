"""
EXP 2 — STEP 3: BASELINE + READABILITY + VARIANT FEATURES
==========================================================
LinearSVC + TF-IDF + 5 readability + 6 variant features per pair × 3 other versions.

For each sentence, find its closest counterpart in every OTHER version
(by token overlap) and compute 6 variant ratios for that pair:
  1. same_ratio          — tokens shared between sentence and closest match
  2. replace_ratio       — tokens only in closest match
  3. insert_ratio        — tokens in closest match but not in sentence (== replace here)
  4. delete_ratio        — tokens in sentence but not in closest match
  5. compression_ratio   — len(closest_match) / len(sentence)
  6. length_diff_ratio   — (len(sentence) - len(closest_match)) / len(sentence)

3 other versions × 6 features = 18 variant features per sentence.

Run from: /Users/oussamaelmasri/Documents/finalVersion/
Output:   results/exp2_step3_variant.csv
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

# ── Readability (same as Step 2) ──────────────────────────────────────────────
PUNCT_SET = set(string.punctuation)

def compute_readability_features(text):
    tokens = text.split()
    n_tokens = max(len(tokens), 1)
    sentence_length   = len(tokens)
    avg_word_length   = np.mean([len(t) for t in tokens]) if tokens else 0.0
    ttr               = len(set(t.lower() for t in tokens)) / n_tokens
    character_length  = len(text)
    punctuation_ratio = sum(1 for c in text if c in PUNCT_SET) / n_tokens
    return [sentence_length, avg_word_length, ttr, character_length, punctuation_ratio]

print("\nComputing readability features...")
read_features = np.array([compute_readability_features(s) for s in df["sentence"]])
print(f"  Shape: {read_features.shape}")

# ── Variant features (6 per pair × 3 versions = 18) ───────────────────────────
def variant_ratios(sent_a, sent_b):
    """
    Compute 6 variant ratios between sentence A (current) and its closest match B.
    """
    tokens_a = sent_a.lower().split()
    tokens_b = sent_b.lower().split()
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    n_a   = max(len(set_a), 1)
    n_b   = max(len(set_b), 1)

    same_ratio        = len(set_a & set_b) / n_a
    replace_ratio     = len(set_b - set_a) / n_a
    insert_ratio      = len(set_b - set_a) / n_b   # tokens added in b
    delete_ratio      = len(set_a - set_b) / n_a   # tokens removed from a
    compression_ratio = len(tokens_b) / max(len(tokens_a), 1)
    length_diff_ratio = (len(tokens_a) - len(tokens_b)) / max(len(tokens_a), 1)

    return [same_ratio, replace_ratio, insert_ratio,
            delete_ratio, compression_ratio, length_diff_ratio]

def find_closest_sentence(sent, candidates):
    """Find closest sentence in candidates by token overlap."""
    set_a = set(sent.lower().split())
    if not set_a: return candidates[0] if candidates else sent
    best_sent, best_overlap = candidates[0], -1
    for c in candidates:
        set_c = set(c.lower().split())
        overlap = len(set_a & set_c) / max(len(set_a), 1)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sent = c
    return best_sent

def compute_variant_features(df):
    """
    For each sentence, find its closest match in each of the 3 other versions
    and compute 6 variant ratios per pair = 18 features per sentence.
    """
    versions = sorted(df["version"].unique())
    version_sentences = {v: df[df["version"]==v]["sentence"].tolist()
                         for v in versions}
    features = []
    for _, row in df.iterrows():
        sent    = row["sentence"]
        version = row["version"]
        row_feats = []
        for other_v in versions:
            if other_v == version: continue
            closest = find_closest_sentence(sent, version_sentences[other_v])
            row_feats.extend(variant_ratios(sent, closest))
        features.append(row_feats)
    return np.array(features)

print("\nComputing variant features (this takes ~1-2 min)...")
var_features = compute_variant_features(df)
print(f"  Shape: {var_features.shape}  (18 features per sentence)")
print(f"  Sample (first sentence): {var_features[0].round(3)}")

# ── Window builder ────────────────────────────────────────────────────────────
def build_windows(df, read_feats, var_feats, indices, window):
    """Average readability and variant features over the window."""
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
                rf   = read_feats[w_idx].mean(axis=0)
                vf   = var_feats[w_idx].mean(axis=0)
                records.append({
                    "text":  text,
                    "label": label,
                    "read":  rf,
                    "var":   vf,
                })
    return records

# ── Run ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — BASELINE + READABILITY + VARIANT (18 features)")
print("=" * 60)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
f1s, accs, all_preds, all_labels = [], [], [], []

for fold, (tr_idx, va_idx) in enumerate(
    skf.split(df.index.values, df["label"].values), 1
):
    tr_recs = build_windows(df, read_features, var_features,
                            df.index.values[tr_idx], WINDOW)
    va_recs = build_windows(df, read_features, var_features,
                            df.index.values[va_idx],  WINDOW)
    if not va_recs: continue

    X_tr = [r["text"]  for r in tr_recs]
    X_va = [r["text"]  for r in va_recs]
    y_tr = [r["label"] for r in tr_recs]
    y_va = [r["label"] for r in va_recs]
    R_tr = np.array([r["read"] for r in tr_recs])
    R_va = np.array([r["read"] for r in va_recs])
    V_tr = np.array([r["var"]  for r in tr_recs])
    V_va = np.array([r["var"]  for r in va_recs])

    # TF-IDF
    wv = TfidfVectorizer(analyzer="word", ngram_range=(1,1),
                         max_features=50000, sublinear_tf=True)
    Xw_tr = wv.fit_transform(X_tr); Xw_va = wv.transform(X_va)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4),
                         max_features=50000, sublinear_tf=True)
    Xc_tr = cv.fit_transform(X_tr); Xc_va = cv.transform(X_va)

    # Scale numeric features together
    numeric_tr = np.hstack([R_tr, V_tr])
    numeric_va = np.hstack([R_va, V_va])
    sc = StandardScaler()
    numeric_tr_s = sc.fit_transform(numeric_tr)
    numeric_va_s = sc.transform(numeric_va)

    Xtr = hstack([Xw_tr, Xc_tr, sp.csr_matrix(numeric_tr_s)])
    Xva = hstack([Xw_va, Xc_va, sp.csr_matrix(numeric_va_s)])

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
    "step":       "3_variant",
    "config":     "LinearSVC + TF-IDF + Readability (5) + Variant (18)",
    "n_features": "TF-IDF + 5 readability + 18 variant",
    "f1_mean":    mean_f1,
    "f1_std":     std_f1,
    "acc_mean":   np.mean(accs),
}]).to_csv(f"{RESULTS_DIR}/exp2_step3_variant.csv", index=False)
print(f"\nSaved to {RESULTS_DIR}/exp2_step3_variant.csv")
