"""
EXP 2 — STEP 4: BASELINE + READABILITY + VARIANT + ENHANCED VARIANT
====================================================================
LinearSVC + TF-IDF + 5 readability + 18 variant + 6 enhanced features.

Enhanced features (per pair × 3 other versions = 6 total):
  1. lemma_ratio  — share of tokens with matching lemma (via spaCy)
  2. pos_ratio    — share of tokens with matching POS tag (via spaCy)

Requires: spaCy with en_core_web_sm model
  pip install spacy
  python -m spacy download en_core_web_sm

Run from: /Users/oussamaelmasri/Documents/finalVersion/
Output:   results/exp2_step4_enhanced.csv
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
import spacy

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH    = "dataset.csv"
WINDOW      = 2
N_FOLDS     = 5
SEED        = 42
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading spaCy model (en_core_web_sm)...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH).reset_index(drop=True)
le = LabelEncoder()
df["label"] = le.fit_transform(df["version"])
CLASSES = list(le.classes_)
print(f"Classes:   {CLASSES}")
print(f"Sentences: {len(df)}")

# ── Cache spaCy analyses (lemma + POS) ────────────────────────────────────────
print("\nProcessing sentences with spaCy (caching lemma + POS)...")
spacy_cache = {}
for i, sent in enumerate(df["sentence"]):
    doc = nlp(sent)
    spacy_cache[i] = [(t.lemma_.lower(), t.pos_) for t in doc if not t.is_space]
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{len(df)} sentences processed")
print(f"  Cached {len(spacy_cache)} sentence analyses")

# ── Readability ───────────────────────────────────────────────────────────────
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

# ── Variant features (18 per sentence) ────────────────────────────────────────
def variant_ratios(sent_a, sent_b):
    tokens_a = sent_a.lower().split()
    tokens_b = sent_b.lower().split()
    set_a = set(tokens_a); set_b = set(tokens_b)
    n_a = max(len(set_a), 1); n_b = max(len(set_b), 1)
    same         = len(set_a & set_b) / n_a
    replace      = len(set_b - set_a) / n_a
    insert       = len(set_b - set_a) / n_b
    delete       = len(set_a - set_b) / n_a
    compression  = len(tokens_b) / max(len(tokens_a), 1)
    length_diff  = (len(tokens_a) - len(tokens_b)) / max(len(tokens_a), 1)
    return [same, replace, insert, delete, compression, length_diff]

def find_closest_index(sent_idx, candidate_indices):
    """Find index of closest sentence in candidates by token overlap."""
    sent = df.loc[sent_idx, "sentence"]
    set_a = set(sent.lower().split())
    if not set_a or not candidate_indices:
        return candidate_indices[0] if candidate_indices else sent_idx
    best_idx, best_overlap = candidate_indices[0], -1
    for c_idx in candidate_indices:
        c_set = set(df.loc[c_idx, "sentence"].lower().split())
        overlap = len(set_a & c_set) / max(len(set_a), 1)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = c_idx
    return best_idx

# ── Enhanced features (2 per pair × 3 other versions = 6 total) ───────────────
def enhanced_ratios_from_idx(idx_a, idx_b):
    """
    Compare token-level lemma and POS between two sentences using spaCy cache.
    Returns:
      lemma_ratio = matching lemmas / total tokens in A
      pos_ratio   = matching POS    / total tokens in A
    """
    analysis_a = spacy_cache[idx_a]
    analysis_b = spacy_cache[idx_b]
    if not analysis_a:
        return [0.0, 0.0]
    lemmas_b = set(l for l, _ in analysis_b)
    pos_b    = set(p for _, p in analysis_b)
    n_a      = len(analysis_a)
    lemma_match = sum(1 for l, _ in analysis_a if l in lemmas_b)
    pos_match   = sum(1 for _, p in analysis_a if p in pos_b)
    return [lemma_match / n_a, pos_match / n_a]

def compute_variant_and_enhanced_features(df):
    """
    For each sentence, find closest match in each of 3 other versions.
    Compute 6 variant + 2 enhanced features per pair.
    Total: 18 variant + 6 enhanced = 24 features per sentence.
    """
    versions = sorted(df["version"].unique())
    version_indices = {v: df[df["version"]==v].index.tolist() for v in versions}

    var_features, enh_features = [], []
    for sent_idx, row in df.iterrows():
        version = row["version"]
        var_row, enh_row = [], []
        for other_v in versions:
            if other_v == version: continue
            closest_idx = find_closest_index(sent_idx, version_indices[other_v])
            sent_a = df.loc[sent_idx,   "sentence"]
            sent_b = df.loc[closest_idx,"sentence"]
            var_row.extend(variant_ratios(sent_a, sent_b))
            enh_row.extend(enhanced_ratios_from_idx(sent_idx, closest_idx))
        var_features.append(var_row)
        enh_features.append(enh_row)
        if (sent_idx+1) % 200 == 0:
            print(f"  {sent_idx+1}/{len(df)} processed")
    return np.array(var_features), np.array(enh_features)

print("\nComputing variant + enhanced features (this takes ~2-3 min)...")
var_features, enh_features = compute_variant_and_enhanced_features(df)
print(f"  Variant shape:  {var_features.shape}  (18 features per sentence)")
print(f"  Enhanced shape: {enh_features.shape}  (6 features per sentence)")
print(f"  Total numeric features per sentence: {5 + 18 + 6}")

# ── Window builder ────────────────────────────────────────────────────────────
def build_windows(df, read_feats, var_feats, enh_feats, indices, window):
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
                records.append({
                    "text":  text,
                    "label": label,
                    "read":  read_feats[w_idx].mean(axis=0),
                    "var":   var_feats[w_idx].mean(axis=0),
                    "enh":   enh_feats[w_idx].mean(axis=0),
                })
    return records

# ── Run ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — BASELINE + READABILITY + VARIANT + ENHANCED VARIANT")
print("=" * 60)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
f1s, accs, all_preds, all_labels = [], [], [], []

for fold, (tr_idx, va_idx) in enumerate(
    skf.split(df.index.values, df["label"].values), 1
):
    tr_recs = build_windows(df, read_features, var_features, enh_features,
                            df.index.values[tr_idx], WINDOW)
    va_recs = build_windows(df, read_features, var_features, enh_features,
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
    E_tr = np.array([r["enh"]  for r in tr_recs])
    E_va = np.array([r["enh"]  for r in va_recs])

    # TF-IDF
    wv = TfidfVectorizer(analyzer="word", ngram_range=(1,1),
                         max_features=50000, sublinear_tf=True)
    Xw_tr = wv.fit_transform(X_tr); Xw_va = wv.transform(X_va)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4),
                         max_features=50000, sublinear_tf=True)
    Xc_tr = cv.fit_transform(X_tr); Xc_va = cv.transform(X_va)

    # Scale numeric features together
    numeric_tr = np.hstack([R_tr, V_tr, E_tr])
    numeric_va = np.hstack([R_va, V_va, E_va])
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
    "step":       "4_enhanced",
    "config":     "LinearSVC + TF-IDF + Readability (5) + Variant (18) + Enhanced (6)",
    "n_features": "TF-IDF + 5 readability + 18 variant + 6 enhanced",
    "f1_mean":    mean_f1,
    "f1_std":     std_f1,
    "acc_mean":   np.mean(accs),
}]).to_csv(f"{RESULTS_DIR}/exp2_step4_enhanced.csv", index=False)
print(f"\nSaved to {RESULTS_DIR}/exp2_step4_enhanced.csv")
