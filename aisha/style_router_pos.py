"""
style_router_pos.py — does adding POS-distribution features to the SBERT
embedding improve routing?

POS features per turn (12-dim, normalised):
  • POS rate per channel (Noun/Verb/Adj/Adv/Pro/Prep/Conj/Interj) — 8 dim
  • mean token length, sentence length, punctuation rate, digit rate — 4 dim

If concat(SBERT, POS_features) → LogReg beats SBERT alone on the held-out
test set, we keep this router.  If not (or it's marginal), we reject and
stick with the SBERT-only Stage 2 router.

Honest test — same train/test split as style_router.py.
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SBERT_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# spacy POS → 8-channel mapping consistent with the universal manifold
SPACY_TO_CH = {
    "NOUN": 0, "PROPN": 0,
    "VERB": 1, "AUX":  1,
    "ADJ":  2,
    "ADV":  3,
    "PRON": 4,
    "ADP":  5,
    "CCONJ": 6, "SCONJ": 6,
    "INTJ": 7,
}


def load_jsonl(path: Path) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            texts.append(r["text"])
            labels.append(r["style"])
    return texts, labels


def pos_feature_vector(doc) -> np.ndarray:
    """12-D feature vector from a spacy doc."""
    counts = np.zeros(8, dtype=np.float64)
    n_total = 0
    n_punct = 0
    n_digit = 0
    char_lens = []
    for tok in doc:
        if tok.pos_ == "PUNCT":
            n_punct += 1
            continue
        if tok.is_digit:
            n_digit += 1
        ch = SPACY_TO_CH.get(tok.pos_)
        if ch is not None:
            counts[ch] += 1
            char_lens.append(len(tok.text))
        n_total += 1
    pos_rate = counts / max(n_total, 1)
    mean_tok_len = float(np.mean(char_lens)) if char_lens else 0.0
    sent_len = float(n_total)
    punct_rate = n_punct / max(n_total + n_punct, 1)
    digit_rate = n_digit / max(n_total, 1)
    return np.concatenate([pos_rate,
                           [mean_tok_len, sent_len, punct_rate, digit_rate]])


def encode_pos_features(texts: list[str]) -> np.ndarray:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    out = np.zeros((len(texts), 12), dtype=np.float64)
    for i, doc in enumerate(nlp.pipe(texts, batch_size=64)):
        out[i] = pos_feature_vector(doc)
    return out


def encode_sbert(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(SBERT_NAME, device="cpu")
    return enc.encode(texts, batch_size=128, show_progress_bar=False)


def main():
    train_path = ROOT / "data" / "processed" / "style_router_train.jsonl"
    test_path  = ROOT / "data" / "processed" / "style_router_test.jsonl"
    X_tr_text, y_tr = load_jsonl(train_path)
    X_te_text, y_te = load_jsonl(test_path)

    print(f"Loading SBERT and encoding {len(X_tr_text)+len(X_te_text)} turns...")
    Xs_tr = encode_sbert(X_tr_text)
    Xs_te = encode_sbert(X_te_text)

    print("Computing POS features (spacy POS-tagging)...")
    Xp_tr = encode_pos_features(X_tr_text)
    Xp_te = encode_pos_features(X_te_text)

    # Standardise POS features so they don't dominate or vanish next to SBERT
    pmean = Xp_tr.mean(axis=0); pstd = Xp_tr.std(axis=0) + 1e-9
    Xp_tr_n = (Xp_tr - pmean) / pstd
    Xp_te_n = (Xp_te - pmean) / pstd

    # SBERT-only baseline
    print("\nFitting SBERT-only baseline...")
    clf_s = LogisticRegression(max_iter=3000, C=2.0,
                                class_weight="balanced", random_state=0)
    clf_s.fit(Xs_tr, y_tr)
    s_acc = float((clf_s.predict(Xs_te) == np.array(y_te)).mean())

    # Concat SBERT + POS features
    print("Fitting SBERT + POS classifier...")
    Xc_tr = np.concatenate([Xs_tr, Xp_tr_n], axis=1)
    Xc_te = np.concatenate([Xs_te, Xp_te_n], axis=1)
    clf_c = LogisticRegression(max_iter=3000, C=2.0,
                                class_weight="balanced", random_state=0)
    clf_c.fit(Xc_tr, y_tr)
    c_acc = float((clf_c.predict(Xc_te) == np.array(y_te)).mean())

    # POS-only (sanity)
    print("Fitting POS-only (sanity)...")
    clf_p = LogisticRegression(max_iter=3000, C=2.0,
                                class_weight="balanced", random_state=0)
    clf_p.fit(Xp_tr_n, y_tr)
    p_acc = float((clf_p.predict(Xp_te_n) == np.array(y_te)).mean())

    print()
    print("=" * 60)
    print(f"  POS-only baseline:    {100*p_acc:.1f}%   (sanity: random ≈ 20%)")
    print(f"  SBERT-only:           {100*s_acc:.1f}%")
    print(f"  SBERT + POS concat:   {100*c_acc:.1f}%")
    delta = c_acc - s_acc
    print(f"  Δ from concat:        {100*delta:+.2f}pp")

    if delta > 0.005:
        print(f"\n  → POS features HELP — saving combined classifier")
        out = ROOT / "data" / "processed" / "style_router_pos.pkl"
        with out.open("wb") as fh:
            pickle.dump({"clf": clf_c, "pos_mean": pmean, "pos_std": pstd}, fh)
        print(f"    wrote {out}")
        print(f"    Stage 5 ACCEPTED: pipeline can be upgraded to use this.")
    elif delta < -0.005:
        print(f"\n  → POS features HURT — Stage 5 REJECTED")
    else:
        print(f"\n  → POS features within noise — Stage 5 REJECTED (no improvement)")

    # Per-class breakdown of where it changed (positive or negative)
    print("\n  per-class accuracy:")
    print(f"  {'class':<14}  {'SBERT':>6}  {'+POS':>6}  {'Δpp':>6}")
    pred_s = clf_s.predict(Xs_te)
    pred_c = clf_c.predict(Xc_te)
    for cls in sorted(set(y_te)):
        idx = [i for i, t in enumerate(y_te) if t == cls]
        s_ok = sum(1 for i in idx if pred_s[i] == cls) / len(idx)
        c_ok = sum(1 for i in idx if pred_c[i] == cls) / len(idx)
        print(f"  {cls:<14}  {100*s_ok:>5.0f}%  {100*c_ok:>5.0f}%  {100*(c_ok-s_ok):>+5.1f}")


if __name__ == "__main__":
    main()
