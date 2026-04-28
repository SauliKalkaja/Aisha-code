"""
style_router.py — SBERT + LogReg classifier picking which per-style
manifold to generate from.

Trains on style_router_train.jsonl, evaluates on style_router_test.jsonl,
and benchmarks against Aisha's existing valence/arousal centroid method.

API:
    from style_router import StyleRouter
    sr = StyleRouter()                       # lazy-loads model + classifier
    style = sr.predict("I miss you so much") # → "emotional"
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CLF_PATH = ROOT / "data" / "processed" / "style_router.pkl"
POS_CLF_PATH = ROOT / "data" / "processed" / "style_router_pos.pkl"
SBERT_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _load_jsonl(path: Path) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            texts.append(r["text"])
            labels.append(r["style"])
    return texts, labels


class StyleRouter:
    """Lazy-loaded style classifier.

    Prefers the POS-augmented classifier (Stage 5, 78%) when available;
    falls back to SBERT-only (Stage 2, 75%) when not.  Both use the same
    SBERT encoder under the hood.
    """

    def __init__(self, clf_path: Path = CLF_PATH,
                 pos_clf_path: Path = POS_CLF_PATH):
        self.clf_path = clf_path
        self.pos_clf_path = pos_clf_path
        self._enc = None
        self._nlp = None
        self._clf: LogisticRegression | None = None
        self._pos_bundle: dict | None = None     # POS-augmented classifier

    def _ensure(self):
        if self._enc is None:
            from sentence_transformers import SentenceTransformer
            self._enc = SentenceTransformer(SBERT_NAME, device="cpu")
        # Prefer POS-augmented bundle when present
        if self._pos_bundle is None and self._clf is None:
            if self.pos_clf_path.exists():
                with self.pos_clf_path.open("rb") as fh:
                    self._pos_bundle = pickle.load(fh)
                import spacy
                self._nlp = spacy.load("en_core_web_sm",
                                        disable=["ner", "parser"])
            else:
                with self.clf_path.open("rb") as fh:
                    self._clf = pickle.load(fh)

    def _encode(self, text: str) -> np.ndarray:
        sbert_vec = self._enc.encode([text], show_progress_bar=False)
        if self._pos_bundle is None:
            return sbert_vec
        # POS-augmented: concat (sbert, standardised POS features)
        from style_router_pos import pos_feature_vector
        doc = self._nlp(text)
        pos_vec = pos_feature_vector(doc)
        pmean = self._pos_bundle["pos_mean"]
        pstd  = self._pos_bundle["pos_std"]
        pos_n = (pos_vec - pmean) / pstd
        return np.concatenate([sbert_vec, pos_n[None, :]], axis=1)

    def predict(self, text: str) -> str:
        self._ensure()
        v = self._encode(text)
        clf = (self._pos_bundle["clf"] if self._pos_bundle is not None
               else self._clf)
        return clf.predict(v)[0]

    def predict_proba(self, text: str) -> dict[str, float]:
        self._ensure()
        v = self._encode(text)
        clf = (self._pos_bundle["clf"] if self._pos_bundle is not None
               else self._clf)
        probs = clf.predict_proba(v)[0]
        return dict(zip(clf.classes_, probs))


def fit():
    train_path = ROOT / "data" / "processed" / "style_router_train.jsonl"
    test_path  = ROOT / "data" / "processed" / "style_router_test.jsonl"
    X_train_text, y_train = _load_jsonl(train_path)
    X_test_text,  y_test  = _load_jsonl(test_path)

    print(f"Training on {len(X_train_text)} turns; testing on {len(X_test_text)}.")
    print("Loading SBERT...")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(SBERT_NAME, device="cpu")
    print("Encoding train...")
    X_train = enc.encode(X_train_text, batch_size=64, show_progress_bar=False)
    print("Encoding test...")
    X_test  = enc.encode(X_test_text,  batch_size=64, show_progress_bar=False)

    clf = LogisticRegression(max_iter=3000, C=2.0,
                              class_weight="balanced", random_state=0)
    clf.fit(X_train, y_train)
    train_acc = float((clf.predict(X_train) == np.array(y_train)).mean())
    test_acc  = float((clf.predict(X_test)  == np.array(y_test)).mean())

    print(f"\n  train acc:  {100*train_acc:.0f}%")
    print(f"  test  acc:  {100*test_acc:.0f}%   (random ≈ 20%)")

    print("\n  per-class on test:")
    pred_test = clf.predict(X_test)
    for cls in sorted(set(y_test)):
        idx = [i for i, y in enumerate(y_test) if y == cls]
        n_ok = sum(1 for i in idx if pred_test[i] == cls)
        print(f"    {cls:<14} {n_ok}/{len(idx)}  ({100*n_ok/len(idx):.0f}%)")

    print("\n  confusion (true → pred):")
    confusion = Counter()
    for true, pred in zip(y_test, pred_test):
        confusion[(true, pred)] += 1
    for (true, pred), n in sorted(confusion.items()):
        if true == pred: continue
        print(f"    {true:<12} → {pred:<12}  {n}")

    with CLF_PATH.open("wb") as fh:
        pickle.dump(clf, fh)
    print(f"\nwrote {CLF_PATH}")


def benchmark_against_va_centroid():
    """Compare against Aisha's existing valence/arousal centroid mechanism."""
    print("\n" + "=" * 60)
    print("Benchmarking SBERT router vs. (v,a) centroid method")
    print("=" * 60)
    from aisha_respond import Responder
    print("Loading Responder...")
    R = Responder()

    test_path = ROOT / "data" / "processed" / "style_router_test.jsonl"
    X_text, y_true = _load_jsonl(test_path)

    # SBERT predictions
    print("SBERT predictions...")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(SBERT_NAME, device="cpu")
    with CLF_PATH.open("rb") as fh:
        clf = pickle.load(fh)
    X_vec = enc.encode(X_text, batch_size=64, show_progress_bar=False)
    sbert_pred = clf.predict(X_vec)

    # Existing (v,a) centroid predictions
    print("Centroid predictions...")
    import re as _re
    centroid_pred = []
    for text in X_text:
        toks = [t.lower() for t in _re.findall(r"[a-zA-Z']+", text)]
        idx = [R.wm.idx[t] for t in toks
                if t in R.wm.idx and R.mask10k[R.wm.idx[t]]]
        if not idx:
            centroid_pred.append("casual")
            continue
        v = float(R.v[np.asarray(idx)].mean())
        a = float(R.a[np.asarray(idx)].mean())
        centroid_pred.append(R._infer_style(v, a))
    centroid_pred = np.array(centroid_pred)

    n = len(y_true)
    sbert_ok = sum(1 for p, t in zip(sbert_pred, y_true) if p == t)
    cent_ok  = sum(1 for p, t in zip(centroid_pred, y_true) if p == t)
    print(f"\n  SBERT:    {sbert_ok}/{n}  ({100*sbert_ok/n:.0f}%)")
    print(f"  centroid: {cent_ok}/{n}   ({100*cent_ok/n:.0f}%)")

    print("\n  per-class accuracy:")
    print(f"  {'class':<14}  {'SBERT':>6}  {'centroid':>9}")
    for cls in sorted(set(y_true)):
        idx = [i for i, y in enumerate(y_true) if y == cls]
        if not idx: continue
        s_ok = sum(1 for i in idx if sbert_pred[i] == cls)
        c_ok = sum(1 for i in idx if centroid_pred[i] == cls)
        n_cls = len(idx)
        print(f"  {cls:<14}  {100*s_ok/n_cls:>5.0f}%  {100*c_ok/n_cls:>8.0f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        benchmark_against_va_centroid()
    else:
        fit()
