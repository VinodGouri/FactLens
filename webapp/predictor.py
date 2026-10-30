"""Inference service for the web app.

Loads ONLY models/best_model (the model selected in the training notebook) and applies the
exact preprocessing + RAG pipeline used during training (functions identical to the notebooks).
"""
import os, re, json, threading
import numpy as np
import pandas as pd
import joblib
import torch
os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")
import nltk
def _ensure_nltk(res):
    for path in (f"corpora/{res}", f"corpora/{res}.zip"):
        try:
            nltk.data.find(path)
            return
        except LookupError:
            pass
    nltk.download(res, quiet=True)
for _res in ["wordnet", "omw-1.4", "stopwords"]:
    _ensure_nltk(_res)
from nltk.corpus import wordnet as wn, stopwords
from nltk.wsd import lesk
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

STOP_WORDS = set(stopwords.words("english"))

def clean_text(text):
    """Normalise quotes/whitespace exactly as done during training."""
    t = str(text)
    t = (t.replace("“", '"').replace("”", '"')
           .replace("‘", "'").replace("’", "'"))
    t = re.sub(r"\s+", " ", t).strip()
    return t

def truncate_words(text, n):
    w = text.split()
    return " ".join(w[:n])

def prepare_statement(text, cfg):
    """Cleaned statement used as model input (very long outliers are capped)."""
    return truncate_words(clean_text(text), cfg["max_statement_words"])

def retrieve_neighbors(query_texts, vectorizer, kb_matrix, k, exclude_idx=None, chunk=1024):
    """TF-IDF cosine retrieval over the fact-checked knowledge base.
    exclude_idx[i] = KB row of query i itself (leave-one-out for training rows), or -1."""
    q = vectorizer.transform(query_texts)
    all_idx, all_sim = [], []
    for s in range(0, q.shape[0], chunk):
        sims = (q[s:s + chunk] @ kb_matrix.T).toarray()
        if exclude_idx is not None:
            for r, ex in enumerate(exclude_idx[s:s + chunk]):
                if ex >= 0:
                    sims[r, ex] = -1.0
        top = np.argsort(-sims, axis=1)[:, :k]
        all_idx.append(top)
        all_sim.append(np.take_along_axis(sims, top, axis=1))
    return np.vstack(all_idx), np.vstack(all_sim)

def polysemy_glosses(text, context_tokens, cfg):
    """Pick the most polysemous content words (WordNet) and disambiguate them with
    Lesk using the statement + retrieved evidence as context (RAG-enhanced WSD)."""
    words, seen = [], set()
    for w in re.findall(r"[A-Za-z]+", text.lower()):
        if w in STOP_WORDS or len(w) < 3 or w in seen:
            continue
        seen.add(w)
        n = len(wn.synsets(w))
        if n >= cfg["min_senses"]:
            words.append((n, w))
    words.sort(key=lambda x: (-x[0], x[1]))
    out = []
    for n, w in words[: cfg["max_polysemous_words"]]:
        sense = lesk(context_tokens, w)
        if sense is not None:
            out.append(f"{w}: {truncate_words(sense.definition(), cfg['max_gloss_words'])}")
    return out

def build_rag_context(text, nb_idx, nb_sim, kb_texts, kb_labels, cfg):
    """Second input segment for the RAG models: retrieved fact-checked claims
    (with their verdicts) + WordNet senses of polysemous words."""
    evid, ctx_tokens = [], re.findall(r"[A-Za-z]+", text.lower())
    for j, s in zip(nb_idx, nb_sim):
        if s < cfg["min_similarity"]:
            continue
        nt = truncate_words(kb_texts[j], cfg["max_evidence_words"])
        evid.append(f"[{cfg['label_tags'][int(kb_labels[j])]}] {nt}")
        ctx_tokens += re.findall(r"[A-Za-z]+", nt.lower())
    gl = polysemy_glosses(text, ctx_tokens, cfg)
    parts = ["Evidence: " + (" | ".join(evid) if evid else "none")]
    parts.append("Senses: " + ("; ".join(gl) if gl else "none"))
    return " ".join(parts)

def make_rag_inputs(texts, vectorizer, kb_matrix, kb_texts, kb_labels, cfg, exclude_idx=None):
    idx, sim = retrieve_neighbors(texts, vectorizer, kb_matrix, cfg["top_k"], exclude_idx)
    return [build_rag_context(t, idx[i], sim[i], kb_texts, kb_labels, cfg) for i, t in enumerate(texts)]

def load_text_classifier(model_dir, device="cpu"):
    """Rebuild architecture from config.json, load PyTorch weights (model.pth) + tokenizer."""
    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_config(config)
    state = torch.load(os.path.join(model_dir, "model.pth"), map_location=device, mmap=True)
    model.load_state_dict(state, assign=True)  # assign + mmap avoids holding two copies in RAM
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model.to(device).eval(), tokenizer

@torch.no_grad()
def predict_proba_transformer(model, tokenizer, texts_a, texts_b=None, max_len=128, batch_size=64):
    """Return P(class=1 'fake') for each input, using the same tokenisation as training."""
    model.eval()
    probs = []
    for s in range(0, len(texts_a), batch_size):
        a = list(texts_a[s:s + batch_size])
        b = None if texts_b is None else list(texts_b[s:s + batch_size])
        enc = tokenizer(a, b, truncation="longest_first" if b is not None else True,
                        max_length=max_len, padding=True, return_tensors="pt")
        enc.pop("token_type_ids", None)
        logits = model(**{k: v.to(model.device) for k, v in enc.items()}).logits
        probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs) if probs else np.array([])


class FakeNewsPredictor:
    """Thread-safe, lazily loaded wrapper around the saved best model."""

    def __init__(self, best_dir):
        self.best_dir = best_dir
        self._lock = threading.Lock()
        self._loaded = False

    def _load(self):
        d = self.best_dir
        self.info = json.load(open(os.path.join(d, "best_model_info.json")))
        self.label_map = {int(k): v for k, v in self.info["label_map"].items()}
        self.threshold = self.info["threshold"]
        ret = os.path.join(d, "rag_retriever")
        self.vec = joblib.load(os.path.join(ret, "tfidf_retriever.joblib"))
        kb = pd.read_csv(os.path.join(ret, "knowledge_base.csv"))
        self.kb_texts, self.kb_labels = kb["text"].astype(str).tolist(), kb["label"].to_numpy()
        self.kb_matrix = self.vec.transform(self.kb_texts)
        self.rag_cfg = json.load(open(os.path.join(ret, "rag_config.json")))
        self.rag_cfg["label_tags"] = {int(k): v for k, v in self.rag_cfg["label_tags"].items()}
        if self.info["type"] == "ensemble":
            ens = json.load(open(os.path.join(d, "ensemble_config.json")))
            self.weights = ens["weights"]
            paths = {n: os.path.join(d, "members", n) for n in self.weights}
        else:
            self.weights = {self.info["best_model"]: 1.0}
            paths = {self.info["best_model"]: os.path.join(d, "model")}
        self.components = {}
        for n, p in paths.items():
            meta = json.load(open(os.path.join(p, "metadata.json")))
            if meta["framework"] == "scikit-learn":
                self.components[n] = {"kind": "sklearn", "model": joblib.load(os.path.join(p, "model.joblib")), "meta": meta}
            else:
                m, t = load_text_classifier(p)
                self.components[n] = {"kind": "transformer", "model": m, "tokenizer": t, "meta": meta}
        self._loaded = True

    def ensure_loaded(self):
        with self._lock:
            if not self._loaded:
                self._load()

    @property
    def model_info(self):
        self.ensure_loaded()
        return self.info

    def predict(self, statements):
        self.ensure_loaded()
        with self._lock, torch.no_grad():
            stmts = [prepare_statement(s, self.rag_cfg) for s in statements]
            clean = [clean_text(s) for s in statements]
            idx, sim = retrieve_neighbors(clean, self.vec, self.kb_matrix, self.rag_cfg["top_k"])
            contexts = [build_rag_context(t, idx[i], sim[i], self.kb_texts, self.kb_labels, self.rag_cfg)
                        for i, t in enumerate(clean)]
            p_fake = np.zeros(len(stmts))
            for n, c in self.components.items():
                if c["kind"] == "sklearn":
                    p = c["model"].predict_proba(stmts)[:, 1]
                else:
                    p = predict_proba_transformer(c["model"], c["tokenizer"], stmts,
                                                  contexts if c["meta"]["uses_rag"] else None, c["meta"]["max_len"])
                p_fake += self.weights[n] * p
        out = []
        for i, s in enumerate(statements):
            pred = int(p_fake[i] >= self.threshold)
            evidence = [{"text": self.kb_texts[j], "verdict": self.label_map[int(self.kb_labels[j])],
                         "similarity": round(float(sim[i][k]), 3)}
                        for k, j in enumerate(idx[i]) if sim[i][k] >= self.rag_cfg["min_similarity"]]
            out.append({"statement": s, "prediction": self.label_map[pred], "p_fake": round(float(p_fake[i]), 4),
                        "confidence": round(float(p_fake[i] if pred == 1 else 1 - p_fake[i]), 4),
                        "rag_context": contexts[i], "evidence": evidence,
                        "uses_rag": any(c["meta"]["uses_rag"] for c in self.components.values())})
        return out
