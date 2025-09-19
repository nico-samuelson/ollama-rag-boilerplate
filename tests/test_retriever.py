import types
import uuid
import pytest
from unittest.mock import MagicMock

# ---- Adjust this import if your filename is different ----
import retriever as hr
# ---------------------------------------------------------

from langchain.schema import Document

@pytest.fixture
def hp_patch(monkeypatch):
    # Ensure predictable hyperparameters inside the module under test
    monkeypatch.setattr(hr, "hp", types.SimpleNamespace(
        EMBEDDING_NAME="all-MiniLM-L6-v2",
        SPARSE_RATIO=0.3,
        RETRIEVAL_K=5
    ))

@pytest.fixture
def dummy_docs():
    # Some docs missing .id on purpose to test BM25 id assignment
    return [
        Document(page_content="alpha text", metadata={"id": "keep-existing"}),
        Document(page_content="bravo text"),
        Document(page_content="charlie text", metadata={"source": "x"}),
    ]

# ----- Utilities (simple fakes) -----
class FakeVectorStore:
    def __init__(self):
        self.as_retriever_called_with = None

    def as_retriever(self, **kwargs):
        self.as_retriever_called_with = kwargs
        marker = types.SimpleNamespace(kind="dense_retriever", kwargs=kwargs)
        return marker

class FakeFAISS:
    def __init__(self):
        self.last_args = None
        self.last_kwargs = None
        self.vectorstore = FakeVectorStore()

    def from_documents(self, *args, **kwargs):
        self.last_args = args
        self.last_kwargs = kwargs
        return self.vectorstore

class FakeBM25:
    def __init__(self):
        self.k = None

    @classmethod
    def from_documents(cls, docs):
        inst = cls()
        inst.docs = docs
        return inst

class CaptureEnsemble:
    """Captures constructor arguments and returns a simple marker object."""
    def __init__(self, retrievers, weights):
        self.retrievers = retrievers
        self.weights = weights

# Fake SentenceTransformer that records encode() calls
class CaptureST:
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device
        self.calls = []

    def encode(self, texts, **kwargs):
        # record call, return fixed-size vectors (len=texts)
        self.calls.append({"texts": texts, "kwargs": kwargs})
        import numpy as np
        arr = np.ones((len(texts), 3), dtype="float32")
        return arr

@pytest.fixture
def patch_heavy_deps(monkeypatch):
    # Patch FAISS
    fake_faiss = FakeFAISS()
    monkeypatch.setattr(hr, "FAISS", types.SimpleNamespace(from_documents=fake_faiss.from_documents))

    # Patch BM25Retriever
    monkeypatch.setattr(hr, "BM25Retriever", FakeBM25)

    # Patch EnsembleRetriever
    captured = {}
    def _capture_ensemble(*, retrievers, weights):
        cap = CaptureEnsemble(retrievers, weights)
        captured["instance"] = cap
        return cap
    monkeypatch.setattr(hr, "EnsembleRetriever", _capture_ensemble)

    # Patch torch device availability deterministically (we’ll override per-test if needed)
    monkeypatch.setattr(hr.torch.cuda, "is_available", lambda: False)
    # MPS presence check
    if getattr(hr.torch.backends, "mps", None):
        monkeypatch.setattr(hr.torch.backends.mps, "is_available", lambda: False)

    # Patch SentenceTransformer (used by Qwen3Embeddings)
    monkeypatch.setattr(hr, "SentenceTransformer", CaptureST)

    # Patch SentenceTransformerEmbeddings (fallback path)
    class FakeSTEmb:
        def __init__(self, model_name, model_kwargs=None, encode_kwargs=None):
            self.model_name = model_name
            self.model_kwargs = model_kwargs or {}
            self.encode_kwargs = encode_kwargs or {}
    monkeypatch.setattr(hr, "SentenceTransformerEmbeddings", FakeSTEmb)

    return {"fake_faiss": fake_faiss, "captured_ensemble": captured}

# ---------- Tests ----------

def test_pick_device_respects_preference_cpu(monkeypatch, hp_patch, dummy_docs, patch_heavy_deps):
    r = hr.HybridRetriever(dummy_docs, device_preference="cpu")
    assert r.device == "cpu"

def test_pick_device_cuda_when_available(monkeypatch, hp_patch, dummy_docs, patch_heavy_deps):
    monkeypatch.setattr(hr.torch.cuda, "is_available", lambda: True)
    r = hr.HybridRetriever(dummy_docs)
    assert r.device == "cuda:0"

def test_dense_embeddings_uses_qwen3_when_name_matches(monkeypatch, hp_patch, dummy_docs, patch_heavy_deps):
    monkeypatch.setattr(hr, "hp", types.SimpleNamespace(
        EMBEDDING_NAME="Qwen3-Embedding-xyz",
        SPARSE_RATIO=0.4,
        RETRIEVAL_K=4
    ))
    r = hr.HybridRetriever(dummy_docs, embedding_name="qwen3-embedding-awesome", device_preference="cpu")
    emb = r._dense_embeddings()
    # Should be instance of custom Qwen3Embeddings (not the fallback)
    assert isinstance(emb, hr.Qwen3Embeddings)
    # Ensure it constructed our CaptureST with the right model/device
    assert isinstance(emb.model, CaptureST)
    assert emb.model.model_name == "qwen3-embedding-awesome"
    assert emb.model.device == "cpu"

def test_qwen3_embeddings_embed_documents_and_query(monkeypatch, hp_patch, patch_heavy_deps):
    q = hr.Qwen3Embeddings(model_name="qwen3-embedding", device="cpu", query_prompt_name="query")
    vecs = q.embed_documents(["a", "b", "c"])
    assert isinstance(vecs, list) and len(vecs) == 3 and all(len(v) == 3 for v in vecs)

    q.embed_query("hello world")
    # Verify prompt_name flowed through to encode
    assert q.model.calls[-1]["kwargs"].get("prompt_name") == "query"
    # normalize_embeddings should be True by default per implementation
    assert q.model.calls[-1]["kwargs"].get("normalize_embeddings") is True

def test_sparse_retriever_assigns_missing_ids(monkeypatch, hp_patch, dummy_docs, patch_heavy_deps):
    r = hr.HybridRetriever(dummy_docs, device_preference="cpu")
    bm25 = r._setup_sparse_retriever()

    # First doc keeps existing id from metadata
    assert dummy_docs[0].id == "keep-existing"
    # Others should be auto-assigned
    assert isinstance(dummy_docs[1].id, str) and dummy_docs[1].id.startswith("doc-")
    assert isinstance(dummy_docs[2].id, str) and dummy_docs[2].id.startswith("doc-")

    # BM25 k should match retrieval_k
    assert bm25.k == r.retrieval_k == hr.hp.RETRIEVAL_K

def test_dense_retriever_wires_faiss_and_k(monkeypatch, hp_patch, dummy_docs, patch_heavy_deps):
    r = hr.HybridRetriever(dummy_docs, device_preference="cpu")
    dense = r._setup_dense_retriever()

    # FAISS.from_documents called with docs, embeddings, and cosine distance
    fa = patch_heavy_deps["fake_faiss"]
    assert fa.last_args[0] == dummy_docs
    # embeddings object is passed as second positional arg
    assert hasattr(fa.last_args[1], "embed_documents") or hasattr(fa.last_args[1], "model_name")
    assert fa.last_kwargs.get("distance_strategy") == hr.DistanceStrategy.COSINE

    # as_retriever called with the module's hp.RETRIEVAL_K
    assert dense.kind == "dense_retriever"
    assert dense.kwargs == {"search_kwargs": {"k": hr.hp.RETRIEVAL_K}}

def test_hybrid_retriever_builds_ensemble_with_normalized_weights(monkeypatch, hp_patch, dummy_docs, patch_heavy_deps):
    # Give an out-of-range sparse_ratio; should clamp to [0,1]
    r = hr.HybridRetriever(dummy_docs, sparse_ratio=2.0, device_preference="cpu")
    cap = patch_heavy_deps["captured_ensemble"]["instance"]
    assert isinstance(cap, CaptureEnsemble)
    # Weights should be [1.0, 0.0] after clamping
    assert cap.weights == [1.0, 0.0]
    # Should contain two retrievers (sparse, dense) in that order
    assert len(cap.retrievers) == 2

def test_as_retriever_returns_ensemble_instance(monkeypatch, hp_patch, dummy_docs, patch_heavy_deps):
    r = hr.HybridRetriever(dummy_docs, device_preference="cpu")
    ens = r.as_retriever()
    cap = patch_heavy_deps["captured_ensemble"]["instance"]
    assert ens is cap
