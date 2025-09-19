import torch
import uuid
import hyperparameters as hp
from typing import List, Optional
from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import SentenceTransformerEmbeddings

class Qwen3Embeddings(Embeddings):
    def __init__(
        self,
        model_name: str = hp.EMBEDDING_NAME,
        device: str = "cuda",
        query_prompt_name: str = "query",
        **encode_kwargs,
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.encode_kwargs = {"normalize_embeddings": True, **encode_kwargs}
        self.query_prompt_name = query_prompt_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, **self.encode_kwargs).tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self.model.encode([text], prompt_name=self.query_prompt_name, **self.encode_kwargs)
        return vec[0].tolist()

class HybridRetriever:
    def __init__(self,
        chunks: List[Document],
        embedding_name : str = hp.EMBEDDING_NAME,
        sparse_ratio: float = hp.SPARSE_RATIO,
        retrieval_k: int = hp.RETRIEVAL_K,
        device_preference: Optional[str] = None,
    ):
        self.chunks = chunks
        self.embedding_name = embedding_name
        self.sparse_ratio = float(sparse_ratio)
        self.retrieval_k = int(retrieval_k)
        self.device = self._pick_device(device_preference)
        self.retriever = self._setup_hybrid_retriever()

    def as_retriever(self) -> EnsembleRetriever:
        return self.retriever

    def _pick_device(self, preference: Optional[str]) -> str:
        """Pick the best available device, with optional user preference."""

        if preference:
            return preference
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _dense_embeddings(self) -> SentenceTransformerEmbeddings:
        """Setup dense embeddings based on the model name."""

        name = self.embedding_name.lower()
        if "qwen3-embedding" in name:
            return Qwen3Embeddings(self.embedding_name, device=self.device)

        # fallback
        kwargs = {"device": self.device} if self.device != "cpu" else {}
        return SentenceTransformerEmbeddings(
            model_name=self.embedding_name,
            model_kwargs=kwargs,
            encode_kwargs={
                "batch_size": 32 if str(self.device).startswith("cuda") else 16,
                "trust_remote_code": True,
                "normalize_embeddings": True,
            },
        )

    def _setup_dense_retriever(self) -> VectorStoreRetriever:
        """Setup dense retriever using FAISS and the specified embeddings."""

        embeddings = self._dense_embeddings()

        vectorstore = FAISS.from_documents(
            self.chunks,
            embeddings,
            distance_strategy=DistanceStrategy.COSINE
        )
        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": hp.RETRIEVAL_K})

        print(f"Dense embeddings on {self.device}")
        return dense_retriever

    def _setup_sparse_retriever(self) -> BM25Retriever:
        """Setup sparse retriever using BM25."""

        # Ensure every doc has an .id (required by langchain_community BM25Retriever)
        for i, d in enumerate(self.chunks):
            if not hasattr(d, "id") or d.id is None:
                # prefer an existing metadata id if present; otherwise synthesize
                md = getattr(d, "metadata", {}) or {}
                d.id = md.get("id") or f"doc-{i}-{uuid.uuid4().hex[:8]}"

        bm25 = BM25Retriever.from_documents(self.chunks)
        bm25.k = self.retrieval_k
        return bm25

    def _setup_hybrid_retriever(self) -> EnsembleRetriever:
        """Setup hybrid retriever combining sparse and dense methods."""

        # validate / normalize weight
        r = min(max(self.sparse_ratio, 0.0), 1.0)
        dense_w = 1.0 - r
        dense = self._setup_dense_retriever()
        sparse = self._setup_sparse_retriever()
        return EnsembleRetriever(
            retrievers=[sparse, dense],
            weights=[r, dense_w],
        )