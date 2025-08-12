from types import SimpleNamespace
from unittest.mock import MagicMock
import builtins
import pytest
import pipeline as mod


# --- helpers ----

class Doc:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata or {}


@pytest.fixture(autouse=True)
def fake_hp(monkeypatch):
    monkeypatch.setattr(
        mod, "hp",
        SimpleNamespace(
            CHUNK_SIZE=200,
            CHUNK_OVERLAP=20,
            RETRIEVAL_K=5,
            SPARSE_RATIO=0.25,
        ),
        raising=False,
    )


# -------------------------
# build_rag_pipeline
# -------------------------

def test_build_rag_pipeline_happy_path(monkeypatch, capsys):
    # loader -> docs -> preprocessed
    docs = [Doc("x" * 500, metadata={"page": 1})]
    monkeypatch.setattr(mod.loader, "load_document", MagicMock(return_value=docs))
    monkeypatch.setattr(mod.loader, "preprocess_document", MagicMock(return_value=docs))

    # text splitter returns short + long chunks; only long (>100) should survive
    short_chunk = Doc("a" * 80, metadata={"page": 2})
    long_chunk = Doc("b" * 150, metadata={"page": 3})

    rc_instance = SimpleNamespace(split_documents=MagicMock(return_value=[short_chunk, long_chunk]))
    rc_ctor = MagicMock(return_value=rc_instance)
    monkeypatch.setattr(mod, "RecursiveCharacterTextSplitter", rc_ctor)

    # dense retriever (GPU embeddings)
    monkeypatch.setattr(mod.retriever, "setup_gpu_embeddings", MagicMock(return_value="DENSE"))

    # BM25 sparse retriever with assignable k
    sparse_obj = SimpleNamespace(k=None)
    bm25 = SimpleNamespace(from_documents=MagicMock(return_value=sparse_obj))
    monkeypatch.setattr(mod, "BM25Retriever", bm25)

    # Ensemble returns a sentinel
    ens_ctor = MagicMock(return_value="HYBRID")
    monkeypatch.setattr(mod, "EnsembleRetriever", ens_ctor)

    # LLM setup returns a sentinel
    monkeypatch.setattr(mod.model, "setup_ollama_llm", MagicMock(return_value="LLM"))

    hybrid, llm = mod.build_rag_pipeline("dummy.pdf")

    # Returns
    assert hybrid == "HYBRID"
    assert llm == "LLM"

    # RecursiveCharacterTextSplitter called with hp config + separators
    args, kwargs = rc_ctor.call_args
    assert kwargs["chunk_size"] == 200
    assert kwargs["chunk_overlap"] == 20
    assert kwargs["separators"] == ["\n\n", "\n", " ", ""]

    # BM25 built from surviving chunks: only the long one
    mod.BM25Retriever.from_documents.assert_called_once()
    (passed_chunks,), _ = mod.BM25Retriever.from_documents.call_args
    assert passed_chunks == [long_chunk]

    # k set from hp
    assert sparse_obj.k == 5

    # Ensemble receives sparse + dense with proper weights
    ens_args, ens_kwargs = ens_ctor.call_args
    assert ens_kwargs["retrievers"] == [sparse_obj, "DENSE"]
    assert ens_kwargs["weights"] == [0.25, 0.75]

    out = capsys.readouterr().out
    assert "Loading documents..." in out
    assert "Splitting documents into chunks..." in out
    assert "After filtering: 1 chunks remain." in out
    assert "Setting up retrievers..." in out
    assert "Setting up Ollama LLM..." in out


def test_build_rag_pipeline_llm_none_raises(monkeypatch):
    docs = [Doc("x" * 500)]
    monkeypatch.setattr(mod.loader, "load_document", MagicMock(return_value=docs))
    monkeypatch.setattr(mod.loader, "preprocess_document", MagicMock(return_value=docs))

    rc_instance = SimpleNamespace(split_documents=MagicMock(return_value=[Doc("y" * 200)]))
    monkeypatch.setattr(mod, "RecursiveCharacterTextSplitter", MagicMock(return_value=rc_instance))

    monkeypatch.setattr(mod.retriever, "setup_gpu_embeddings", MagicMock(return_value="DENSE"))
    bm25 = SimpleNamespace(from_documents=MagicMock(return_value=SimpleNamespace(k=None)))
    monkeypatch.setattr(mod, "BM25Retriever", bm25)
    monkeypatch.setattr(mod, "EnsembleRetriever", MagicMock(return_value="HYBRID"))

    # LLM setup fails -> RuntimeError expected
    monkeypatch.setattr(mod.model, "setup_ollama_llm", MagicMock(return_value=None))

    with pytest.raises(RuntimeError, match="No LLM could be initialized"):
        mod.build_rag_pipeline("dummy.pdf")


# -------------------------
# format_docs
# -------------------------

def test_format_docs_formats_sources_and_pages(capsys):
    docs = [
        Doc("Alpha", metadata={"page": 3}),       # has page
        Doc("Beta", metadata={}),                 # falsy metadata -> no page label
        Doc("Gamma", metadata={"foo": "bar"}),    # truthy metadata w/o 'page' -> Unknown
    ]

    out = mod.format_docs(docs)
    # Source numbering and page labeling
    assert "Source 1 (Page 3):" in out
    assert "Source 2:" in out and "(Page " not in out.split("Source 2:")[1].split("\n")[0]
    assert "Source 3 (Page Unknown):" in out
    assert "Alpha" in out and "Beta" in out and "Gamma" in out

    printed = capsys.readouterr().out
    assert "Context length:" in printed


# -------------------------
# generate_response_with_progress (streaming)
# -------------------------

def test_generate_response_with_progress_streams_and_counts_words(monkeypatch, capsys):
    # retriever that returns two docs
    docs = [Doc("Doc1 text"), Doc("Doc2 text", metadata={"page": 2})]
    retr = SimpleNamespace(invoke=MagicMock(return_value=docs))

    # llm streaming output
    class FakeLLM:
        def generate_stream(self, prompt):
            assert "Context:" in prompt and "Question:" in prompt
            yield "Hello"
            yield " "
            yield "world!"

    llm = FakeLLM()

    full, returned_docs = mod.generate_response_with_progress("What is up?", retr, llm)
    assert full == "Hello world!"
    assert returned_docs == docs

    out = capsys.readouterr().out
    assert "AI RESPONSE (Streaming)" in out
    assert "Found 2 documents" in out
    assert "Generated" in out  # word count line


def test_generate_response_with_progress_fallback_on_exception(monkeypatch, capsys):
    docs = [Doc("Doc text")]
    retr = SimpleNamespace(invoke=MagicMock(return_value=docs))

    class FakeLLM:
        def generate_stream(self, _):
            raise RuntimeError("stream broke")
        def _call(self, _):
            return "FALLBACK ANSWER"

    llm = FakeLLM()

    full, returned_docs = mod.generate_response_with_progress("Q", retr, llm)
    assert full == "FALLBACK ANSWER"
    assert returned_docs == docs

    out = capsys.readouterr().out
    assert "Streaming error: stream broke" in out
    assert "Falling back to standard generation" in out


# -------------------------
# generate_response (non-streaming)
# -------------------------

def test_generate_response_calls_llm_with_final_answer(monkeypatch, capsys):
    docs = [Doc("Doc A"), Doc("Doc B")]
    retr = SimpleNamespace(invoke=MagicMock(return_value=docs))

    called = {}
    class FakeLLM:
        def _call(self, prompt):
            called["prompt"] = prompt
            return "NONSTREAM ANSWER"

    llm = FakeLLM()

    full, returned_docs = mod.generate_response("Why?", retr, llm)
    assert full == "NONSTREAM ANSWER"
    assert returned_docs == docs
    assert "Final Answer:" in called["prompt"]

    out = capsys.readouterr().out
    assert "ANSWER (NON-STREAMING)" in out
    assert "Retrieved 2 documents" in out


# -------------------------
# interactive_streaming_loop (commands only)
# -------------------------

def test_interactive_streaming_loop_command_toggles(monkeypatch, capsys):
    # Provide inputs: toggle sources, toggle streaming, blank, quit
    inputs = iter(["sources", "stream", "", "quit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))

    # Dummy objects (we won't hit generation paths here)
    retr = SimpleNamespace()
    llm = SimpleNamespace()

    mod.interactive_streaming_loop(retr, llm)

    out = capsys.readouterr().out
    assert "RAG Pipeline ready" in out
    assert "Source documents disabled" in out
    assert "Switched to standard mode" in out
