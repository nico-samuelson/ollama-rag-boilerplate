from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
import retriever as mod


@pytest.fixture(autouse=True)
def fake_hp(monkeypatch):
    # Ensure predictable hyperparameters for tests
    monkeypatch.setattr(
        mod, "hp",
        SimpleNamespace(EMBEDDING_NAME="fake-embed-model", RETRIEVAL_K=7),
        raising=False
    )


def _make_fake_vectorstore():
    # A minimal vectorstore with as_retriever()
    retriever = object()
    fake_vs = SimpleNamespace(as_retriever=Mock(return_value=retriever))
    return fake_vs, retriever


def test_gpu_path_uses_cuda_and_batchsize_32(capsys):
    chunks = ["c1", "c2"]
    fake_vs, retriever_obj = _make_fake_vectorstore()

    with patch.object(mod.torch.cuda, "is_available", return_value=True), \
         patch.object(mod, "SentenceTransformerEmbeddings", autospec=True) as STE, \
         patch.object(mod, "Chroma", autospec=True) as Chroma:

        Chroma.from_documents.return_value = fake_vs

        result = mod.setup_gpu_embeddings(chunks)

        # Returned retriever is whatever vectorstore.as_retriever() yielded
        assert result is retriever_obj

        # Embeddings constructed with proper device + batch_size 32
        assert STE.call_count == 1
        kwargs = STE.call_args.kwargs
        assert kwargs["model_name"] == "fake-embed-model"
        assert kwargs["model_kwargs"] == {"device": "cuda", "trust_remote_code": True}
        assert kwargs["encode_kwargs"] == {"batch_size": 32, "trust_remote_code": True}

        # Chroma called with persist_directory
        Chroma.from_documents.assert_called_once()
        _, ch_kwargs = Chroma.from_documents.call_args
        assert ch_kwargs["persist_directory"] == "./chroma_db"

        out = capsys.readouterr().out
        assert "Embeddings using cuda" in out


def test_try_path_on_cpu_sets_batchsize_16_and_prints_cpu(capsys):
    chunks = ["c1"]

    fake_vs, retriever_obj = _make_fake_vectorstore()

    with patch.object(mod.torch.cuda, "is_available", return_value=False), \
         patch.object(mod, "SentenceTransformerEmbeddings", autospec=True) as STE, \
         patch.object(mod, "Chroma", autospec=True) as Chroma:

        Chroma.from_documents.return_value = fake_vs

        result = mod.setup_gpu_embeddings(chunks)
        assert result is retriever_obj

        kwargs = STE.call_args.kwargs
        assert kwargs["model_kwargs"]["device"] == "cpu"
        assert kwargs["encode_kwargs"]["batch_size"] == 16

        out = capsys.readouterr().out
        # In the try-path it prints "cpu" (fallback prints a different message)
        assert "Embeddings using cpu" in out


def test_gpu_path_exception_falls_back_to_cpu(monkeypatch, capsys):
    chunks = ["c1"]

    # Force an exception inside the try-block
    with patch.object(mod, "SentenceTransformerEmbeddings", side_effect=RuntimeError("boom")), \
         patch.object(mod, "setup_cpu_embeddings", autospec=True) as fallback:

        fallback.return_value = "CPU_RET"
        result = mod.setup_gpu_embeddings(chunks)

        assert result == "CPU_RET"
        fallback.assert_called_once_with(chunks)

        out = capsys.readouterr().out
        assert "GPU embeddings failed: boom" in out


def test_setup_cpu_embeddings_builds_vectorstore_and_uses_k(capsys):
    chunks = ["c1", "c2"]
    fake_vs, retriever_obj = _make_fake_vectorstore()

    with patch.object(mod, "SentenceTransformerEmbeddings", autospec=True) as STE, \
         patch.object(mod, "Chroma", autospec=True) as Chroma:

        Chroma.from_documents.return_value = fake_vs

        result = mod.setup_cpu_embeddings(chunks)
        assert result is retriever_obj

        # CPU builder does NOT pass device; only trust_remote_code
        kwargs = STE.call_args.kwargs
        assert kwargs["model_name"] == "fake-embed-model"
        assert kwargs["model_kwargs"] == {"trust_remote_code": True}
        assert "encode_kwargs" not in kwargs  # not used in CPU helper

        # as_retriever is called with RETRIEVAL_K from hp
        fake_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 7})

        # Persist directory wired through
        _, ch_kwargs = Chroma.from_documents.call_args
        assert ch_kwargs["persist_directory"] == "./chroma_db"

        out = capsys.readouterr().out
        assert "Embeddings using CPU (fallback)" in out
