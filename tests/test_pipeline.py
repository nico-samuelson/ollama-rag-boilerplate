import types
import builtins
import pytest

# --- Helpers / fakes ---------------------------------------------------------

class FakeDoc:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata or {}

class FakeLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.loaded = None
        self.preprocessed = None

    def load_document(self, file_path):
        # Return what the test configured on the class variable
        return type(self).loaded

    def preprocess_document(self, document):
        # Optionally modify / pass through
        type(self).preprocessed = document
        return document

class FakeSplitter:
    """Returns whatever the test sets on FakeSplitter.to_return."""
    to_return = []

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def split_documents(self, docs):
        return list(type(self).to_return)

class FakeRetrieverShim:
    """What HybridRetriever.as_retriever() returns."""
    def __init__(self, invoke_return):
        self._invoke_return = invoke_return
        self.invocations = []

    def invoke(self, query):
        self.invocations.append(query)
        return list(self._invoke_return)

class FakeHybridRetriever:
    """Constructed with chunks; as_retriever() -> object with invoke()."""
    def __init__(self, chunks):
        self.chunks_ref = chunks  # will be list set by pipeline
        # Tests will set the class var to determine what invoke returns:
        # FakeHybridRetriever.invoke_return = [...]
        self.as_ret = FakeRetrieverShim(type(self).invoke_return)

    def as_retriever(self):
        return self.as_ret

class FakeReranker:
    """rerank(query, docs, k) -> (top_docs, scores)."""
    def __init__(self, name):
        self.name = name
        self.calls = []

    def rerank(self, query, docs, k):
        self.calls.append((query, list(docs), k))
        # Tests will set the class vars:
        # top_docs and scores. Default to echoing back.
        top = getattr(type(self), "top_docs", list(docs))
        scores = getattr(type(self), "scores", [1.0] * len(top))
        return (top, scores)

class FakeLLM:
    def __init__(self, model, temperature, max_new_tokens):
        self.model = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        # tests will set these attributes on the instance
        self._stream_tokens = ["ok"]
        self._call_result = "ok"

    def generate_stream(self, prompt):
        for t in self._stream_tokens:
            yield t

    def _call(self, prompt):
        return self._call_result


# --- Fixtures ----------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    """
    Patch heavy external deps inside rag_pipeline module with fakes.
    """
    import pipeline  # adjust if your module name differs

    # Swap in fakes
    monkeypatch.setattr(pipeline, "DocumentLoader", FakeLoader)
    monkeypatch.setattr(pipeline, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(pipeline, "RAGReranker", FakeReranker)
    monkeypatch.setattr(pipeline, "OllamaLLM", FakeLLM)
    monkeypatch.setattr(pipeline, "RecursiveCharacterTextSplitter", FakeSplitter)

    # Provide simple hp defaults if your real hp values are used in signature
    # (these were already bound at class definition time, but just in case)
    if not hasattr(pipeline, "hp"):
        hp_mod = types.SimpleNamespace(
            CHUNK_SIZE=500,
            CHUNK_OVERLAP=50,
            RERANKER_NAME="fake-reranker",
            MODEL_NAME="fake-model",
        )
        monkeypatch.setattr(pipeline, "hp", hp_mod, raising=False)

    # Ensure loader has something to load on import/initialization
    FakeLoader.loaded = [FakeDoc("A" * 120, {"page": 1})]  # one >100 chars
    FakeSplitter.to_return = list(FakeLoader.loaded)
    FakeHybridRetriever.invoke_return = []  # can override per-test
    FakeReranker.top_docs = list(FakeLoader.loaded)
    FakeReranker.scores = [0.9]

    yield


@pytest.fixture
def pipeline():
    import pipeline  # adjust if your module name differs
    return pipeline.RAGPipeline(file_path="dummy.txt")


# --- Tests -------------------------------------------------------------------

def test_split_document_filters_small_chunks(monkeypatch):
    import pipeline

    # Prepare a mix: one small (<100), one large (>=100)
    small = FakeDoc("short text", {"page": 2})
    large = FakeDoc("X" * 150, {"page": 3})
    FakeLoader.loaded = [small, large]
    FakeSplitter.to_return = [small, large]

    pipe = pipeline.RAGPipeline(file_path="dummy.txt")

    # Only large should remain
    assert len(pipe.chunks) == 1
    assert pipe.chunks[0].page_content == large.page_content
    assert pipe.chunk_size == pipeline.hp.CHUNK_SIZE
    assert pipe.chunk_overlap == pipeline.hp.CHUNK_OVERLAP


def test_format_docs_includes_metadata_and_order(pipeline):
    docs = [
        FakeDoc("Hello world", {"page": 5}),
        FakeDoc("Second doc with body", {"page": 9}),
    ]
    formatted = pipeline.format_docs(docs)
    # Contains both sources with page metadata
    assert "Source 1 (Page 5):" in formatted
    assert "Hello world" in formatted
    assert "Source 2 (Page 9):" in formatted
    assert "Second doc with body" in formatted


def test_retrieve_relevant_docs_calls_retriever_and_reranker(patch_dependencies, monkeypatch):
    import pipeline  # your module name

    retrieved = [
        FakeDoc("Doc A " + "A"*120, {"page": 1}),
        FakeDoc("Doc B " + "B"*120, {"page": 2}),
        FakeDoc("Doc C " + "C"*120, {"page": 3}),
    ]
    FakeHybridRetriever.invoke_return = list(retrieved)
    FakeReranker.top_docs = retrieved[:2]
    FakeReranker.scores = [0.8, 0.7]

    pipe = pipeline.RAGPipeline(file_path="dummy.txt")  # construct *after* setting
    got_retrieved, (top_docs, scores) = pipe.retrieve_relevant_docs("what is A?")

    assert got_retrieved == retrieved
    assert top_docs == retrieved[:2]
    assert scores == [0.8, 0.7]


def test_prepare_prompt_injects_context_and_query(monkeypatch, pipeline):
    # Mock format_docs to return a fixed string
    monkeypatch.setattr(pipeline, "format_docs", lambda d: "CTX-BODY")
    prompt = pipeline.prepare_prompt("Why?", [FakeDoc("irrelevant")])
    assert "CTX-BODY" in prompt
    assert "Question: Why?" in prompt


def test_generate_response_with_progress_streaming_path(monkeypatch, pipeline):
    # Setup retrieval & rerank to yield one doc context
    doc = FakeDoc("Context " + "X"*120, {"page": 1})
    FakeHybridRetriever.invoke_return = [doc]
    FakeReranker.top_docs = [doc]
    FakeReranker.scores = [0.99]

    # Configure streaming tokens
    pipeline.llm._stream_tokens = ["Hello", " ", "World!"]

    resp, sources = pipeline.generate_response_with_progress("greet me")
    assert resp == "Hello World!"
    assert sources == [doc]


def test_generate_response_with_progress_fallback_on_exception(monkeypatch, pipeline):
    # Retrieval/rerank
    doc = FakeDoc("Ctx " + "Y"*120, {"page": 2})
    FakeHybridRetriever.invoke_return = [doc]
    FakeReranker.top_docs = [doc]
    FakeReranker.scores = [0.9]

    # Make generate_stream raise
    def boom(_):
        raise RuntimeError("streaming broken")

    pipeline.llm.generate_stream = boom
    pipeline.llm._call_result = "Fallback OK"

    resp, sources = pipeline.generate_response_with_progress("test")
    assert resp == "Fallback OK"
    assert sources == [doc]


def test_generate_response_non_streaming(monkeypatch, pipeline):
    # Retrieval/rerank
    doc = FakeDoc("Ctx " + "Z"*120, {"page": 3})
    FakeHybridRetriever.invoke_return = [doc]
    FakeReranker.top_docs = [doc]
    FakeReranker.scores = [0.95]

    pipeline.llm._call_result = "Non-streamed answer"
    resp, sources = pipeline.generate_response("q")
    assert resp == "Non-streamed answer"
    assert sources == [doc]


def test_interactive_streaming_loop_quits_immediately(monkeypatch, pipeline):
    # Make input return 'quit' the first time
    inputs = iter(["quit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))

    # Should not raise / hang
    pipeline.interactive_streaming_loop()
