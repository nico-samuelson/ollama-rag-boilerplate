import json
import math
import numpy as np
import pytest

# ----------------- Fakes & helpers -----------------

class FakeDoc:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata or {}

class FakeRetrieverShim:
    def __init__(self, results):
        self._results = results
        self.calls = []

    def invoke(self, query):
        self.calls.append(query)
        return list(self._results)

class FakeRetriever:
    def __init__(self, results):
        self._results = results

    def as_retriever(self):
        return FakeRetrieverShim(self._results)

class FakeReranker:
    """Return (top_docs, scores) when .rerank(...) is called."""
    def __init__(self, top_docs, scores=None):
        self.top_docs = list(top_docs)
        self.scores = scores or [1.0] * len(top_docs)
        self.calls = []

    def rerank(self, query, docs, k):
        self.calls.append((query, list(docs), k))
        # Emulate a top-k by truncating the configured list
        return (self.top_docs[:k], self.scores[:k])

class FakePipeline:
    """Minimal pipeline stub with retriever, reranker, and generate_response()."""
    def __init__(self, retrieved_docs, reranked_docs, gen_answers=None):
        self.retriever = FakeRetriever(retrieved_docs)
        self.reranker = FakeReranker(reranked_docs)
        # gen_answers: list of strings to return per query
        self._answers = gen_answers or []

    def generate_response(self, query):
        # Return next configured answer or a default
        ans = self._answers.pop(0) if self._answers else "Final Answer: default"
        # second value is "source docs"; not used in evaluation
        return ans, []

class Tensorish:
    """Mimics a torch tensor just enough for evaluator.encode(..., convert_to_tensor=True)."""
    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=float)

    def cpu(self):
        return self

    def reshape(self, *shape):
        return self.arr.reshape(*shape)

class FakeSentenceTransformer:
    def __init__(self, name):
        self.name = name

    def encode(self, text, convert_to_tensor=True):
        # Deterministic simple embeddings based on text length parity.
        # Short & sweet so tests are predictable and fast.
        if "hello" in text.lower():
            vec = [1.0, 0.0, 0.0]
        elif "world" in text.lower():
            vec = [0.0, 1.0, 0.0]
        else:
            vec = [0.0, 0.0, 1.0]
        return Tensorish(vec)

# ----------------- Unit tests for primitives -----------------

def test_dcg_matches_manual_calc():
    from evaluate import dcg
    rel = [3, 2, 1]
    # manual calculation
    discounts = np.log2(np.arange(2, len(rel) + 2))
    expected = np.sum((2**np.array(rel) - 1) / discounts)
    assert dcg(rel) == pytest.approx(expected)

def test_ndcg_basic_and_cutoff():
    from evaluate import ndcg
    # retrieved docs with ground-truth snippets embedded
    retrieved = [
        ".... contains GT1 ....",
        "no match here",
        ".... contains GT2 ....",
    ]
    ground_truth = ["GT1", "GT2"]

    # At k=2, relevances = [1, 0], ideal = [1, 0] -> NDCG@2 = 1.0
    expected = 1.0 / (1.0 + 1.0 / np.log2(3))
    assert ndcg(retrieved, ground_truth, k=2) == pytest.approx(expected)

    # At full length, relevances [1,0,1], ideal [1,1,0]
    ndcg_full = ndcg(retrieved, ground_truth, k=None)
    assert 0 < ndcg_full < 1  # strictly between (since not ideal order)

    # No relevant docs => idcg == 0 => ndcg -> 0.0
    assert ndcg(["a", "b"], ["ZZZ"], k=2) == 0.0

# ----------------- evaluate_retriever -----------------

def test_evaluate_retriever_single_query_metrics(tmp_path, monkeypatch):
    import evaluate

    # Create a simple eval dataset with one query and two ground-truth snippets
    eval_items = [{
        "query": "Q1",
        "relevant_chunks": [
            {"text_snippet": "GT1"},
            {"text_snippet": "GT2"},
        ],
    }]
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(eval_items), encoding="utf-8")

    # Retrieved docs (before rerank)
    retrieved_docs = [
        FakeDoc("GT1 plus some text"),      # relevant
        FakeDoc("irrelevant stuff"),        # not relevant
        FakeDoc("later has GT2 somewhere")  # relevant, but not in top-2 after rerank
    ]

    # Reranker will pick the first two only (so top-k=2 includes only GT1)
    reranked_docs = [retrieved_docs[0], retrieved_docs[1]]

    pipe = FakePipeline(retrieved_docs=retrieved_docs, reranked_docs=reranked_docs)

    # Run evaluation at k=2
    metrics = evaluate.evaluate_retriever(pipe, str(p), k_eval=2)

    # Expectations:
    # - Hit rate: 100% (GT1 appears in top-2)
    # - MRR: 1.0   (first relevant at rank 1)
    # - NDCG@2: relevances [1,0] vs ideal [1,0] -> 1.0
    # - Recall@2: matched GTs in top-2 = {GT1} / 2 total => 0.5
    assert metrics["hit_rate"] == pytest.approx(100.0)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["ndcg@2"] == pytest.approx(1.0)
    assert metrics["recall@2"] == pytest.approx(0.5)

def test_evaluate_retriever_aggregates_multiple_queries(tmp_path):
    import evaluate

    # Two queries:
    # Q1 has 1 GT, appears at rank 2 in top-2 -> hit, MRR=1/2, NDCG@2<1, recall@2=1/1=1
    # Q2 has no GT -> ndcg/recall terms ignored in the average (handled via counts)
    eval_items = [
        {
            "query": "Q1",
            "relevant_chunks": [{"text_snippet": "GTX"}],
        },
        {
            "query": "Q2",
            "relevant_chunks": [],  # no ground truth chunks
        },
    ]
    p = tmp_path / "eval2.json"
    p.write_text(json.dumps(eval_items), encoding="utf-8")

    # Build a pipeline whose retriever returns some docs, and reranker chooses top-2
    retrieved = [
        FakeDoc("no match"),
        FakeDoc("... GTX ..."),  # relevant appears at position 2 in top-2
        FakeDoc("no match again"),
    ]
    reranked = [retrieved[0], retrieved[1]]  # top-2

    pipe = FakePipeline(retrieved_docs=retrieved, reranked_docs=reranked)

    metrics = evaluate.evaluate_retriever(pipe, str(p), k_eval=2)

    # Aggregate expectations:
    # total_queries = 2
    # hits: Q1 hit -> 1/2 => 50%
    assert metrics["hit_rate"] == pytest.approx(50.0)

    # MRR: Q1 contributes 1/2, Q2 contributes 0 -> average over total_queries=2 => 0.25
    assert metrics["mrr"] == pytest.approx(0.25)

    # NDCG@2: only Q1 counts (Q2 has no GT, skipped in ndcg_count)
    # Q1 relevances in top-2: [0,1]; ideal [1,0]
    # DCG = (2^0-1)/log2(2) + (2^1-1)/log2(3) = 0 + 1/log2(3)
    # IDCG = 1/1 + 0/log2(3) = 1
    expected_ndcg_q1 = 1 / math.log2(3)
    assert metrics["ndcg@2"] == pytest.approx(expected_ndcg_q1)

    # Recall@2: only Q1 counts -> matched {GTX}/1 => 1.0
    assert metrics["recall@2"] == pytest.approx(1.0)

# ----------------- evaluate_generation -----------------

def test_evaluate_generation_strips_prefix_and_averages(tmp_path, monkeypatch):
    import evaluate

    # Patch SentenceTransformer with a fake to avoid downloads/heavy deps
    monkeypatch.setattr(evaluate, "SentenceTransformer", FakeSentenceTransformer)

    # Minimal hp with embedding name, if not present
    if not hasattr(evaluate, "hp"):
        class HP:
            EMBEDDING_NAME = "fake"
        evaluate.hp = HP()

    # Dataset of two items; answers will be generated by fake pipeline
    eval_items = [
        {"query": "q1", "ground_truth_answer": "hello"},
        {"query": "q2", "ground_truth_answer": "world"},
    ]
    p = tmp_path / "gen_eval.json"
    p.write_text(json.dumps(eval_items), encoding="utf-8")

    # Fake pipeline returns answers with and without the "Final Answer:" prefix
    answers = [
        "Final Answer: hello there",  # should be stripped to "hello there"
        "world it is",                # no prefix, left as is
    ]
    pipe = FakePipeline(retrieved_docs=[], reranked_docs=[], gen_answers=answers)

    avg = evaluate.evaluate_generation(pipe, str(p))

    # Our FakeSentenceTransformer encodes "hello*" to [1,0,0] and "world*" to [0,1,0].
    # The two generated answers map to the *same* one-hot as their GTs, making
    # cosine similarity 1.0 for both -> average 1.0.
    assert avg == pytest.approx(1.0)
