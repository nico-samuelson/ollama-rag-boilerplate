import json
from types import SimpleNamespace
from unittest.mock import MagicMock
import torch
import pytest
import evaluate as mod


# --------- Helpers ---------

class Doc:
    def __init__(self, text):
        self.page_content = text
        self.metadata = {}


@pytest.fixture(autouse=True)
def fake_hp(monkeypatch):
    # Ensure deterministic hyperparameters
    monkeypatch.setattr(mod, "hp", SimpleNamespace(EMBEDDING_NAME="fake-embed"), raising=False)


# ===========================
# evaluate_retriever
# ===========================

def test_evaluate_retriever_hits_and_mrr(tmp_path, capsys):
    # Build eval dataset with two queries
    eval_data = [
        {
            "query": "q1",
            "relevant_chunks": [{"text_snippet": "needle"}],
        },
        {
            "query": "q2",
            "relevant_chunks": [{"text_snippet": "xyz"}],
        },
    ]
    p = tmp_path / "retrieval.json"
    p.write_text(json.dumps(eval_data))

    # Fake retriever: for q1 -> first doc hits (rank 1); for q2 -> second doc hits (rank 2)
    def _invoke(query):
        if query == "q1":
            return [Doc("... needle ..."), Doc("irrelevant")]
        if query == "q2":
            return [Doc("nope"), Doc("contains xyz here")]
        return []

    retriever = SimpleNamespace(invoke=_invoke)

    hit_rate, mrr = mod.evaluate_retriever(retriever, str(p))

    # q1 and q2 both hit => 100% hit rate
    assert pytest.approx(hit_rate, rel=1e-6) == 100.0
    # MRR = (1/1 + 1/2) / 2 = 0.75
    assert pytest.approx(mrr, rel=1e-6) == 0.75

    out = capsys.readouterr().out
    assert "Evaluating Retriever Performance..." in out
    assert "Retrieval Evaluation Summary" in out
    assert "Hit Rate" in out
    assert "Mean Reciprocal Rank" in out


def test_evaluate_retriever_no_hits(tmp_path):
    eval_data = [
        {"query": "q", "relevant_chunks": [{"text_snippet": "zzz"}]},
    ]
    p = tmp_path / "retrieval_nohits.json"
    p.write_text(json.dumps(eval_data))

    retriever = SimpleNamespace(invoke=lambda q: [Doc("no match here"), Doc("still no")])
    hit_rate, mrr = mod.evaluate_retriever(retriever, str(p))

    assert hit_rate == 0.0
    assert mrr == 0.0


# ===========================
# evaluate_generation
# ===========================

def test_evaluate_generation_avg_similarity(tmp_path, monkeypatch, capsys):
    # Two examples: one exact match (should give cosine 1.0), one completely different (cosine ~0.0)
    eval_data = [
        {
            "query": "q1",
            "ground_truth_answer": "Paris is the capital of France.",
        },
        {
            "query": "q2",
            "ground_truth_answer": "Blue sky",
        },
    ]
    p = tmp_path / "gen.json"
    p.write_text(json.dumps(eval_data))

    # Stub generate_response:
    # - First returns with the "Final Answer:" prefix (to exercise stripping)
    # - Second returns a different answer with no prefix
    def fake_generate_response(query, retriever, llm):
        if query == "q1":
            return ("Final Answer: Paris is the capital of France.", []), []
        else:
            return ("Totally unrelated answer", []), []

    monkeypatch.setattr(mod, "generate_response", lambda q, r, l: fake_generate_response(q, r, l)[0])

    # Fake SentenceTransformer that returns controlled vectors
    class FakeST:
        def __init__(self, model_name, device=None):
            self.model_name = model_name
            self.device = device
        def encode(self, text, convert_to_tensor=True):
            # Map specific strings to orthogonal vectors
            if text.strip() == "Paris is the capital of France.":
                return torch.tensor([1.0, 0.0])
            if text.strip() == "Totally unrelated answer":
                return torch.tensor([0.0, 1.0])
            if text.strip() == "Blue sky":
                return torch.tensor([1.0, 0.0])  # ground truth for q2
            # Fallback
            return torch.tensor([0.0, 0.0])

    monkeypatch.setattr(mod, "SentenceTransformer", FakeST, raising=True)
    # Ensure device selection path is stable
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False, raising=False)

    # Dummy retriever/llm; they aren't used directly by our fake generator
    retriever = object()
    llm = object()

    avg = mod.evaluate_generation(retriever, llm, str(p))

    # First pair identical → cosine 1; second pair orthogonal → 0; average = 0.5
    assert pytest.approx(avg, rel=1e-6) == 0.5

    out = capsys.readouterr().out
    assert "Evaluating End-to-End Generation Quality..." in out
    assert "Average Semantic Similarity" in out


def test_evaluate_generation_handles_answer_without_prefix(tmp_path, monkeypatch):
    # Single sample where generated answer has no "Final Answer:" prefix
    eval_data = [
        {"query": "q1", "ground_truth_answer": "Hello world"},
    ]
    p = tmp_path / "gen_noprefix.json"
    p.write_text(json.dumps(eval_data))

    monkeypatch.setattr(mod, "generate_response", lambda q, r, l: ("Hello world", []), raising=True)

    class FakeST:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, text, convert_to_tensor=True):
            # Both identical -> cosine 1
            return torch.tensor([1.0, 0.0])

    monkeypatch.setattr(mod, "SentenceTransformer", FakeST, raising=True)
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False, raising=False)

    avg = mod.evaluate_generation(object(), object(), str(p))
    assert pytest.approx(avg, rel=1e-6) == 1.0
