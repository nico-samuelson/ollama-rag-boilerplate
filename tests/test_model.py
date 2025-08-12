from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
import model as mod


# --- Fixtures ---

@pytest.fixture(autouse=True)
def fake_hp(monkeypatch):
    """Make MODEL_NAME deterministic for tests."""
    monkeypatch.setattr(mod, "hp", SimpleNamespace(MODEL_NAME="fake-model"), raising=False)

# --- _call (non-streaming) ---

def test_call_success(monkeypatch):
    llm = mod.OllamaLLM(model="fake-model", temperature=0.7, max_new_tokens=256)

    mock_chat = MagicMock(return_value={"message": {"content": "hello back"}})
    monkeypatch.setattr(mod, "model", SimpleNamespace(chat=mock_chat), raising=False)

    out = llm._call("hello")
    assert out == "hello back"

    # Verify correct arguments sent to model.chat
    _, kwargs = mock_chat.call_args
    assert kwargs["model"] == "fake-model"
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert kwargs["options"] == {"temperature": 0.7, "num_predict": 256}
    assert kwargs["stream"] is False


def test_call_exception_returns_error_and_logs(monkeypatch, capsys):
    llm = mod.OllamaLLM(model="fake-model")
    monkeypatch.setattr(
        mod, "model",
        SimpleNamespace(chat=MagicMock(side_effect=RuntimeError("boom"))),
        raising=False,
    )

    out = llm._call("hi")
    assert out == "Error generating response."

    printed = capsys.readouterr().out
    assert "Ollama non-streaming generation error: boom" in printed


# --- _llm_type ---

def test_llm_type_property():
    llm = mod.OllamaLLM(model="fake-model")
    assert llm._llm_type == "ollama_llm"


# --- generate_stream (streaming) ---

def test_generate_stream_yields_content_and_uses_stream_true(monkeypatch):
    llm = mod.OllamaLLM(model="fake-model", temperature=0.1, max_new_tokens=9)

    # Build a fake streaming generator
    chunks = [
        {"message": {"content": "A"}},
        {"message": {}},  # no content -> should be skipped
        {"message": {"content": "B"}},
    ]

    recorded_kwargs = {}
    def fake_chat(**kwargs):
        recorded_kwargs.update(kwargs)
        def gen():
            yield {"message": {"content": "A"}}
            yield {"message": {}}
            yield {"message": {"content": "B"}}
        return gen()

    monkeypatch.setattr(mod, "model", SimpleNamespace(chat=fake_chat), raising=False)

    out = list(llm.generate_stream("go"))
    assert out == ["A", "B"]

    # Verify stream=True and options passed through
    assert recorded_kwargs["stream"] is True
    assert recorded_kwargs["model"] == "fake-model"
    assert recorded_kwargs["messages"] == [{"role": "user", "content": "go"}]
    assert recorded_kwargs["options"] == {"temperature": 0.1, "num_predict": 9}


def test_generate_stream_exception_yields_error_and_logs(monkeypatch, capsys):
    llm = mod.OllamaLLM(model="fake-model")
    monkeypatch.setattr(
        mod, "model",
        SimpleNamespace(chat=MagicMock(side_effect=RuntimeError("nope"))),
        raising=False,
    )

    out = list(llm.generate_stream("start"))
    assert out == ["Error generating response."]

    printed = capsys.readouterr().out
    assert "Ollama streaming generation error: nope" in printed


# --- setup_ollama_llm() ---

def test_setup_ollama_llm_success(capsys):
    llm = mod.setup_ollama_llm()
    assert isinstance(llm, mod.OllamaLLM)
    assert llm.model == "fake-model"

    printed = capsys.readouterr().out
    assert "Ollama LLM initialized with model: fake-model" in printed


def test_setup_ollama_llm_failure(monkeypatch, capsys):
    # Make the constructor explode to exercise the except path
    def bad_ctor(*args, **kwargs):
        raise RuntimeError("ctor error")

    monkeypatch.setattr(mod, "OllamaLLM", bad_ctor, raising=True)

    llm = mod.setup_ollama_llm()
    assert llm is None

    printed = capsys.readouterr().out
    assert "Ollama LLM setup failed: ctor error" in printed
    assert "Please ensure Ollama server is running and the model is pulled." in printed
