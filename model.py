import model
import hyperparameters as hp
from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM

class OllamaLLM(LLM):
    model: str = hp.MODEL_NAME
    host: str = "http://localhost:11434"
    temperature: float = 0.5
    max_new_tokens: int = 1024

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]
        try:
            response = model.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_new_tokens,
                },
                stream=False # For non-streaming call
            )
            return response['message']['content']
        except Exception as e:
            print(f"Ollama non-streaming generation error: {e}")
            return "Error generating response."

    @property
    def _llm_type(self) -> str:
        return "ollama_llm"

    def generate_stream(self, prompt: str):
        """Generate response with real-time streaming output using Ollama."""
        messages = [
            {"role": "user", "content": prompt}
        ]
        try:
            for chunk in model.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_new_tokens,
                },
                stream=True # Enable streaming
            ):
                if 'content' in chunk['message']:
                    yield chunk['message']['content']
        except Exception as e:
            print(f"Ollama streaming generation error: {e}")
            yield "Error generating response."

def setup_ollama_llm():
    try:
        llm = OllamaLLM(
            model=hp.MODEL_NAME,
            temperature=0.5,
            max_new_tokens=1024,
        )
        print(f"Ollama LLM initialized with model: {llm.model}")
        return llm

    except Exception as e:
        print(f"Ollama LLM setup failed: {e}")
        print("Please ensure Ollama server is running and the model is pulled.")
        return None