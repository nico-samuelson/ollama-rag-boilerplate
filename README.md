# RAG-Boilerplate

A robust, modular Retrieval-Augmented Generation (RAG) boilerplate for building, evaluating, and deploying RAG pipelines with your own data. This project is designed for rapid prototyping and experimentation with RAG architectures, making it easy to plug in different retrievers, models, and evaluation strategies.

## Features

- **Plug-and-play architecture:** Swap out retrievers, models, and pipelines with minimal code changes.
- **ChromaDB integration:** Efficient vector storage and retrieval using ChromaDB.
- **Evaluation suite:** Built-in tools for evaluating RAG pipeline performance.
- **Jupyter notebook demo:** Quick start and experimentation in `src/demo.ipynb`.
- **Extensible:** Easily add new retrievers, models, or data loaders.

## Project Structure

```
RAG-Boilerplate/
├── data/                # Data files (e.g., HobbitBook.txt, hobbit_qna.json)
├── src/                 # Source code
│   ├── demo.ipynb       # Jupyter notebook demo
│   ├── evaluate.py      # Evaluation utilities
│   ├── hyperparameters.py # Hyperparameter configs
│   ├── loader.py        # Data loading utilities
│   ├── model.py         # Model wrapper(s)
│   ├── pipeline.py      # RAG pipeline logic
│   └── retriever.py     # Retriever implementations
├── tests/               # Unit tests
├── requirements.txt     # Python dependencies
├── pytest.ini           # Pytest configuration
└── README.md            # Project documentation
```

## Quickstart

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the demo notebook:**
   Open `src/demo.ipynb` in Jupyter and follow the cells to see the RAG pipeline in action.

3. **Run tests:**
   ```bash
   pytest
   ```

## Customization

- **Add new retrievers:** Implement and register in `retriever.py`.
- **Swap models:** Update or extend `model.py` to use your preferred LLM.
- **Change data:** Place your own documents and Q&A in the `data/` folder.

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies.

## License

MIT License

---
