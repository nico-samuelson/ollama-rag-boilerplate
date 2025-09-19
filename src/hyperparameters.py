# Name of the LLM model to use with Ollama
# This should match the model name in your Ollama setup
# For complete list of models, see: https://ollama.com/library
MODEL_NAME = "gemma3:latest"

# Name of the embedding model for semantic search
# For complete list of embedding models, see: https://huggingface.co/spaces/mteb/leaderboard
EMBEDDING_NAME = 'Qwen/Qwen3-Embedding-0.6B'

# Name of the reranker model for refining search results
RERANKER_NAME = "Qwen/Qwen3-Reranker-0.6B"

# Size of document chunks for processing
CHUNK_SIZE = 1250

# Amount of overlap between chunks to maintain context
CHUNK_OVERLAP = 250

# Number of documents to retrieve during search
RETRIEVAL_K = 5

# Weight for BM25 (sparse) retriever in the ensemble
# 0.5 means 50% weight to sparse retrieval, 50% to dense
SPARSE_RATIO = 0.5

# Path to the document to be loaded and processed
DOCUMENT_PATH = "../data/HobbitBook.txt"