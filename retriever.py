import torch
import hyperparameters as hp
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

def setup_gpu_embeddings(chunks):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = torch.device("mps") # USE THIS WHILE RUNNING ON MAC SYSTEM

        embeddings = SentenceTransformerEmbeddings(
            model_name=hp.EMBEDDING_NAME,
            model_kwargs={'device': device, "trust_remote_code": True},
            encode_kwargs={'batch_size': 32 if device == "cuda" else 16, "trust_remote_code": True},
        )

        persist_directory = "./chroma_db"
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": hp.RETRIEVAL_K})
        print(f"Embeddings using {device}")
        return dense_retriever

    except Exception as e:
        print(f"GPU embeddings failed: {e}")
        return setup_cpu_embeddings(chunks)

def setup_cpu_embeddings(chunks):
    embeddings = SentenceTransformerEmbeddings(model_name=hp.EMBEDDING_NAME, model_kwargs={"trust_remote_code": True})
    persist_directory = "./chroma_db"
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    print("Embeddings using CPU (fallback)")
    return vectorstore.as_retriever(search_kwargs={"k": hp.RETRIEVAL_K})