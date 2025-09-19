import json
import numpy as np
import hyperparameters as hp
from typing import List, Dict
from pipeline import RAGPipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def dcg(relevances: List[int], k: int = None) -> float:
    if k is not None:
        relevances = relevances[:k]
    relevances = np.array(relevances)
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return np.sum((2**relevances - 1) / discounts)

def ndcg(retrieved_docs: List[str], ground_truth_docs: List[str], k: int = None) -> float:
    """
    Compute NDCG for RAG retrieval

    Args:
        retrieved_docs (list): Retrieved document IDs in ranked order
        ground_truth_docs (list or set): Ground-truth relevant document IDs
        k (int, optional): Rank cutoff (e.g., 5 for NDCG@5)

    Returns:
        float: NDCG score
    """
    # Convert to binary relevance labels (1 = relevant, 0 = not relevant)
    # relevances = [1 if doc in ground_truth_docs else 0 for doc in retrieved_docs]

    relevances = [1 if any(gt_chunk in doc for gt_chunk in ground_truth_docs) else 0 for doc in retrieved_docs]

    # DCG for retrieved order
    dcg_score = dcg(relevances, k)

    # Ideal DCG (best possible ordering: all relevant first)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg_score = dcg(ideal_relevances, k)

    return dcg_score / idcg_score if idcg_score > 0 else 0.0

def evaluate_retriever(pipeline: RAGPipeline, eval_dataset_path: str, k_eval: int = 5) -> Dict[str, float]:
    """
    Evaluates the performance of the retriever using a ground-truth dataset.

    Args:
        pipeline (RAGPipeline): The RAG pipeline containing the retriever and reranker
        eval_dataset_path (str): Path to the evaluation dataset
        k_eval (int): Cutoff for NDCG@k and recall@k

    Returns:
        dict: Dictionary with evaluation metrics (Hit Rate, MRR, NDCG@k, Recall@k)
    """

    print("\n" + "="*50)
    print("Evaluating Retriever Performance...")
    print("="*50)

    with open(eval_dataset_path, 'r') as f:
        eval_data = json.load(f)

    hit_count = 0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    ndcg_count = 0
    recall_sum = 0.0
    recall_count = 0
    total_queries = len(eval_data)

    for item in eval_data:
        query = item['query']
        ground_truth_chunks = [chunk['text_snippet'] for chunk in item.get('relevant_chunks', [])]

        # Retrieve & rerank
        retrieved_docs = pipeline.retriever.as_retriever().invoke(query)
        reranked = pipeline.reranker.rerank(query, retrieved_docs, k_eval)
        top_docs = reranked[0] if isinstance(reranked, (list, tuple)) else reranked
        retrieved_contents = [getattr(doc, "page_content", str(doc)) for doc in top_docs]

        # Per-query NDCG@k
        if len(ground_truth_chunks) > 0:
            ndcg_q = ndcg(retrieved_contents, ground_truth_chunks, k=k_eval)
            ndcg_sum += ndcg_q
            ndcg_count += 1
        else:
            ndcg_q = 0.0

        # Hit Rate + MRR (first relevant rank)
        is_hit = False
        first_hit_rank = None
        for i, doc_content in enumerate(retrieved_contents):
            if any(gt in doc_content for gt in ground_truth_chunks):
                if not is_hit:
                    is_hit = True
                    first_hit_rank = i + 1
                    hit_count += 1
                    mrr_sum += 1.0 / first_hit_rank

        # Recall@k
        if len(ground_truth_chunks) > 0:
            # Count distinct docs that are matched at least once in top-k
            matched_docs = set()
            for doc_content in retrieved_contents:
                for gt in ground_truth_chunks:
                    if gt in doc_content:
                        matched_docs.add(gt)
            recall_q = len(matched_docs) / len(ground_truth_chunks)
            recall_sum += recall_q
            recall_count += 1
        else:
            recall_q = 0.0

        print(
            f"Query: '{query[:40]}...' | "
            f"Hit: {'✅' if is_hit else '❌'} | "
            f"First Hit Rank: {first_hit_rank if is_hit else 'N/A'}"
        )

    avg_hit_rate = (hit_count / total_queries) * 100 if total_queries else 0.0
    avg_mrr = mrr_sum / total_queries if total_queries else 0.0
    mean_ndcg = ndcg_sum / ndcg_count if ndcg_count else 0.0
    mean_recall = recall_sum / recall_count if recall_count else 0.0

    print("\n--- Retrieval Evaluation Summary ---")
    print(f"Total Queries: {total_queries}")
    print(f"Hit Rate: {avg_hit_rate:.2f}%")
    print(f"Mean Reciprocal Rank (MRR): {avg_mrr:.4f}")
    print(f"Mean NDCG@{k_eval}: {mean_ndcg:.4f}")
    print(f"Mean Recall@{k_eval}: {mean_recall:.4f}")
    print("--------------------------------------")

    return {
        "hit_rate": avg_hit_rate,
        "mrr": avg_mrr,
        f"ndcg@{k_eval}": mean_ndcg,
        f"recall@{k_eval}": mean_recall
    }

def evaluate_generation(pipeline: RAGPipeline, eval_dataset_path: str) -> float:
    """Evaluate the generation quality of the model.
    
    Args:
        pipeline (RAGPipeline): The RAG pipeline containing the retriever and reranker
        eval_dataset_path (str): Path to the evaluation dataset

    Returns:
        float: Average semantic similarity score
    """
    

    print("\n" + "="*50)
    print("Evaluating End-to-End Generation Quality...")
    print("="*50)

    with open(eval_dataset_path, 'r') as f:
        eval_data = json.load(f)

    embedding_model = SentenceTransformer(hp.EMBEDDING_NAME)

    total_queries = len(eval_data)
    total_similarity_score = 0.0

    for i, item in enumerate(eval_data):
        query = item['query']
        ground_truth_answer = item['ground_truth_answer']

        print(f"\n({i+1}/{total_queries}) Processing Query: '{query}'")

        generated_answer, _ = pipeline.generate_response(query)

        # Clean up the generated answer if it includes the prefix
        if "Final Answer:" in generated_answer:
            generated_answer = generated_answer.split("Final Answer:")[1].strip()

        # Calculate semantic similarity
        gt_embedding = embedding_model.encode(ground_truth_answer, convert_to_tensor=True)
        gen_embedding = embedding_model.encode(generated_answer, convert_to_tensor=True)
        similarity = cosine_similarity(
            gt_embedding.cpu().reshape(1, -1),
            gen_embedding.cpu().reshape(1, -1)
        )[0][0]

        total_similarity_score += similarity

        print(f"  Ground Truth: '{ground_truth_answer[:60]}...'")
        print(f"  Generated:    '{generated_answer[:60]}...'")
        print(f"  Semantic Similarity: {similarity:.4f}")

    average_similarity = total_similarity_score / total_queries

    print("\n--- Generation Evaluation Summary ---")
    print(f"Total Queries: {total_queries}")
    print(f"📊 Average Semantic Similarity: {average_similarity:.4f}")
    print("--------------------------------------")

    return average_similarity