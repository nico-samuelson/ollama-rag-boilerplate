import json
import torch
import hyperparameters as hp
from pipeline import generate_response
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_retriever(retriever, eval_dataset_path):
    """
    Evaluates the performance of the retriever using a ground-truth dataset.

    Args:
        retriever: The RAG retriever instance.
        eval_dataset_path: Path to the JSON evaluation file.
    """
    print("\n" + "="*50)
    print("Evaluating Retriever Performance...")
    print("="*50)

    with open(eval_dataset_path, 'r') as f:
        eval_data = json.load(f)

    hit_rate = 0
    mrr_score = 0.0
    total_queries = len(eval_data)

    for item in eval_data:
        query = item['query']
        ground_truth_chunks = [chunk['text_snippet'] for chunk in item['relevant_chunks']]

        # Get retrieved documents
        retrieved_docs = retriever.invoke(query)
        retrieved_contents = [doc.page_content for doc in retrieved_docs]

        # Check for a "hit"
        is_hit = False
        first_hit_rank = 0

        for i, doc_content in enumerate(retrieved_contents):
            # Check if any ground truth snippet is in the retrieved document
            if any(gt_chunk in doc_content for gt_chunk in ground_truth_chunks):
                if not is_hit:
                    hit_rate += 1
                    is_hit = True
                    first_hit_rank = i + 1
                    mrr_score += 1.0 / first_hit_rank

        print(f"Query: '{query[:40]}...' | Hit: {'✅' if is_hit else '❌'} | Rank of First Hit: {first_hit_rank if is_hit else 'N/A'}")

    avg_hit_rate = (hit_rate / total_queries) * 100
    avg_mrr = mrr_score / total_queries

    print("\n--- Retrieval Evaluation Summary ---")
    print(f"Total Queries: {total_queries}")
    print(f"Hit Rate: {avg_hit_rate:.2f}%")
    print(f"Mean Reciprocal Rank (MRR): {avg_mrr:.4f}")
    print("--------------------------------------")

    return avg_hit_rate, avg_mrr

def evaluate_generation(retriever, llm, eval_dataset_path):
    print("\n" + "="*50)
    print("Evaluating End-to-End Generation Quality...")
    print("="*50)

    with open(eval_dataset_path, 'r') as f:
        eval_data = json.load(f)

    embedding_model = SentenceTransformer(hp.EMBEDDING_NAME, device="cuda" if torch.cuda.is_available() else "cpu")

    total_queries = len(eval_data)
    total_similarity_score = 0.0

    for i, item in enumerate(eval_data):
        query = item['query']
        ground_truth_answer = item['ground_truth_answer']

        print(f"\n({i+1}/{total_queries}) Processing Query: '{query}'")

        generated_answer, _ = generate_response(query, retriever, llm)

        # Clean up the generated answer if it includes the prefi    x
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