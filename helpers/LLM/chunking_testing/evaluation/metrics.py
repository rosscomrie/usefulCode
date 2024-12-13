from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from configs.config import SIMILARITY_THRESHOLD_FOR_RELEVANCE

def calculate_similarity_metrics(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray
) -> Dict[str, float]:
    """Calculate similarity-based metrics between query and chunks."""
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    
    return {
        "avg_similarity": float(np.mean(similarities)),
        "max_similarity": float(np.max(similarities)),
        "min_similarity": float(np.min(similarities)),
        "std_similarity": float(np.std(similarities))
    }

def calculate_ranking_metrics(
    similarities: np.ndarray,
    relevant_indices: List[int]
) -> Dict[str, float]:
    """Calculate ranking-based metrics."""
    # Sort indices by similarity
    ranked_indices = np.argsort(similarities)[::-1]
    
    # Calculate MRR
    relevant_ranks = [
        1.0 / (rank + 1)
        for rank, idx in enumerate(ranked_indices)
        if idx in relevant_indices
    ]
    mrr = np.mean(relevant_ranks) if relevant_ranks else 0.0
    
    # Calculate precision@k
    k = min(len(similarities), 3)  # Use top 3 or all if less
    precision_at_k = np.mean([
        1 if idx in relevant_indices else 0
        for idx in ranked_indices[:k]
    ])
    
    return {
        "mrr": float(mrr),
        "precision_at_k": float(precision_at_k)
    }

def evaluate_retrieval_quality(
    query: str,
    retrieved_chunks: List[str],
    all_chunks: List[str],
    embeddings
) -> Dict[str, float]:
    """Evaluate retrieval quality using multiple metrics."""
    # Get embeddings
    query_embedding = embeddings.embed_query(query)
    retrieved_embeddings = embeddings.embed_documents(retrieved_chunks)
    all_embeddings = embeddings.embed_documents(all_chunks)
    
    # Calculate similarities
    retrieved_similarities = cosine_similarity([query_embedding], retrieved_embeddings)[0]
    all_similarities = cosine_similarity([query_embedding], all_embeddings)[0]
    
    # Get relevant chunk indices (using similarity threshold)
    relevant_indices = np.where(all_similarities > SIMILARITY_THRESHOLD_FOR_RELEVANCE)[0]
    
    # Calculate basic similarity metrics
    similarity_metrics = calculate_similarity_metrics(query_embedding, retrieved_embeddings)
    
    # Calculate ranking metrics
    ranking_metrics = calculate_ranking_metrics(all_similarities, relevant_indices)
    
    # Calculate coverage (how many relevant chunks were retrieved)
    retrieved_indices = [
        i for i, chunk in enumerate(all_chunks)
        if chunk in retrieved_chunks
    ]
    coverage = len(set(retrieved_indices) & set(relevant_indices)) / max(len(relevant_indices), 1)
    
    return {
        **similarity_metrics,
        **ranking_metrics,
        "coverage": float(coverage)
    }
