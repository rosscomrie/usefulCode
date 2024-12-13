from typing import List, Dict, Any
import pandas as pd
import numpy as np
from .metrics import evaluate_retrieval_quality
from langchain.vectorstores import FAISS

class ChunkingEvaluator:
    def __init__(self, embeddings):
        self.embeddings = embeddings
    
    def evaluate_chunking_method(
        self,
        config: Dict[str, Any],
        text: str,
        queries: List[str]
    ) -> Dict[str, Any]:
        """Evaluate a single chunking method."""
        try:
            # Get chunks using the specified method
            if config['type'] == 'traditional':
                chunks = config['splitter'].split_text(text)
            else:
                chunks = config['splitter'](text)
            
            # Calculate chunk-level metrics
            chunk_metrics = {
                "num_chunks": len(chunks),
                "avg_chunk_length": np.mean([len(chunk) for chunk in chunks]),
                "std_chunk_length": np.std([len(chunk) for chunk in chunks]),
                "min_chunk_length": min(len(chunk) for chunk in chunks),
                "max_chunk_length": max(len(chunk) for chunk in chunks)
            }
            
            # Run retrieval tests for each query
            retrieval_metrics = []
            vectorstore = FAISS.from_texts(chunks, self.embeddings)
            
            for query in queries:
                # Get relevant documents
                retrieved_chunks = [
                    doc.page_content 
                    for doc in vectorstore.similarity_search(query, k=2)
                ]
                
                # Evaluate retrieval quality
                metrics = evaluate_retrieval_quality(
                    query,
                    retrieved_chunks,
                    chunks,
                    self.embeddings
                )
                retrieval_metrics.append(metrics)
            
            # Average metrics across queries
            avg_metrics = {
                k: np.mean([m[k] for m in retrieval_metrics])
                for k in retrieval_metrics[0].keys()
            }
            
            # Combine all metrics
            return {
                "Configuration": config["name"],
                "Type": config["type"],
                **{f"Chunk_{k}": v for k, v in chunk_metrics.items()},
                **{k.title(): v for k, v in avg_metrics.items()}
            }
            
        except Exception as e:
            print(f"Error evaluating {config['name']}: {str(e)}")
            return None

class BatchEvaluator:
    def __init__(self, embeddings):
        self.evaluator = ChunkingEvaluator(embeddings)
    
    def evaluate_all_methods(
        self,
        configs: List[Dict[str, Any]],
        text: str,
        queries: List[str]
    ) -> pd.DataFrame:
        """Evaluate all chunking methods and return results as DataFrame."""
        results = []
        
        for config in configs:
            print(f"Evaluating {config['name']}...")
            result = self.evaluator.evaluate_chunking_method(
                config, text, queries
            )
            if result is not None:
                results.append(result)
        
        return pd.DataFrame(results)

    def run_comparison(
        self,
        configs: List[Dict[str, Any]],
        text: str,
        queries: List[str],
        output_file: str = None
    ) -> pd.DataFrame:
        """Run comparison and optionally save results."""
        results_df = self.evaluate_all_methods(configs, text, queries)
        
        if output_file:
            results_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        
        return results_df