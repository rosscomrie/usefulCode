import os
import time
from typing import List, Dict, Any, Callable
import numpy as np
from openai import OpenAI
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample test document
SAMPLE_TEXT = """
Artificial Intelligence (AI) is transforming how we live and work. Machine learning, a subset of AI, 
enables computers to learn from data without explicit programming. Deep learning, a type of machine 
learning, uses neural networks with multiple layers to process complex patterns.

Natural Language Processing (NLP) is another crucial area of AI. It helps computers understand, 
interpret, and generate human language. Applications include machine translation, sentiment analysis, 
and chatbots.

Computer vision enables machines to understand and process visual information from the world. 
This technology is used in facial recognition, autonomous vehicles, and medical image analysis.

Robotics combines AI with mechanical engineering to create machines that can perform physical tasks. 
Modern robots use sensors, AI, and sophisticated control systems to interact with their environment.

Ethics in AI is an important consideration. Issues include bias in algorithms, privacy concerns, 
and the impact of automation on employment. Responsible AI development requires addressing these 
challenges while maximizing benefits to society.
"""

class SemanticTextSplitter:
    def __init__(self, embeddings, overlap_size=2, similarity_threshold=0.7):
        self.embeddings = embeddings
        self.overlap_size = overlap_size
        self.similarity_threshold = similarity_threshold
        
        # Initialize NLTK for sentence tokenization
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

    def split_text(self, text: str) -> List[str]:
        """Default split method - can be overridden by specific configurations"""
        return self.split_by_paragraphs(text)

    def split_by_sentences(self, text: str, n_sentences: int = 6) -> List[str]:
        """Split text with overlapping sentences."""
        import nltk
        sentences = nltk.sent_tokenize(text)
        chunks = []
        
        for i in range(0, len(sentences) - n_sentences + 1, n_sentences // 2):
            chunk = " ".join(sentences[i:i + n_sentences])
            chunks.append(chunk)
        
        return chunks if chunks else [text]

    def split_by_paragraphs(self, text: str, n_paragraphs: int = 3) -> List[str]:
        """Split text with overlapping paragraphs."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        
        for i in range(0, len(paragraphs) - n_paragraphs + 1):
            chunk = "\n\n".join(paragraphs[i:i + n_paragraphs])
            chunks.append(chunk)
        
        return chunks if chunks else [text]

    def split_by_structure(self, text: str, section_markers: List[str] = None) -> List[str]:
        """Split text based on document structure."""
        if section_markers is None:
            section_markers = ['Chapter', 'Section', '#']
        
        current_chunk = []
        chunks = []
        
        for line in text.split('\n'):
            if any(marker in line for marker in section_markers):
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks if chunks else [text]

    def split_by_similarity(self, text: str) -> List[str]:
        """Split text based on semantic similarity between paragraphs."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) <= 1:
            return paragraphs

        embeddings_list = self.embeddings.embed_documents(paragraphs)
        similarity_matrix = cosine_similarity(embeddings_list)
        
        chunks = []
        current_chunk = [paragraphs[0]]
        
        for i in range(1, len(paragraphs)):
            avg_similarity = np.mean([similarity_matrix[i][j] for j in range(i)])
            if avg_similarity < self.similarity_threshold:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
            current_chunk.append(paragraphs[i])
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks

    def split_with_intro(self, text: str, intro_length: int = 200) -> List[str]:
        """Split text and include introductory material with each chunk."""
        intro = text[:intro_length]
        main_chunks = self.split_by_paragraphs(text[intro_length:])
        
        return [f"{intro}\n\n{chunk}" for chunk in main_chunks]

def create_test_queries() -> List[str]:
    """Create a list of test queries with varying complexity."""
    return [
        "What is machine learning?",
        "Explain the relationship between AI and robotics.",
        "What are the ethical concerns in AI?",
        "How does computer vision work?",
        "What is the role of NLP in AI?"
    ]

def create_chunking_configs(embeddings: OpenAIEmbeddings) -> List[Dict[str, Any]]:
    """Create both traditional and semantic chunking configurations to test."""
    semantic_splitter = SemanticTextSplitter(embeddings)
    
    # Traditional chunking configurations
    traditional_configs = [
        {
            "name": "Small Fixed Chunks",
            "type": "traditional",
            "splitter": CharacterTextSplitter(
                separator="\n",
                chunk_size=200,
                chunk_overlap=20
            )
        },
        {
            "name": "Medium Fixed Chunks",
            "type": "traditional",
            "splitter": CharacterTextSplitter(
                separator="\n",
                chunk_size=400,
                chunk_overlap=40
            )
        },
        {
            "name": "Recursive Paragraph",
            "type": "traditional",
            "splitter": RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " "],
                chunk_size=300,
                chunk_overlap=30
            )
        },
        {
            "name": "Token Based",
            "type": "traditional",
            "splitter": TokenTextSplitter(
                chunk_size=200,
                chunk_overlap=20
            )
        }
    ]
    
    # Semantic chunking configurations
    semantic_configs = [
        {
            "name": "Overlapping Sentences",
            "type": "semantic",
            "splitter": lambda text: semantic_splitter.split_by_sentences(text, n_sentences=6)
        },
        {
            "name": "Overlapping Paragraphs",
            "type": "semantic",
            "splitter": lambda text: semantic_splitter.split_by_paragraphs(text, n_paragraphs=3)
        },
        {
            "name": "Document Structure",
            "type": "semantic",
            "splitter": lambda text: semantic_splitter.split_by_structure(text)
        },
        {
            "name": "Subject Similarity",
            "type": "semantic",
            "splitter": lambda text: semantic_splitter.split_by_similarity(text)
        },
        {
            "name": "Copy Introductory",
            "type": "semantic",
            "splitter": lambda text: semantic_splitter.split_with_intro(text)
        }
    ]
    
    return traditional_configs + semantic_configs

def evaluate_retrieval_quality(
    query: str,
    retrieved_chunks: List[str],
    all_chunks: List[str],
    embeddings: OpenAIEmbeddings
) -> Dict[str, float]:
    """Evaluate retrieval quality using multiple metrics."""
    query_embedding = embeddings.embed_query(query)
    retrieved_embeddings = embeddings.embed_documents(retrieved_chunks)
    all_embeddings = embeddings.embed_documents(all_chunks)
    
    retrieved_similarities = cosine_similarity([query_embedding], retrieved_embeddings)[0]
    all_similarities = cosine_similarity([query_embedding], all_embeddings)[0]
    
    top_k_accuracy = len(set(retrieved_chunks) & set([all_chunks[i] for i in np.argsort(all_similarities)[-len(retrieved_chunks):]]))
    mean_reciprocal_rank = 1.0 / (np.argsort(all_similarities)[::-1].tolist().index(np.argmax(retrieved_similarities)) + 1)
    precision_at_k = np.mean([1 if sim > 0.7 else 0 for sim in retrieved_similarities])
    
    return {
        "avg_similarity": float(np.mean(retrieved_similarities)),
        "max_similarity": float(np.max(retrieved_similarities)),
        "top_k_accuracy": float(top_k_accuracy / len(retrieved_chunks)),
        "mrr": float(mean_reciprocal_rank),
        "precision_at_k": float(precision_at_k)
    }

def main():
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Get test queries
    queries = create_test_queries()
    
    # Get all chunking configurations
    configs = create_chunking_configs(embeddings)
    
    # Store results
    results = []
    
    # Run tests for each configuration
    for config in configs:
        print(f"\nTesting {config['name']} ({config['type']})...")
        
        try:
            # Split text using the appropriate method
            if config['type'] == 'traditional':
                chunks = config['splitter'].split_text(SAMPLE_TEXT)
            else:
                chunks = config['splitter'](SAMPLE_TEXT)
            
            # Store chunk-level metrics
            chunk_metrics = {
                "num_chunks": len(chunks),
                "avg_chunk_length": np.mean([len(chunk) for chunk in chunks]),
                "std_chunk_length": np.std([len(chunk) for chunk in chunks])
            }
            
            # Run retrieval tests
            retrieval_metrics = []
            for query in queries:
                # Create vector store and retrieve chunks
                vectorstore = FAISS.from_texts(chunks, embeddings)
                retrieved_chunks = [doc.page_content for doc in vectorstore.similarity_search(query, k=2)]
                
                # Evaluate retrieval quality
                quality_metrics = evaluate_retrieval_quality(query, retrieved_chunks, chunks, embeddings)
                retrieval_metrics.append(quality_metrics)
            
            # Average metrics across queries
            avg_metrics = {
                k: np.mean([m[k] for m in retrieval_metrics])
                for k in retrieval_metrics[0].keys()
            }
            
            # Combine results
            results.append({
                "Configuration": config["name"],
                "Type": config["type"],
                "Number of Chunks": chunk_metrics["num_chunks"],
                "Avg Chunk Length": round(chunk_metrics["avg_chunk_length"], 2),
                "Chunk Length Std": round(chunk_metrics["std_chunk_length"], 2),
                "Avg Similarity": round(avg_metrics["avg_similarity"], 3),
                "Max Similarity": round(avg_metrics["max_similarity"], 3),
                "Top-K Accuracy": round(avg_metrics["top_k_accuracy"], 3),
                "MRR": round(avg_metrics["mrr"], 3),
                "Precision@K": round(avg_metrics["precision_at_k"], 3)
            })
            
        except Exception as e:
            print(f"Error processing {config['name']}: {str(e)}")
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Group results by type and display
    print("\nTraditional Chunking Results:")
    print(df[df['Type'] == 'traditional'].drop('Type', axis=1).to_string(index=False))
    print("\nSemantic Chunking Results:")
    print(df[df['Type'] == 'semantic'].drop('Type', axis=1).to_string(index=False))
    
    # Save results to CSV
    df.to_csv("chunking_comparison_results.csv", index=False)
    print("\nResults saved to chunking_comparison_results.csv")

if __name__ == "__main__":
    main()