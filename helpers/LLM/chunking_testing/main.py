import os
from typing import Any
from langchain.embeddings import OpenAIEmbeddings
from splitters.semantic_splitter import SemanticTextSplitter
from splitters.traditional_splitter import get_traditional_splitters
from evaluation.evaluator import BatchEvaluator
from utils.visualisation import create_comparison_plots
from data.sample_text import SAMPLE_TEXT, TEST_QUERIES
from configs.config import (
    OPENAI_API_KEY,
    OUTPUT_DIR,
    PLOTS_DIR,
    RESULTS_FILE
)

def get_semantic_configs(embeddings) -> list[dict[str, Any]]:
    """Get semantic splitter configurations."""
    semantic_splitter = SemanticTextSplitter(embeddings)
    
    return [
        {
            "name": "Overlapping Sentences",
            "type": "semantic",
            "splitter": lambda text: semantic_splitter.split_text(text, 'sentences')
        },
        {
            "name": "Overlapping Paragraphs",
            "type": "semantic",
            "splitter": lambda text: semantic_splitter.split_text(text, 'paragraphs')
        },
        {
            "name": "Document Structure",
            "type": "semantic",
            "splitter": lambda text: semantic_splitter.split_text(text, 'structure')
        },
        {
            "name": "Subject Similarity",
            "type": "semantic",
            "splitter": lambda text: semantic_splitter.split_text(text, 'similarity')
        },
        {
            "name": "Copy Introductory",
            "type": "semantic",
            "splitter": lambda text: semantic_splitter.split_text(text, 'intro')
        }
    ]

def main():
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Get all configurations
    traditional_configs = get_traditional_splitters()
    semantic_configs = get_semantic_configs(embeddings)
    all_configs = traditional_configs + semantic_configs
    
    # Initialize batch evaluator
    evaluator = BatchEvaluator(embeddings)
    
    # Run evaluation
    print("\nRunning chunking evaluation...")
    results_df = evaluator.run_comparison(
        all_configs,
        SAMPLE_TEXT,
        TEST_QUERIES,
        output_file=RESULTS_FILE
    )
    
    # Create visualizations
    print("\nGenerating visualization plots...")
    create_comparison_plots(results_df, PLOTS_DIR)
    
    # Print summary
    print("\nEvaluation complete!")
    print(f"Results saved to: {RESULTS_FILE}")
    print(f"Plots saved to: {PLOTS_DIR}")
    
    # Display summary statistics
    print("\nSummary by chunking type:")
    summary = results_df.groupby('Type').agg({
        'Avg Similarity': ['mean', 'std'],
        'Precision@K': ['mean', 'std'],
        'Coverage': ['mean', 'std']
    }).round(3)
    print(summary)

if __name__ == "__main__":
    main()