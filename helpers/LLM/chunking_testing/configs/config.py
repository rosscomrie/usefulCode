import os
from typing import List

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Chunking configuration
SMALL_CHUNK_SIZE = 200
MEDIUM_CHUNK_SIZE = 400
DEFAULT_OVERLAP = 20
SIMILARITY_THRESHOLD = 0.7

# Semantic chunking settings
SENTENCE_WINDOW_SIZE = 6
PARAGRAPH_WINDOW_SIZE = 3
INTRO_LENGTH = 200

# Evaluation settings
TOP_K = 2
SIMILARITY_THRESHOLD_FOR_RELEVANCE = 0.7

# Section markers for document structure
SECTION_MARKERS: List[str] = ['Chapter', 'Section', '#']

# Output configuration
OUTPUT_DIR = 'output'
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
RESULTS_FILE = os.path.join(OUTPUT_DIR, 'chunking_results.csv')

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)