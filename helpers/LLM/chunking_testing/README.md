# Text Chunking Comparison Tool

A comprehensive tool for comparing different text chunking strategies, including both traditional and semantic approaches. This tool helps evaluate the effectiveness of various chunking methods for document retrieval tasks.

## Features

- Multiple chunking strategies:
  - Traditional methods (fixed-size, recursive, token-based)
  - Semantic methods (sentence-based, paragraph-based, structure-based, similarity-based)
- Detailed evaluation metrics
- Visualization of results
- Configurable parameters
- Support for custom text inputs

## Project Structure

```
chunking_comparison/
├── configs/
│   └── config.py           # Configuration settings
├── data/
│   └── sample_text.py      # Sample texts and queries
├── splitters/
│   ├── base_splitter.py    # Base splitter class
│   ├── semantic_splitter.py # Semantic chunking implementations
│   └── traditional_splitter.py # Traditional chunking implementations
├── evaluation/
│   ├── metrics.py          # Evaluation metrics
│   └── evaluator.py        # Evaluation logic
├── utils/
│   └── visualisation.py    # Visualisation utilities
├── output/
│   └── plots/             # Generated plots directory
├── main.py                # Main script
└── requirements.txt       # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chunking-comparison.git
cd chunking-comparison
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key'
```

## Usage

1. Basic usage:
```bash
python main.py
```

2. Custom text input:
```python
from data.sample_text import get_sample_text, get_test_queries

# Use complex sample text
text = get_sample_text('complex')

# Get specific category queries
queries = get_test_queries('systems')
```

3. Customize configurations:
Edit `configs/config.py` to modify:
- Chunk sizes
- Overlap settings
- Similarity thresholds
- Output directories

## Output

The tool generates:
1. CSV file with detailed metrics
2. Visualization plots:
   - Chunk size distribution
   - Similarity comparisons
   - Performance radar charts
   - Metric comparisons

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies listed in requirements.txt

## Dependencies

```
openai
langchain
faiss-cpu
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
python-dotenv
tqdm
```