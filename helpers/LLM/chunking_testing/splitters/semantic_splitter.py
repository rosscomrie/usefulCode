from typing import List, Dict, Any
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .base_splitter import BaseSplitter
from configs.config import *

class SemanticTextSplitter(BaseSplitter):
    """Implementation of semantic text splitting strategies."""
    
    def __init__(self, embeddings, overlap_size=2, similarity_threshold=SIMILARITY_THRESHOLD):
        self.embeddings = embeddings
        self.overlap_size = overlap_size
        self.similarity_threshold = similarity_threshold
        
        # Initialize NLTK
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def split_text(self, text: str, method: str = 'paragraphs') -> List[str]:
        """Split text using specified semantic method."""
        method_map = {
            'sentences': self.split_by_sentences,
            'paragraphs': self.split_by_paragraphs,
            'structure': self.split_by_structure,
            'similarity': self.split_by_similarity,
            'intro': self.split_with_intro
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown splitting method: {method}")
        
        return method_map[method](text)

    def split_by_sentences(self, text: str, n_sentences: int = SENTENCE_WINDOW_SIZE) -> List[str]:
        """Split text with overlapping sentences."""
        sentences = nltk.sent_tokenize(text)
        chunks = []
        
        for i in range(0, len(sentences) - n_sentences + 1, n_sentences // 2):
            chunk = " ".join(sentences[i:i + n_sentences])
            chunks.append(chunk)
        
        return chunks if chunks else [text]

    def split_by_paragraphs(self, text: str, n_paragraphs: int = PARAGRAPH_WINDOW_SIZE) -> List[str]:
        """Split text with overlapping paragraphs."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        
        for i in range(0, len(paragraphs) - n_paragraphs + 1):
            chunk = "\n\n".join(paragraphs[i:i + n_paragraphs])
            chunks.append(chunk)
        
        return chunks if chunks else [text]

    def split_by_structure(self, text: str, markers: List[str] = None) -> List[str]:
        """Split text based on document structure."""
        if markers is None:
            markers = SECTION_MARKERS
        
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        
        for line in lines:
            if any(marker in line for marker in markers):
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks

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

    def split_with_intro(self, text: str, intro_length: int = INTRO_LENGTH) -> List[str]:
        """Split text and include introductory material with each chunk."""
        intro = text[:intro_length]
        main_chunks = self.split_by_paragraphs(text[intro_length:])
        return [f"{intro}\n\n{chunk}" for chunk in main_chunks]

    @property
    def name(self) -> str:
        return "Semantic Text Splitter"