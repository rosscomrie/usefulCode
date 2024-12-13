from typing import List, Dict, Any
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from .base_splitter import BaseSplitter
from configs.config import *

class TraditionalSplitterWrapper(BaseSplitter):
    """Wrapper for LangChain text splitters."""
    
    def __init__(self, splitter, name: str):
        self.splitter = splitter
        self._name = name
    
    def split_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)
    
    @property
    def name(self) -> str:
        return self._name

def get_traditional_splitters() -> List[Dict[str, Any]]:
    """Get traditional text splitter configurations."""
    return [
        {
            "name": "Small Fixed Chunks",
            "type": "traditional",
            "splitter": TraditionalSplitterWrapper(
                CharacterTextSplitter(
                    separator="\n",
                    chunk_size=SMALL_CHUNK_SIZE,
                    chunk_overlap=DEFAULT_OVERLAP
                ),
                "Small Fixed Chunks"
            )
        },
        {
            "name": "Medium Fixed Chunks",
            "type": "traditional",
            "splitter": TraditionalSplitterWrapper(
                CharacterTextSplitter(
                    separator="\n",
                    chunk_size=MEDIUM_CHUNK_SIZE,
                    chunk_overlap=DEFAULT_OVERLAP
                ),
                "Medium Fixed Chunks"
            )
        },
        {
            "name": "Recursive Paragraph",
            "type": "traditional",
            "splitter": TraditionalSplitterWrapper(
                RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ". ", " "],
                    chunk_size=MEDIUM_CHUNK_SIZE,
                    chunk_overlap=DEFAULT_OVERLAP
                ),
                "Recursive Paragraph"
            )
        },
        {
            "name": "Token Based",
            "type": "traditional",
            "splitter": TraditionalSplitterWrapper(
                TokenTextSplitter(
                    chunk_size=SMALL_CHUNK_SIZE,
                    chunk_overlap=DEFAULT_OVERLAP
                ),
                "Token Based"
            )
        }
    ]