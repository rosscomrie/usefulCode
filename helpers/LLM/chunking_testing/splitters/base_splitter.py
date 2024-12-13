from abc import ABC, abstractmethod
from typing import List

class BaseSplitter(ABC):
    """Base class for text splitters."""
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the splitter."""
        pass