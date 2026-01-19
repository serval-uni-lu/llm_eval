from pydantic import BaseModel
from typing import List, Optional
from chonkie import SemanticChunker, SDPMChunker


class Chunk(BaseModel):
    text: str
    token_count: int
    embedding: Optional[List[float]] = None


class Chunker:
    """Text chunker with multiple strategies."""

    def __init__(
        self,
        strategy: str = "semantic",
        min_chunk_size: int = 256,
        max_chunk_size: int = 1024,
        embedding_model: str = 'Lajavaness/bilingual-embedding-small',
    ):
        """Initialize chunker with specified strategy and parameters.
        
        Args:
            strategy (str): Chunking strategy - "semantic", "sdpm", or "fixed".
            min_chunk_size (int): Minimum chunk size in tokens.
            max_chunk_size (int): Maximum chunk size in tokens.
            embedding_model (str): Embedding model to use for semantic chunking.
        """
        self.strategy = strategy
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        if strategy in ["semantic", "sdpm"]:
            chunker_kwargs = {
                "embedding_model": embedding_model,
                "chunk_size": max_chunk_size,
                "min_chunk_size": min_chunk_size,
                "min_sentences": 1,
            }

            if strategy == "semantic":
                self.chunker = SemanticChunker(
                    **chunker_kwargs,
                    similarity_window=3
                )
            elif strategy == "sdpm":
                self.chunker = SDPMChunker(
                    **chunker_kwargs,
                    skip_window=2
                )
        elif strategy == "fixed":
            self.chunker = None

    def _fixed_chunk(self, text: str) -> List[Chunk]:
        """Fixed-size chunking that respects paragraph boundaries."""
        # Split on double newlines (paragraphs)
        paragraphs = text.split('\n\n')

        chunks_list = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            # If single paragraph exceeds max_chunk_size, add it as its own chunk
            if para_size > self.max_chunk_size:
                if current_chunk:
                    chunks_list.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                chunks_list.append(para)
                continue

            # If adding this paragraph exceeds max_chunk_size, start new chunk
            if current_size + para_size > self.max_chunk_size and current_size >= self.min_chunk_size:
                chunks_list.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size + 2  # +2 for '\n\n'

        # Add remaining paragraphs
        if current_chunk:
            chunks_list.append('\n\n'.join(current_chunk))

        return [
            Chunk(
                text=chunk_text,
                token_count=int(len(chunk_text) / 4.66),
                embedding=None
            )
            for chunk_text in chunks_list
        ]

    def chunk(self, text: str) -> List[Chunk]:
        """Chunk text using the selected strategy."""
        if self.strategy == "fixed":
            return self._fixed_chunk(text)
        else:
            chunks = self.chunker.chunk(text)
            return [
                Chunk(
                    text=chunk.text,
                    token_count=chunk.token_count,
                    embedding=None
                )
                for chunk in chunks
            ]
    