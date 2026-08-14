"""
Token-aware chunker with sliding window and overlap.

Splits passages into fixed-size token windows with configurable overlap,
suitable for transformer-based models with maximum context lengths.
"""

from typing import Sequence

from app.chunking.base import BaseChunker
from app.chunking.models import Chunk, ChunkingStrategy
from app.chunking.tokenizer import TokenizerProtocol
from app.dataset.models import CanonicalPassage


class TokenChunker(BaseChunker):
    """
    Chunker that splits passages based on token counts with sliding windows.
    
    This strategy tokenizes the passage and creates fixed-size windows of tokens
    with configurable overlap. Each window is decoded back into text to create
    a chunk. This approach is ideal for:
    
    - Transformer models with maximum context length constraints
    - Consistent chunk sizes across different languages
    - Precise control over input token counts
    - Long passages that exceed model context windows
    
    Configuration:
        - chunk_size: Maximum number of tokens per chunk (default: 256)
        - token_overlap: Number of tokens to overlap between chunks (default: 32)
        
    Algorithm:
        1. Tokenize the passage text
        2. Create sliding windows of chunk_size tokens
        3. Advance by (chunk_size - token_overlap) tokens per window
        4. Decode each window back to text
        5. Generate sequential chunk_index values
        
    Example with chunk_size=256, token_overlap=32:
        - Chunk 0: tokens [0:256]
        - Chunk 1: tokens [224:480]  (224 = 256 - 32)
        - Chunk 2: tokens [448:704]  (448 = 224 + 224)
        
    Validation:
        - chunk_size must be positive
        - token_overlap must be non-negative
        - token_overlap must be less than chunk_size
    """
    
    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        chunk_size: int = 256,
        token_overlap: int = 32
    ):
        """
        Initialize the token chunker.
        
        Args:
            tokenizer: A tokenizer conforming to TokenizerProtocol
            chunk_size: Maximum number of tokens per chunk
            token_overlap: Number of tokens to overlap between consecutive chunks
            
        Raises:
            ValueError: If configuration parameters are invalid
            
        Example:
            >>> from app.chunking.tokenizer import create_default_tokenizer
            >>> tokenizer = create_default_tokenizer()
            >>> chunker = TokenChunker(tokenizer, chunk_size=512, token_overlap=64)
        """
        if tokenizer is None:
            raise ValueError(
                "TokenChunker requires a valid tokenizer instance conforming to TokenizerProtocol"
            )
            
        if chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be positive, got {chunk_size}"
            )
        
        if token_overlap < 0:
            raise ValueError(
                f"token_overlap must be non-negative, got {token_overlap}"
            )
        
        if token_overlap >= chunk_size:
            raise ValueError(
                f"token_overlap ({token_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.token_overlap = token_overlap
    
    def _chunk_tokens(
        self,
        token_ids: Sequence[int]
    ) -> list[Sequence[int]]:
        """
        Split token sequence into overlapping windows.
        
        Args:
            token_ids: Complete sequence of token IDs from the passage
            
        Returns:
            List of token ID sequences, one per chunk
            
        Notes:
            - Handles passages shorter than chunk_size
            - Includes final partial chunk if present
            - Never creates empty chunks
            - Always advances at least 1 token to prevent infinite loops
        """
        if not token_ids:
            return []
        
        # If passage is shorter than chunk_size, return as single chunk
        if len(token_ids) <= self.chunk_size:
            return [token_ids]
        
        chunks = []
        stride = max(1, self.chunk_size - self.token_overlap)  # Safety: always advance
        start = 0
        
        while start < len(token_ids):
            # Take up to chunk_size tokens from current position
            end = min(start + self.chunk_size, len(token_ids))
            chunk_tokens = token_ids[start:end]
            
            # Only add non-empty chunks
            if chunk_tokens:
                chunks.append(chunk_tokens)
            
            # If we've reached the end, break
            if end >= len(token_ids):
                break
            
            # Advance by stride
            start += stride
            
            # Safety check: ensure we're making progress
            if stride == 0:
                # This should never happen due to validation, but be defensive
                start += 1
        
        return chunks
    
    def chunk(self, passage: CanonicalPassage) -> list[Chunk]:
        """
        Convert a CanonicalPassage into token-based Chunks.
        
        Args:
            passage: The canonical passage to chunk
            
        Returns:
            List of Chunks (one or more depending on token count)
            
        Notes:
            - Empty or whitespace-only text returns empty list
            - Short passages (< chunk_size tokens) return single chunk
            - Long passages return multiple overlapping chunks
            - chunk_index starts at 0 and increments
            - All chunks preserve source metadata
            
        Example:
            >>> chunker = TokenChunker(tokenizer, chunk_size=100, token_overlap=20)
            >>> passage = CanonicalPassage(...)
            >>> chunks = chunker.chunk(passage)
            >>> for chunk in chunks:
            ...     print(f"Chunk {chunk.chunk_index}: {chunk.token_count} tokens")
        """
        text = passage.translated_passage
        
        # Handle empty or whitespace-only text
        if not text or not text.strip():
            return []
        
        # Tokenize the passage
        token_ids = self.tokenizer.encode(text)
        
        # Handle edge case: tokenization returns empty
        if not token_ids:
            return []
        
        # Split into overlapping windows
        token_windows = self._chunk_tokens(token_ids)
        
        # Create Chunk objects
        chunks = []
        for chunk_index, window_tokens in enumerate(token_windows):
            # Decode tokens back to text
            chunk_text = self.tokenizer.decode(window_tokens)
            
            # Skip if decode produces empty text (shouldn't happen, but be safe)
            if not chunk_text or not chunk_text.strip():
                continue
            
            chunk = Chunk.from_passage_segment(
                document_id=passage.document_id,
                chunk_index=chunk_index,
                strategy=ChunkingStrategy.TOKEN,
                chunk_text=chunk_text,
                query_id=passage.query_id,
                passage_index=passage.passage_index,
                target_lang=passage.target_lang,
                source_lang=passage.source_lang,
                query=passage.query,
                eng_query=passage.eng_query,
                query_type=passage.query_type,
                answer=passage.answer,
                eng_answer=passage.eng_answer,
                is_selected=passage.is_selected,
                character_count=len(chunk_text),
                token_count=len(window_tokens),  # Actual token count
                start_offset=None,  # Token boundaries don't map cleanly to characters
                end_offset=None,
                overlap_before=None,  # Could calculate, but complex with tokens
                overlap_after=None,
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_batch(self, passages: list[CanonicalPassage]) -> list[Chunk]:
        """
        Convert a batch of CanonicalPassages into token-based Chunks.
        
        Args:
            passages: List of canonical passages to chunk
            
        Returns:
            List of all Chunks from all passages in order
            
        Notes:
            - Input ordering is preserved
            - Input passages are not mutated
            - Empty input returns empty output
            
        Example:
            >>> chunker = TokenChunker(tokenizer)
            >>> passages = [passage1, passage2]
            >>> chunks = chunker.chunk_batch(passages)
            >>> # Returns chunks from passage1 followed by chunks from passage2
        """
        chunks = []
        for passage in passages:
            chunks.extend(self.chunk(passage))
        return chunks
    
    def __repr__(self) -> str:
        return (
            f"TokenChunker("
            f"tokenizer={self.tokenizer!r}, "
            f"chunk_size={self.chunk_size}, "
            f"token_overlap={self.token_overlap})"
        )
