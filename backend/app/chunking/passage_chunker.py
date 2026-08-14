"""
Passage-preserving chunker.

Converts each CanonicalPassage into exactly one Chunk without splitting or modifying
the text content. This is the simplest chunking strategy, useful when passages are
already appropriately sized or when semantic coherence is paramount.
"""

from app.chunking.base import BaseChunker
from app.chunking.models import Chunk, ChunkingStrategy
from app.dataset.models import CanonicalPassage


class PassageChunker(BaseChunker):
    """
    Chunker that preserves entire passages as single chunks.
    
    This strategy creates exactly one chunk per passage, preserving the complete
    translated_passage text without any modifications. All source metadata is
    preserved for full traceability.
    
    Behavior:
        - Each CanonicalPassage → exactly one Chunk
        - chunk_index is always 0
        - chunk_text = passage.translated_passage (unmodified)
        - character_count = len(chunk_text)
        - All source metadata preserved
        - Deterministic chunk IDs
        
    Use cases:
        - Passages are already appropriately sized
        - Maximum semantic coherence required
        - Baseline chunking strategy
        - Preprocessing step before more granular chunking
    """
    
    def chunk(self, passage: CanonicalPassage) -> list[Chunk]:
        """
        Convert a single CanonicalPassage into a single Chunk.
        
        Args:
            passage: The canonical passage to chunk
            
        Returns:
            A list containing exactly one Chunk with the complete passage text
            
        Example:
            >>> chunker = PassageChunker()
            >>> passage = CanonicalPassage(...)
            >>> chunks = chunker.chunk(passage)
            >>> len(chunks)
            1
            >>> chunks[0].chunk_text == passage.translated_passage
            True
        """
        chunk_text = passage.translated_passage
        
        chunk = Chunk.from_passage_segment(
            document_id=passage.document_id,
            chunk_index=0,  # Always 0 for passage-level chunking
            strategy=ChunkingStrategy.PASSAGE,
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
            token_count=None,  # No tokenizer available
            start_offset=0,
            end_offset=len(chunk_text),
            overlap_before=0,
            overlap_after=0,
        )
        
        return [chunk]
    
    def chunk_batch(self, passages: list[CanonicalPassage]) -> list[Chunk]:
        """
        Convert a batch of CanonicalPassages into Chunks.
        
        Args:
            passages: List of canonical passages to chunk
            
        Returns:
            List of Chunks (one per passage) in the same order as input
            
        Notes:
            - Input ordering is preserved
            - Input passages are not mutated
            - Empty input returns empty output
            
        Example:
            >>> chunker = PassageChunker()
            >>> passages = [passage1, passage2, passage3]
            >>> chunks = chunker.chunk_batch(passages)
            >>> len(chunks)
            3
        """
        chunks = []
        for passage in passages:
            chunks.extend(self.chunk(passage))
        return chunks
