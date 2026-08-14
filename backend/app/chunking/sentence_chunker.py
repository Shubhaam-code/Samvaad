"""
Sentence-aware chunker with configurable grouping and overlap.

Splits passages into sentence-based chunks with support for multilingual
sentence boundary detection (English and Hindi/Indic scripts).
"""

import re
from typing import List

from app.chunking.base import BaseChunker
from app.chunking.models import Chunk, ChunkingStrategy
from app.dataset.models import CanonicalPassage


class SentenceChunker(BaseChunker):
    """
    Chunker that splits passages into sentence-based chunks.
    
    This strategy detects sentence boundaries using multilingual punctuation
    patterns and groups consecutive sentences into chunks based on configurable
    parameters. Supports overlap between chunks for context preservation.
    
    Configuration:
        - sentences_per_chunk: Number of sentences per chunk (default: 3)
        - sentence_overlap: Number of sentences to overlap between chunks (default: 1)
        
    Sentence Detection:
        - English: . ? !
        - Hindi/Indic: । (Devanagari danda)
        - Handles repeated punctuation: ..., !!, ?!
        - Preserves punctuation and whitespace
        
    Behavior:
        - Passages with fewer sentences than sentences_per_chunk → 1 chunk
        - Longer passages → multiple overlapping chunks
        - chunk_index increments: 0, 1, 2, ...
        - All source metadata preserved
        - Deterministic chunk IDs
        
    Validation:
        - sentences_per_chunk must be positive
        - 0 <= sentence_overlap < sentences_per_chunk
        
    Use cases:
        - Long passages need splitting
        - Sentence-level semantic boundaries preferred
        - Context window constraints
        - Improved retrieval granularity
    """
    
    # Multilingual sentence boundary pattern
    # Matches: . ? ! । (with optional repetition) followed by whitespace or end
    SENTENCE_BOUNDARY_PATTERN = re.compile(
        r'[.?!।]+(?:\s+|$)',
        re.UNICODE
    )
    
    def __init__(
        self,
        sentences_per_chunk: int = 3,
        sentence_overlap: int = 1
    ):
        """
        Initialize the sentence chunker.
        
        Args:
            sentences_per_chunk: Number of sentences to include in each chunk
            sentence_overlap: Number of sentences to overlap between consecutive chunks
            
        Raises:
            ValueError: If configuration parameters are invalid
            
        Example:
            >>> chunker = SentenceChunker(sentences_per_chunk=3, sentence_overlap=1)
            >>> # Chunks: [S1,S2,S3], [S3,S4,S5], [S5,S6,S7]
        """
        if sentences_per_chunk <= 0:
            raise ValueError(
                f"sentences_per_chunk must be positive, got {sentences_per_chunk}"
            )
        
        if sentence_overlap < 0:
            raise ValueError(
                f"sentence_overlap must be non-negative, got {sentence_overlap}"
            )
        
        if sentence_overlap >= sentences_per_chunk:
            raise ValueError(
                f"sentence_overlap ({sentence_overlap}) must be less than "
                f"sentences_per_chunk ({sentences_per_chunk})"
            )
        
        self.sentences_per_chunk = sentences_per_chunk
        self.sentence_overlap = sentence_overlap
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using multilingual boundary detection.
        
        Args:
            text: The text to split
            
        Returns:
            List of sentence strings (including trailing punctuation and whitespace)
            
        Notes:
            - Preserves original punctuation and spacing
            - Handles English (. ? !) and Hindi (।) boundaries
            - Handles repeated punctuation (..., !!)
            - Empty or whitespace-only text returns empty list
        """
        if not text or not text.strip():
            return []
        
        sentences = []
        last_end = 0
        
        for match in self.SENTENCE_BOUNDARY_PATTERN.finditer(text):
            sentence = text[last_end:match.end()].strip()
            if sentence:  # Only add non-empty sentences
                sentences.append(sentence)
            last_end = match.end()
        
        # Capture any remaining text after the last boundary
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                sentences.append(remaining)
        
        return sentences
    
    def _group_sentences(
        self,
        sentences: List[str]
    ) -> List[str]:
        """
        Group sentences into overlapping chunks.
        
        Args:
            sentences: List of sentence strings
            
        Returns:
            List of chunk texts (each containing grouped sentences)
            
        Example:
            With sentences_per_chunk=3, sentence_overlap=1:
            Input: ["S1", "S2", "S3", "S4", "S5"]
            Output: ["S1 S2 S3", "S3 S4 S5"]
        """
        if not sentences:
            return []
        
        # If we have fewer sentences than sentences_per_chunk, return as one chunk
        if len(sentences) <= self.sentences_per_chunk:
            return [" ".join(sentences)]
        
        chunks = []
        stride = self.sentences_per_chunk - self.sentence_overlap
        
        i = 0
        while i < len(sentences):
            # Take sentences_per_chunk sentences starting from i
            chunk_sentences = sentences[i:i + self.sentences_per_chunk]
            
            # Only create chunk if we have sentences
            if chunk_sentences:
                chunks.append(" ".join(chunk_sentences))
            
            # Move forward by stride
            i += stride
            
            # If we've covered all sentences, break
            # (avoid creating duplicate chunks at the end)
            if i >= len(sentences):
                break
        
        return chunks
    
    def chunk(self, passage: CanonicalPassage) -> list[Chunk]:
        """
        Convert a CanonicalPassage into sentence-based Chunks.
        
        Args:
            passage: The canonical passage to chunk
            
        Returns:
            List of Chunks (one or more depending on sentence count)
            
        Notes:
            - Passages with few sentences → single chunk
            - Longer passages → multiple overlapping chunks
            - chunk_index starts at 0 and increments
            - All chunks preserve source metadata
            
        Example:
            >>> chunker = SentenceChunker(sentences_per_chunk=2, sentence_overlap=1)
            >>> passage = CanonicalPassage(translated_passage="S1. S2. S3. S4.", ...)
            >>> chunks = chunker.chunk(passage)
            >>> len(chunks)
            3
            >>> chunks[0].chunk_index
            0
            >>> chunks[1].chunk_index
            1
        """
        text = passage.translated_passage
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        # Group sentences into chunks
        chunk_texts = self._group_sentences(sentences)
        
        # Handle edge case: no sentences detected
        if not chunk_texts:
            # Treat entire text as one chunk
            chunk_texts = [text.strip()] if text.strip() else []
        
        # Create Chunk objects
        chunks = []
        for chunk_index, chunk_text in enumerate(chunk_texts):
            chunk = Chunk.from_passage_segment(
                document_id=passage.document_id,
                chunk_index=chunk_index,
                strategy=ChunkingStrategy.SENTENCE,
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
                start_offset=None,  # Sentence boundaries make offsets complex
                end_offset=None,
                overlap_before=None,
                overlap_after=None,
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_batch(self, passages: list[CanonicalPassage]) -> list[Chunk]:
        """
        Convert a batch of CanonicalPassages into sentence-based Chunks.
        
        Args:
            passages: List of canonical passages to chunk
            
        Returns:
            List of all Chunks from all passages in order
            
        Notes:
            - Input ordering is preserved
            - Input passages are not mutated
            - Empty input returns empty output
            
        Example:
            >>> chunker = SentenceChunker()
            >>> passages = [passage1, passage2]
            >>> chunks = chunker.chunk_batch(passages)
            >>> # Returns chunks from passage1 followed by chunks from passage2
        """
        chunks = []
        for passage in passages:
            chunks.extend(self.chunk(passage))
        return chunks
