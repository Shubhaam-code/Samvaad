"""
Adaptive chunker with rule-based strategy selection.

Automatically selects the most appropriate chunking strategy based on
passage characteristics using deterministic, measurable rules.
"""

from app.chunking.base import BaseChunker
from app.chunking.models import Chunk, ChunkingStrategy
from app.chunking.passage_chunker import PassageChunker
from app.chunking.sentence_chunker import SentenceChunker
from app.chunking.token_chunker import TokenChunker
from app.chunking.tokenizer import TokenizerProtocol
from app.dataset.models import CanonicalPassage


class AdaptiveChunker(BaseChunker):
    """
    Adaptive chunker that selects strategies based on passage characteristics.
    
    Uses deterministic, rule-based decisions to choose the most appropriate
    chunking strategy for each passage. Decision factors include:
    
    - Passage length (character count)
    - Estimated token count
    - Sentence count
    - Sentence length distribution
    
    Decision Rules:
    ---------------
    
    1. **Short passages** (≤ short_passage_max_chars):
       → PassageChunker (preserve semantic coherence)
       
    2. **Medium passages with reasonable sentence structure**:
       → SentenceChunker (sentence-aware boundaries)
       
    3. **Long passages or very long sentences**:
       → TokenChunker (precise token control)
    
    Configuration:
        - short_passage_max_chars: Character threshold for "short" (default: 500)
        - medium_passage_max_chars: Character threshold for "medium" (default: 2000)
        - long_sentence_threshold: Character count for "long sentence" (default: 500)
        - token_chunk_size: Token window size (default: 256)
        - token_overlap: Token overlap (default: 32)
        - sentence_chunk_size: Sentences per chunk (default: 3)
        - sentence_overlap: Sentence overlap (default: 1)
        
    Note: All decisions are deterministic and based solely on measurable
    text properties. No LLM, embeddings, or semantic analysis is used.
    """
    
    def __init__(
        self,
        tokenizer: TokenizerProtocol | None = None,
        short_passage_max_chars: int = 500,
        medium_passage_max_chars: int = 2000,
        long_sentence_threshold: int = 500,
        token_chunk_size: int = 256,
        token_overlap: int = 32,
        sentence_chunk_size: int = 3,
        sentence_overlap: int = 1,
    ):
        """
        Initialize the adaptive chunker.
        
        Args:
            tokenizer: Optional tokenizer for token-based chunking
            short_passage_max_chars: Character threshold for short passages
            medium_passage_max_chars: Character threshold for medium passages
            long_sentence_threshold: Character threshold for long sentences
            token_chunk_size: Tokens per chunk for token strategy
            token_overlap: Token overlap for token strategy
            sentence_chunk_size: Sentences per chunk for sentence strategy
            sentence_overlap: Sentence overlap for sentence strategy
            
        Raises:
            ValueError: If configuration parameters are invalid
            
        Example:
            >>> from app.chunking.tokenizer import create_default_tokenizer
            >>> tokenizer = create_default_tokenizer()
            >>> chunker = AdaptiveChunker(
            ...     tokenizer=tokenizer,
            ...     short_passage_max_chars=300,
            ...     medium_passage_max_chars=1500
            ... )
        """
        # Validate configuration
        if short_passage_max_chars <= 0:
            raise ValueError(
                f"short_passage_max_chars must be positive, got {short_passage_max_chars}"
            )
        
        if medium_passage_max_chars <= short_passage_max_chars:
            raise ValueError(
                f"medium_passage_max_chars ({medium_passage_max_chars}) must be "
                f"greater than short_passage_max_chars ({short_passage_max_chars})"
            )
        
        if long_sentence_threshold <= 0:
            raise ValueError(
                f"long_sentence_threshold must be positive, got {long_sentence_threshold}"
            )
        
        self.tokenizer = tokenizer
        self.short_passage_max_chars = short_passage_max_chars
        self.medium_passage_max_chars = medium_passage_max_chars
        self.long_sentence_threshold = long_sentence_threshold
        self.token_chunk_size = token_chunk_size
        self.token_overlap = token_overlap
        self.sentence_chunk_size = sentence_chunk_size
        self.sentence_overlap = sentence_overlap
        
        # Initialize delegate chunkers (created lazily to avoid unnecessary construction)
        self._passage_chunker: PassageChunker | None = None
        self._sentence_chunker: SentenceChunker | None = None
        self._token_chunker: TokenChunker | None = None
    
    def _get_passage_chunker(self) -> PassageChunker:
        """Get or create the passage chunker."""
        if self._passage_chunker is None:
            self._passage_chunker = PassageChunker()
        return self._passage_chunker
    
    def _get_sentence_chunker(self) -> SentenceChunker:
        """Get or create the sentence chunker."""
        if self._sentence_chunker is None:
            self._sentence_chunker = SentenceChunker(
                sentences_per_chunk=self.sentence_chunk_size,
                sentence_overlap=self.sentence_overlap
            )
        return self._sentence_chunker
    
    def _get_token_chunker(self) -> TokenChunker:
        """Get or create the token chunker."""
        if self._token_chunker is None:
            if self.tokenizer is None:
                raise ValueError(
                    "TokenChunker requires a tokenizer, but none was provided"
                )
            self._token_chunker = TokenChunker(
                tokenizer=self.tokenizer,
                chunk_size=self.token_chunk_size,
                token_overlap=self.token_overlap
            )
        return self._token_chunker
    
    def _count_sentences_rough(self, text: str) -> int:
        """
        Rough sentence count using simple heuristics.
        
        Counts occurrences of sentence-ending punctuation.
        This is an approximation for decision-making, not precise parsing.
        
        Args:
            text: The text to analyze
            
        Returns:
            Estimated number of sentences
        """
        if not text:
            return 0
        
        # Count sentence-ending punctuation
        count = 0
        for char in text:
            if char in '.?!।':
                count += 1
        
        # If no punctuation found, treat as one sentence
        return max(1, count)
    
    def _has_very_long_sentence(self, text: str) -> bool:
        """
        Check if text contains a very long sentence.
        
        A very long sentence suggests the text may not have clear sentence
        boundaries or may be better suited to token-based chunking.
        
        Args:
            text: The text to analyze
            
        Returns:
            True if any sentence exceeds long_sentence_threshold characters
        """
        if not text:
            return False
        
        # Split on sentence punctuation
        import re
        sentences = re.split(r'[.?!।]+', text)
        
        for sentence in sentences:
            if len(sentence.strip()) > self.long_sentence_threshold:
                return True
        
        return False
    
    def _select_strategy(self, passage: CanonicalPassage) -> str:
        """
        Select the chunking strategy for a passage.
        
        This is the core decision logic using deterministic rules.
        
        Args:
            passage: The passage to analyze
            
        Returns:
            Strategy name: "passage", "sentence", or "token"
        """
        text = passage.translated_passage
        char_count = len(text)
        
        # Rule 1: Short passages → preserve whole
        if char_count <= self.short_passage_max_chars:
            return "passage"
        
        # Rule 2: Very long sentence → token-based
        if self._has_very_long_sentence(text):
            if self.tokenizer is not None:
                return "token"
            # Fallback to sentence if no tokenizer
            return "sentence"
        
        # Rule 3: Medium passages with good sentence structure → sentence-based
        if char_count <= self.medium_passage_max_chars:
            sentence_count = self._count_sentences_rough(text)
            # If we have multiple sentences, use sentence chunking
            if sentence_count >= 2:
                return "sentence"
            # Single sentence in medium range → keep whole
            return "passage"
        
        # Rule 4: Long passages → token-based if tokenizer available
        if self.tokenizer is not None:
            return "token"
        
        # Fallback: sentence-based
        return "sentence"
    
    def chunk(self, passage: CanonicalPassage) -> list[Chunk]:
        """
        Adaptively chunk a CanonicalPassage.
        
        Selects the most appropriate chunking strategy based on passage
        characteristics and delegates to the corresponding chunker.
        
        Args:
            passage: The canonical passage to chunk
            
        Returns:
            List of Chunks with strategy=ADAPTIVE
            
        Notes:
            - Decision is deterministic based on passage properties
            - All chunks have strategy=ChunkingStrategy.ADAPTIVE
            - Empty passages return empty list
            - Metadata is preserved through delegation
            
        Example:
            >>> chunker = AdaptiveChunker(tokenizer)
            >>> short_passage = CanonicalPassage(translated_passage="Short.", ...)
            >>> medium_passage = CanonicalPassage(translated_passage="S1. S2. S3. S4.", ...)
            >>> long_passage = CanonicalPassage(translated_passage="..." * 1000, ...)
            >>> 
            >>> # Short → preserved whole
            >>> len(chunker.chunk(short_passage))
            1
            >>> # Medium → sentence-based
            >>> len(chunker.chunk(medium_passage)) > 1
            True
            >>> # Long → token-based
            >>> len(chunker.chunk(long_passage)) > 1
            True
        """
        # Handle empty text
        text = passage.translated_passage
        if not text or not text.strip():
            return []
        
        # Select strategy
        strategy_name = self._select_strategy(passage)
        
        # Delegate to appropriate chunker
        if strategy_name == "passage":
            chunker = self._get_passage_chunker()
        elif strategy_name == "sentence":
            chunker = self._get_sentence_chunker()
        elif strategy_name == "token":
            chunker = self._get_token_chunker()
        else:
            # Should never happen, but be defensive
            chunker = self._get_passage_chunker()
        
        # Get chunks from delegate
        chunks = chunker.chunk(passage)
        
        # Override strategy to ADAPTIVE
        # (chunks from delegates have their own strategies)
        adaptive_chunks = []
        for chunk in chunks:
            # Create new chunk with ADAPTIVE strategy
            adaptive_chunk = Chunk.from_passage_segment(
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                strategy=ChunkingStrategy.ADAPTIVE,
                chunk_text=chunk.chunk_text,
                query_id=chunk.query_id,
                passage_index=chunk.passage_index,
                target_lang=chunk.target_lang,
                source_lang=chunk.source_lang,
                query=chunk.query,
                eng_query=chunk.eng_query,
                query_type=chunk.query_type,
                answer=chunk.answer,
                eng_answer=chunk.eng_answer,
                is_selected=chunk.is_selected,
                character_count=chunk.character_count,
                token_count=chunk.token_count,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                overlap_before=chunk.overlap_before,
                overlap_after=chunk.overlap_after,
            )
            adaptive_chunks.append(adaptive_chunk)
        
        return adaptive_chunks
    
    def chunk_batch(self, passages: list[CanonicalPassage]) -> list[Chunk]:
        """
        Adaptively chunk a batch of CanonicalPassages.
        
        Args:
            passages: List of canonical passages to chunk
            
        Returns:
            List of all Chunks from all passages in order
            
        Notes:
            - Input ordering is preserved
            - Input passages are not mutated
            - Empty input returns empty output
            - Each passage may use a different strategy
            
        Example:
            >>> chunker = AdaptiveChunker(tokenizer)
            >>> passages = [short_passage, long_passage]
            >>> chunks = chunker.chunk_batch(passages)
            >>> # All chunks have strategy=ADAPTIVE
            >>> all(c.strategy == ChunkingStrategy.ADAPTIVE for c in chunks)
            True
        """
        chunks = []
        for passage in passages:
            chunks.extend(self.chunk(passage))
        return chunks
    
    def __repr__(self) -> str:
        return (
            f"AdaptiveChunker("
            f"short_max={self.short_passage_max_chars}, "
            f"medium_max={self.medium_passage_max_chars}, "
            f"tokenizer={'present' if self.tokenizer else 'None'})"
        )
