"""
Multi-strategy chunking engine with registry-based dispatch.

Provides a unified interface for chunking passages using different strategies,
with clean strategy selection and extensibility.
"""

from typing import Callable

from app.chunking.adaptive_chunker import AdaptiveChunker
from app.chunking.base import BaseChunker
from app.chunking.models import Chunk, ChunkingStrategy
from app.chunking.passage_chunker import PassageChunker
from app.chunking.sentence_chunker import SentenceChunker
from app.chunking.token_chunker import TokenChunker
from app.chunking.tokenizer import TokenizerProtocol
from app.dataset.models import CanonicalPassage


class ChunkingEngine:
    """
    Multi-strategy chunking engine with registry-based dispatch.
    
    Provides a unified interface for chunking passages using different strategies.
    The engine maintains a registry of available chunkers and routes requests to
    the appropriate implementation based on the requested strategy.
    
    Supported strategies:
        - PASSAGE: Preserve entire passages
        - SENTENCE: Sentence-aware chunking with overlap
        - TOKEN: Token-based windowing with overlap
        - ADAPTIVE: Rule-based strategy selection
        
    Features:
        - Clean strategy selection via enum
        - Extensible via chunker registration
        - Type-safe through BaseChunker interface
        - No strategy fallback (explicit error on unsupported)
        - Stateless operation (no shared mutable state)
        
    Example:
        >>> from app.chunking.tokenizer import create_default_tokenizer
        >>> tokenizer = create_default_tokenizer()
        >>> engine = ChunkingEngine(tokenizer=tokenizer)
        >>> 
        >>> # Chunk with specific strategy
        >>> chunks = engine.chunk(passage, strategy=ChunkingStrategy.SENTENCE)
        >>> 
        >>> # Batch processing
        >>> all_chunks = engine.chunk_batch(passages, strategy=ChunkingStrategy.TOKEN)
        >>> 
        >>> # Adaptive strategy selection
        >>> adaptive_chunks = engine.chunk(passage, strategy=ChunkingStrategy.ADAPTIVE)
    """
    
    def __init__(
        self,
        tokenizer: TokenizerProtocol | None = None,
        sentence_chunk_size: int = 3,
        sentence_overlap: int = 1,
        token_chunk_size: int = 256,
        token_overlap: int = 32,
        adaptive_short_max: int = 500,
        adaptive_medium_max: int = 2000,
    ):
        """
        Initialize the chunking engine.
        
        Args:
            tokenizer: Optional tokenizer for token-based and adaptive strategies
            sentence_chunk_size: Sentences per chunk for sentence strategy
            sentence_overlap: Sentence overlap for sentence strategy
            token_chunk_size: Tokens per chunk for token strategy
            token_overlap: Token overlap for token strategy
            adaptive_short_max: Short passage threshold for adaptive strategy
            adaptive_medium_max: Medium passage threshold for adaptive strategy
            
        Example:
            >>> engine = ChunkingEngine(
            ...     tokenizer=my_tokenizer,
            ...     sentence_chunk_size=5,
            ...     token_chunk_size=512
            ... )
        """
        self.tokenizer = tokenizer
        self.sentence_chunk_size = sentence_chunk_size
        self.sentence_overlap = sentence_overlap
        self.token_chunk_size = token_chunk_size
        self.token_overlap = token_overlap
        self.adaptive_short_max = adaptive_short_max
        self.adaptive_medium_max = adaptive_medium_max
        
        # Chunker registry: maps strategy to factory function
        self._registry: dict[ChunkingStrategy, Callable[[], BaseChunker]] = {}
        
        # Register default chunkers
        self._register_default_chunkers()
    
    def _register_default_chunkers(self) -> None:
        """Register the default set of chunkers."""
        # PassageChunker
        self._registry[ChunkingStrategy.PASSAGE] = lambda: PassageChunker()
        
        # SentenceChunker
        self._registry[ChunkingStrategy.SENTENCE] = lambda: SentenceChunker(
            sentences_per_chunk=self.sentence_chunk_size,
            sentence_overlap=self.sentence_overlap
        )
        
        # TokenChunker (requires tokenizer)
        def create_token_chunker() -> BaseChunker:
            if self.tokenizer is None:
                raise ValueError(
                    f"Strategy {ChunkingStrategy.TOKEN} requires a tokenizer, "
                    f"but none was provided to ChunkingEngine"
                )
            return TokenChunker(
                tokenizer=self.tokenizer,
                chunk_size=self.token_chunk_size,
                token_overlap=self.token_overlap
            )
        self._registry[ChunkingStrategy.TOKEN] = create_token_chunker
        
        # AdaptiveChunker
        self._registry[ChunkingStrategy.ADAPTIVE] = lambda: AdaptiveChunker(
            tokenizer=self.tokenizer,
            short_passage_max_chars=self.adaptive_short_max,
            medium_passage_max_chars=self.adaptive_medium_max,
            token_chunk_size=self.token_chunk_size,
            token_overlap=self.token_overlap,
            sentence_chunk_size=self.sentence_chunk_size,
            sentence_overlap=self.sentence_overlap
        )
    
    def register_chunker(
        self,
        strategy: ChunkingStrategy,
        factory: Callable[[], BaseChunker]
    ) -> None:
        """
        Register a custom chunker for a strategy.
        
        This allows extending the engine with custom chunking implementations.
        
        Args:
            strategy: The strategy enum value
            factory: A callable that creates a chunker instance
            
        Example:
            >>> def create_my_chunker():
            ...     return MyCustomChunker()
            >>> engine.register_chunker(ChunkingStrategy.CUSTOM, create_my_chunker)
        """
        self._registry[strategy] = factory
    
    def _get_chunker(self, strategy: ChunkingStrategy) -> BaseChunker:
        """
        Get a chunker instance for the requested strategy.
        
        Args:
            strategy: The chunking strategy to use
            
        Returns:
            A chunker instance
            
        Raises:
            ValueError: If the strategy is not supported
        """
        if strategy not in self._registry:
            supported = ", ".join(s.value for s in self._registry.keys())
            raise ValueError(
                f"Unsupported chunking strategy: {strategy.value}. "
                f"Supported strategies: {supported}"
            )
        
        # Call the factory to create a fresh chunker instance
        return self._registry[strategy]()
    
    def chunk(
        self,
        passage: CanonicalPassage,
        strategy: ChunkingStrategy
    ) -> list[Chunk]:
        """
        Chunk a single passage using the specified strategy.
        
        Args:
            passage: The canonical passage to chunk
            strategy: The chunking strategy to use
            
        Returns:
            List of Chunks
            
        Raises:
            ValueError: If the strategy is not supported or required
                       dependencies (e.g., tokenizer) are missing
                       
        Example:
            >>> chunks = engine.chunk(passage, ChunkingStrategy.SENTENCE)
            >>> len(chunks)
            3
            >>> chunks[0].strategy
            <ChunkingStrategy.SENTENCE: 'sentence'>
        """
        chunker = self._get_chunker(strategy)
        return chunker.chunk(passage)
    
    def chunk_batch(
        self,
        passages: list[CanonicalPassage],
        strategy: ChunkingStrategy
    ) -> list[Chunk]:
        """
        Chunk a batch of passages using the specified strategy.
        
        Args:
            passages: List of canonical passages to chunk
            strategy: The chunking strategy to use
            
        Returns:
            List of all Chunks from all passages in order
            
        Notes:
            - Input ordering is preserved
            - Input passages are not mutated
            - Empty input returns empty output
            
        Example:
            >>> passages = [passage1, passage2, passage3]
            >>> chunks = engine.chunk_batch(passages, ChunkingStrategy.TOKEN)
            >>> # Returns all chunks from all passages
        """
        chunker = self._get_chunker(strategy)
        return chunker.chunk_batch(passages)
    
    def get_supported_strategies(self) -> list[ChunkingStrategy]:
        """
        Get the list of supported chunking strategies.
        
        Returns:
            List of supported ChunkingStrategy values
            
        Example:
            >>> engine.get_supported_strategies()
            [<ChunkingStrategy.PASSAGE>, <ChunkingStrategy.SENTENCE>, ...]
        """
        return list(self._registry.keys())
    
    def __repr__(self) -> str:
        strategies = [s.value for s in self._registry.keys()]
        return f"ChunkingEngine(strategies={strategies})"
