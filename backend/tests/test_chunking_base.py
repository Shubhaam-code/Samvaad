"""Tests for chunking base interface.

Phase 3.1: Base interface testing (no concrete implementations).
"""

from app.chunking.base import BaseChunker, ChunkerProtocol
from app.chunking.models import Chunk
from app.dataset.models import CanonicalPassage


def test_base_chunker_can_be_imported():
    """Test that BaseChunker can be imported."""
    assert BaseChunker is not None
    assert hasattr(BaseChunker, "chunk")
    assert hasattr(BaseChunker, "chunk_batch")


def test_chunker_protocol_can_be_imported():
    """Test that ChunkerProtocol can be imported."""
    assert ChunkerProtocol is not None


def test_base_chunker_is_abstract():
    """Test that BaseChunker cannot be instantiated directly."""
    try:
        # Should not be able to instantiate abstract class
        chunker = BaseChunker()
        assert False, "Should not be able to instantiate BaseChunker"
    except TypeError:
        # Expected - abstract class cannot be instantiated
        pass


def test_base_chunker_requires_chunk_method():
    """Test that BaseChunker subclass must implement chunk()."""
    class IncompleteChunker(BaseChunker):
        # Missing chunk() implementation
        def chunk_batch(self, passages):
            return []
    
    try:
        chunker = IncompleteChunker()
        assert False, "Should not allow instantiation without chunk()"
    except TypeError:
        # Expected - abstract method not implemented
        pass


def test_base_chunker_requires_chunk_batch_method():
    """Test that BaseChunker subclass must implement chunk_batch()."""
    class IncompleteChunker(BaseChunker):
        def chunk(self, passage):
            return []
        # Missing chunk_batch() implementation
    
    try:
        chunker = IncompleteChunker()
        assert False, "Should not allow instantiation without chunk_batch()"
    except TypeError:
        # Expected - abstract method not implemented
        pass


def test_base_chunker_subclass_with_both_methods():
    """Test that BaseChunker subclass can be created with both methods."""
    class CompleteChunker(BaseChunker):
        def chunk(self, passage: CanonicalPassage) -> list[Chunk]:
            # Dummy implementation for testing
            return []
        
        def chunk_batch(self, passages: list[CanonicalPassage]) -> list[Chunk]:
            # Dummy implementation for testing
            return []
    
    # Should be able to instantiate
    chunker = CompleteChunker()
    assert chunker is not None
    assert isinstance(chunker, BaseChunker)


def test_chunker_protocol_duck_typing():
    """Test that any class with chunk methods satisfies ChunkerProtocol."""
    class DuckTypedChunker:
        """Not inheriting from BaseChunker, but has the right methods."""
        def chunk(self, passage: CanonicalPassage) -> list[Chunk]:
            return []
        
        def chunk_batch(self, passages: list[CanonicalPassage]) -> list[Chunk]:
            return []
    
    chunker = DuckTypedChunker()
    
    # Should satisfy protocol (checked by type checkers at static analysis time)
    # At runtime, we can verify it has the methods
    assert hasattr(chunker, "chunk")
    assert hasattr(chunker, "chunk_batch")
    assert callable(chunker.chunk)
    assert callable(chunker.chunk_batch)


def test_base_chunker_method_signatures():
    """Test that BaseChunker defines correct method signatures."""
    class TestChunker(BaseChunker):
        def chunk(self, passage: CanonicalPassage) -> list[Chunk]:
            return []
        
        def chunk_batch(self, passages: list[CanonicalPassage]) -> list[Chunk]:
            return []
    
    chunker = TestChunker()
    
    # Create a test passage
    passage = CanonicalPassage.from_msmarco_record(
        query_id=1,
        query="test",
        query_type=None,
        answer=None,
        source_lang="en",
        target_lang="hi",
        eng_query="test",
        eng_answer=None,
        passage_index=0,
        translated_passage="test",
        english_passage="test",
        is_selected=False,
    )
    
    # Methods should be callable
    result1 = chunker.chunk(passage)
    result2 = chunker.chunk_batch([passage])
    
    assert isinstance(result1, list)
    assert isinstance(result2, list)
