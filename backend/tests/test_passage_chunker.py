"""
Tests for PassageChunker.

Tests the passage-preserving chunking strategy that converts each
CanonicalPassage into exactly one Chunk without splitting or modification.
"""

import pytest

from app.chunking.models import Chunk, ChunkingStrategy
from app.chunking.passage_chunker import PassageChunker
from app.dataset.models import CanonicalPassage


def create_test_passage(
    document_id: str = "doc1",
    translated_passage: str = "This is a test passage.",
    english_passage: str = "This is a test passage.",
    query_id: int = 1,
    passage_index: int = 0,
    target_lang: str = "hi",
    source_lang: str = "en",
    query: str = "test query",
    eng_query: str = "test query",
    is_selected: bool = True,
    query_type: str | None = None,
    answer: str | None = None,
    eng_answer: str | None = None,
) -> CanonicalPassage:
    """Helper to create test CanonicalPassage instances."""
    return CanonicalPassage(
        document_id=document_id,
        translated_passage=translated_passage,
        english_passage=english_passage,
        query_id=query_id,
        passage_index=passage_index,
        target_lang=target_lang,
        source_lang=source_lang,
        query=query,
        eng_query=eng_query,
        is_selected=is_selected,
        query_type=query_type,
        answer=answer,
        eng_answer=eng_answer,
    )


class TestPassageChunkerBasics:
    """Test basic PassageChunker behavior."""
    
    def test_one_passage_one_chunk(self):
        """Test that one passage produces exactly one chunk."""
        chunker = PassageChunker()
        passage = create_test_passage()
        
        chunks = chunker.chunk(passage)
        
        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
    
    def test_chunk_text_preservation(self):
        """Test that chunk_text exactly matches translated_passage."""
        chunker = PassageChunker()
        passage_text = "This is the original passage text that must not be modified."
        passage = create_test_passage(translated_passage=passage_text)
        
        chunks = chunker.chunk(passage)
        
        assert chunks[0].chunk_text == passage_text
    
    def test_passage_strategy(self):
        """Test that PASSAGE strategy is used."""
        chunker = PassageChunker()
        passage = create_test_passage()
        
        chunks = chunker.chunk(passage)
        
        assert chunks[0].strategy == ChunkingStrategy.PASSAGE
    
    def test_chunk_index_is_zero(self):
        """Test that chunk_index is always 0 for passage chunking."""
        chunker = PassageChunker()
        passage = create_test_passage()
        
        chunks = chunker.chunk(passage)
        
        assert chunks[0].chunk_index == 0


class TestPassageChunkerMetadata:
    """Test metadata preservation."""
    
    def test_metadata_preservation(self):
        """Test that all source metadata is preserved."""
        chunker = PassageChunker()
        passage = create_test_passage(
            document_id="doc123",
            query_id=42,
            passage_index=5,
            target_lang="hi",
            source_lang="en",
            query="प्रश्न",
            eng_query="question",
            query_type="factoid",
            answer="उत्तर",
            eng_answer="answer",
            is_selected=True,
        )
        
        chunks = chunker.chunk(passage)
        chunk = chunks[0]
        
        assert chunk.document_id == "doc123"
        assert chunk.query_id == 42
        assert chunk.passage_index == 5
        assert chunk.target_lang == "hi"
        assert chunk.source_lang == "en"
        assert chunk.query == "प्रश्न"
        assert chunk.eng_query == "question"
        assert chunk.query_type == "factoid"
        assert chunk.answer == "उत्तर"
        assert chunk.eng_answer == "answer"
        assert chunk.is_selected is True
    
    def test_optional_metadata_none(self):
        """Test handling of optional metadata when None."""
        chunker = PassageChunker()
        passage = create_test_passage(
            query_type=None,
            answer=None,
            eng_answer=None,
        )
        
        chunks = chunker.chunk(passage)
        chunk = chunks[0]
        
        assert chunk.query_type is None
        assert chunk.answer is None
        assert chunk.eng_answer is None


class TestPassageChunkerUnicode:
    """Test Unicode and multilingual text handling."""
    
    def test_hindi_unicode_preservation(self):
        """Test that Hindi Unicode text is preserved exactly."""
        chunker = PassageChunker()
        hindi_text = "यह एक परीक्षण पैसेज है। इसमें हिंदी पाठ है।"
        passage = create_test_passage(translated_passage=hindi_text)
        
        chunks = chunker.chunk(passage)
        
        assert chunks[0].chunk_text == hindi_text
    
    def test_mixed_script_preservation(self):
        """Test preservation of mixed English/Hindi text."""
        chunker = PassageChunker()
        mixed_text = "This is English. यह हिंदी है। Mixed text."
        passage = create_test_passage(translated_passage=mixed_text)
        
        chunks = chunker.chunk(passage)
        
        assert chunks[0].chunk_text == mixed_text
    
    def test_punctuation_preservation(self):
        """Test that punctuation is preserved exactly."""
        chunker = PassageChunker()
        punctuated = "Hello! How are you? I'm fine. What about you...?"
        passage = create_test_passage(translated_passage=punctuated)
        
        chunks = chunker.chunk(passage)
        
        assert chunks[0].chunk_text == punctuated
    
    def test_whitespace_preservation(self):
        """Test that whitespace is preserved."""
        chunker = PassageChunker()
        spaced_text = "Line 1\nLine 2\n\nLine 3   with   spaces"
        passage = create_test_passage(translated_passage=spaced_text)
        
        chunks = chunker.chunk(passage)
        
        assert chunks[0].chunk_text == spaced_text


class TestPassageChunkerMetrics:
    """Test chunk metrics calculation."""
    
    def test_character_count(self):
        """Test that character_count equals len(chunk_text)."""
        chunker = PassageChunker()
        text = "This is a test passage with some length."
        passage = create_test_passage(translated_passage=text)
        
        chunks = chunker.chunk(passage)
        
        assert chunks[0].character_count == len(text)
    
    def test_character_count_hindi(self):
        """Test character_count with Hindi text."""
        chunker = PassageChunker()
        hindi_text = "यह परीक्षण है।"
        passage = create_test_passage(translated_passage=hindi_text)
        
        chunks = chunker.chunk(passage)
        
        assert chunks[0].character_count == len(hindi_text)
    
    def test_offsets(self):
        """Test that start_offset and end_offset are set."""
        chunker = PassageChunker()
        text = "Test passage"
        passage = create_test_passage(translated_passage=text)
        
        chunks = chunker.chunk(passage)
        
        assert chunks[0].start_offset == 0
        assert chunks[0].end_offset == len(text)
    
    def test_no_token_count(self):
        """Test that token_count is None (no tokenizer)."""
        chunker = PassageChunker()
        passage = create_test_passage()
        
        chunks = chunker.chunk(passage)
        
        assert chunks[0].token_count is None
    
    def test_overlap_zero(self):
        """Test that overlap values are 0 for passage chunking."""
        chunker = PassageChunker()
        passage = create_test_passage()
        
        chunks = chunker.chunk(passage)
        
        assert chunks[0].overlap_before == 0
        assert chunks[0].overlap_after == 0


class TestPassageChunkerDeterminism:
    """Test deterministic behavior."""
    
    def test_deterministic_chunk_id(self):
        """Test that chunk_id is deterministic."""
        chunker = PassageChunker()
        passage = create_test_passage(document_id="doc1")
        
        chunks1 = chunker.chunk(passage)
        chunks2 = chunker.chunk(passage)
        
        assert chunks1[0].chunk_id == chunks2[0].chunk_id
    
    def test_repeated_execution_identical(self):
        """Test that repeated execution produces identical chunks."""
        chunker = PassageChunker()
        passage = create_test_passage()
        
        chunks1 = chunker.chunk(passage)
        chunks2 = chunker.chunk(passage)
        
        assert chunks1[0].model_dump() == chunks2[0].model_dump()
    
    def test_different_passages_different_ids(self):
        """Test that different passages produce different chunk IDs."""
        chunker = PassageChunker()
        passage1 = create_test_passage(document_id="doc1")
        passage2 = create_test_passage(document_id="doc2")
        
        chunks1 = chunker.chunk(passage1)
        chunks2 = chunker.chunk(passage2)
        
        assert chunks1[0].chunk_id != chunks2[0].chunk_id


class TestPassageChunkerBatch:
    """Test batch processing."""
    
    def test_batch_one_chunk_per_passage(self):
        """Test that batch processing creates one chunk per passage."""
        chunker = PassageChunker()
        passages = [
            create_test_passage(document_id="doc1"),
            create_test_passage(document_id="doc2"),
            create_test_passage(document_id="doc3"),
        ]
        
        chunks = chunker.chunk_batch(passages)
        
        assert len(chunks) == 3
    
    def test_batch_ordering_preserved(self):
        """Test that input ordering is preserved in batch processing."""
        chunker = PassageChunker()
        passages = [
            create_test_passage(document_id="doc1", translated_passage="First"),
            create_test_passage(document_id="doc2", translated_passage="Second"),
            create_test_passage(document_id="doc3", translated_passage="Third"),
        ]
        
        chunks = chunker.chunk_batch(passages)
        
        assert chunks[0].chunk_text == "First"
        assert chunks[1].chunk_text == "Second"
        assert chunks[2].chunk_text == "Third"
    
    def test_batch_empty_input(self):
        """Test that empty batch returns empty list."""
        chunker = PassageChunker()
        
        chunks = chunker.chunk_batch([])
        
        assert chunks == []
    
    def test_batch_no_mutation(self):
        """Test that batch processing does not mutate input passages."""
        chunker = PassageChunker()
        passage = create_test_passage(document_id="doc1")
        original_dict = passage.model_dump()
        
        chunker.chunk_batch([passage])
        
        assert passage.model_dump() == original_dict


class TestPassageChunkerEdgeCases:
    """Test edge cases and unusual inputs."""
    
    def test_long_passage(self):
        """Test that long passages remain intact."""
        chunker = PassageChunker()
        long_text = "This is a sentence. " * 100  # Very long passage
        passage = create_test_passage(translated_passage=long_text)
        
        chunks = chunker.chunk(passage)
        
        assert len(chunks) == 1
        assert chunks[0].chunk_text == long_text
    
    def test_short_passage(self):
        """Test handling of very short passages."""
        chunker = PassageChunker()
        passage = create_test_passage(translated_passage="Hi")
        
        chunks = chunker.chunk(passage)
        
        assert len(chunks) == 1
        assert chunks[0].chunk_text == "Hi"
    
    def test_single_word(self):
        """Test single-word passage."""
        chunker = PassageChunker()
        passage = create_test_passage(translated_passage="Word")
        
        chunks = chunker.chunk(passage)
        
        assert len(chunks) == 1
        assert chunks[0].chunk_text == "Word"


class TestPassageChunkerConformance:
    """Test conformance to BaseChunker interface."""
    
    def test_conforms_to_base_chunker(self):
        """Test that PassageChunker properly implements BaseChunker."""
        from app.chunking.base import BaseChunker
        
        chunker = PassageChunker()
        
        assert isinstance(chunker, BaseChunker)
        assert hasattr(chunker, "chunk")
        assert hasattr(chunker, "chunk_batch")
        assert callable(chunker.chunk)
        assert callable(chunker.chunk_batch)
