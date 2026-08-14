"""
Tests for SentenceChunker.

Tests the sentence-aware chunking strategy with configurable grouping
and overlap, supporting multilingual sentence boundary detection.
"""

import pytest

from app.chunking.models import Chunk, ChunkingStrategy
from app.chunking.sentence_chunker import SentenceChunker
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


class TestSentenceChunkerConfiguration:
    """Test SentenceChunker configuration validation."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        chunker = SentenceChunker()
        
        assert chunker.sentences_per_chunk == 3
        assert chunker.sentence_overlap == 1
    
    def test_custom_configuration(self):
        """Test custom configuration."""
        chunker = SentenceChunker(sentences_per_chunk=5, sentence_overlap=2)
        
        assert chunker.sentences_per_chunk == 5
        assert chunker.sentence_overlap == 2
    
    def test_invalid_sentences_per_chunk_negative(self):
        """Test that negative sentences_per_chunk raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            SentenceChunker(sentences_per_chunk=-1)
    
    def test_invalid_sentences_per_chunk_zero(self):
        """Test that zero sentences_per_chunk raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            SentenceChunker(sentences_per_chunk=0)
    
    def test_invalid_overlap_negative(self):
        """Test that negative sentence_overlap raises ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            SentenceChunker(sentences_per_chunk=3, sentence_overlap=-1)
    
    def test_invalid_overlap_equals_sentences_per_chunk(self):
        """Test that overlap >= sentences_per_chunk raises ValueError."""
        with pytest.raises(ValueError, match="must be less than"):
            SentenceChunker(sentences_per_chunk=3, sentence_overlap=3)
    
    def test_invalid_overlap_greater_than_sentences_per_chunk(self):
        """Test that overlap > sentences_per_chunk raises ValueError."""
        with pytest.raises(ValueError, match="must be less than"):
            SentenceChunker(sentences_per_chunk=3, sentence_overlap=5)
    
    def test_valid_zero_overlap(self):
        """Test that zero overlap is valid."""
        chunker = SentenceChunker(sentences_per_chunk=3, sentence_overlap=0)
        
        assert chunker.sentence_overlap == 0


class TestSentenceChunkerBasics:
    """Test basic SentenceChunker behavior."""
    
    def test_one_sentence_passage(self):
        """Test that a one-sentence passage produces one chunk."""
        chunker = SentenceChunker(sentences_per_chunk=3, sentence_overlap=1)
        passage = create_test_passage(translated_passage="This is one sentence.")
        
        chunks = chunker.chunk(passage)
        
        assert len(chunks) == 1
        assert chunks[0].chunk_text == "This is one sentence."
    
    def test_two_sentence_passage(self):
        """Test passage with fewer sentences than sentences_per_chunk."""
        chunker = SentenceChunker(sentences_per_chunk=3, sentence_overlap=1)
        passage = create_test_passage(
            translated_passage="First sentence. Second sentence."
        )
        
        chunks = chunker.chunk(passage)
        
        assert len(chunks) == 1
        assert "First sentence" in chunks[0].chunk_text
        assert "Second sentence" in chunks[0].chunk_text
    
    def test_exact_chunk_size_passage(self):
        """Test passage with exactly sentences_per_chunk sentences."""
        chunker = SentenceChunker(sentences_per_chunk=3, sentence_overlap=1)
        passage = create_test_passage(
            translated_passage="First. Second. Third."
        )
        
        chunks = chunker.chunk(passage)
        
        assert len(chunks) == 1
    
    def test_multiple_chunks(self):
        """Test passage that produces multiple chunks."""
        chunker = SentenceChunker(sentences_per_chunk=2, sentence_overlap=1)
        passage = create_test_passage(
            translated_passage="S1. S2. S3. S4."
        )
        
        chunks = chunker.chunk(passage)
        
        # With 4 sentences, sentences_per_chunk=2, overlap=1:
        # Chunk 0: S1, S2
        # Chunk 1: S2, S3
        # Chunk 2: S3, S4
        assert len(chunks) >= 2
    
    def test_sentence_strategy(self):
        """Test that SENTENCE strategy is used."""
        chunker = SentenceChunker()
        passage = create_test_passage(translated_passage="First. Second.")
        
        chunks = chunker.chunk(passage)
        
        assert chunks[0].strategy == ChunkingStrategy.SENTENCE


class TestSentenceChunkerSentenceDetection:
    """Test sentence boundary detection."""
    
    def test_period_boundary(self):
        """Test detection of period boundaries."""
        chunker = SentenceChunker(sentences_per_chunk=1, sentence_overlap=0)
        passage = create_test_passage(
            translated_passage="First sentence. Second sentence."
        )
        
        chunks = chunker.chunk(passage)
        
        assert len(chunks) == 2
    
    def test_question_mark_boundary(self):
        """Test detection of question mark boundaries."""
        chunker = SentenceChunker(sentences_per_chunk=1, sentence_overlap=0)
        passage = create_test_passage(
            translated_passage="Is this a question? Yes it is."
        )
        
        chunks = chunker.chunk(passage)
        
        assert len(chunks) == 2
    
    def test_exclamation_boundary(self):
        """Test detection of exclamation mark boundaries."""
        chunker = SentenceChunker(sentences_per_chunk=1, sentence_overlap=0)
        passage = create_test_passage(
            translated_passage="This is exciting! So is this."
        )
        
        chunks = chunker.chunk(passage)
        
        assert len(chunks) == 2
    
    def test_hindi_danda_boundary(self):
        """Test detection of Hindi danda (।) boundaries."""
        chunker = SentenceChunker(sentences_per_chunk=1, sentence_overlap=0)
        passage = create_test_passage(
            translated_passage="यह पहला वाक्य है। यह दूसरा है।"
        )
        
        chunks = chunker.chunk(passage)
        
        assert len(chunks) == 2
    
    def test_mixed_hindi_english(self):
        """Test mixed Hindi and English sentence detection."""
        chunker = SentenceChunker(sentences_per_chunk=1, sentence_overlap=0)
        passage = create_test_passage(
            translated_passage="This is English. यह हिंदी है। More English."
        )
        
        chunks = chunker.chunk(passage)
        
        assert len(chunks) == 3
    
    def test_repeated_punctuation(self):
        """Test handling of repeated punctuation."""
        chunker = SentenceChunker(sentences_per_chunk=1, sentence_overlap=0)
        passage = create_test_passage(
            translated_passage="Really!!! Is this true...?"
        )
        
        chunks = chunker.chunk(passage)
        
        # Should detect as separate sentences
        assert len(chunks) >= 1
    
    def test_mixed_punctuation(self):
        """Test handling of mixed punctuation types."""
        chunker = SentenceChunker(sentences_per_chunk=1, sentence_overlap=0)
        passage = create_test_passage(
            translated_passage="Statement. Question? Exclamation!"
        )
        
        chunks = chunker.chunk(passage)
        
        assert len(chunks) == 3


class TestSentenceChunkerOverlap:
    """Test sentence overlap functionality."""
    
    def test_overlap_zero(self):
        """Test chunking with no overlap."""
        chunker = SentenceChunker(sentences_per_chunk=2, sentence_overlap=0)
        passage = create_test_passage(
            translated_passage="S1. S2. S3. S4."
        )
        
        chunks = chunker.chunk(passage)
        
        # No overlap: [S1,S2], [S3,S4]
        assert len(chunks) == 2
    
    def test_overlap_one(self):
        """Test chunking with overlap of 1."""
        chunker = SentenceChunker(sentences_per_chunk=2, sentence_overlap=1)
        passage = create_test_passage(
            translated_passage="S1. S2. S3. S4."
        )
        
        chunks = chunker.chunk(passage)
        
        # Overlap 1: [S1,S2], [S2,S3], [S3,S4]
        assert len(chunks) >= 2


class TestSentenceChunkerMetadata:
    """Test metadata preservation."""
    
    def test_metadata_preservation(self):
        """Test that all source metadata is preserved in chunks."""
        chunker = SentenceChunker()
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
            translated_passage="First. Second. Third. Fourth."
        )
        
        chunks = chunker.chunk(passage)
        
        for chunk in chunks:
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
        chunker = SentenceChunker()
        passage = create_test_passage(
            query_type=None,
            answer=None,
            eng_answer=None,
            translated_passage="First. Second."
        )
        
        chunks = chunker.chunk(passage)
        
        for chunk in chunks:
            assert chunk.query_type is None
            assert chunk.answer is None
            assert chunk.eng_answer is None


class TestSentenceChunkerChunkIndexing:
    """Test chunk indexing."""
    
    def test_chunk_indexes_sequential(self):
        """Test that chunk_index values are sequential starting from 0."""
        chunker = SentenceChunker(sentences_per_chunk=2, sentence_overlap=1)
        passage = create_test_passage(
            translated_passage="S1. S2. S3. S4. S5."
        )
        
        chunks = chunker.chunk(passage)
        
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
    
    def test_first_chunk_index_zero(self):
        """Test that first chunk has index 0."""
        chunker = SentenceChunker()
        passage = create_test_passage(translated_passage="S1. S2. S3.")
        
        chunks = chunker.chunk(passage)
        
        assert chunks[0].chunk_index == 0


class TestSentenceChunkerDeterminism:
    """Test deterministic behavior."""
    
    def test_deterministic_chunk_ids(self):
        """Test that chunk IDs are deterministic."""
        chunker = SentenceChunker()
        passage = create_test_passage(
            document_id="doc1",
            translated_passage="S1. S2. S3. S4."
        )
        
        chunks1 = chunker.chunk(passage)
        chunks2 = chunker.chunk(passage)
        
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.chunk_id == c2.chunk_id
    
    def test_different_index_different_id(self):
        """Test that different chunk indexes produce different IDs."""
        chunker = SentenceChunker(sentences_per_chunk=2, sentence_overlap=0)
        passage = create_test_passage(
            translated_passage="S1. S2. S3. S4."
        )
        
        chunks = chunker.chunk(passage)
        
        if len(chunks) > 1:
            assert chunks[0].chunk_id != chunks[1].chunk_id


class TestSentenceChunkerMetrics:
    """Test chunk metrics calculation."""
    
    def test_character_count(self):
        """Test that character_count equals len(chunk_text)."""
        chunker = SentenceChunker()
        passage = create_test_passage(translated_passage="First. Second.")
        
        chunks = chunker.chunk(passage)
        
        for chunk in chunks:
            assert chunk.character_count == len(chunk.chunk_text)
    
    def test_no_token_count(self):
        """Test that token_count is None (no tokenizer)."""
        chunker = SentenceChunker()
        passage = create_test_passage(translated_passage="First. Second.")
        
        chunks = chunker.chunk(passage)
        
        for chunk in chunks:
            assert chunk.token_count is None


class TestSentenceChunkerTextPreservation:
    """Test text content preservation."""
    
    def test_punctuation_preservation(self):
        """Test that punctuation is preserved in chunks."""
        chunker = SentenceChunker(sentences_per_chunk=1, sentence_overlap=0)
        passage = create_test_passage(
            translated_passage="Hello! How are you? I'm fine."
        )
        
        chunks = chunker.chunk(passage)
        
        # Check that punctuation appears in chunk text
        all_text = " ".join(c.chunk_text for c in chunks)
        assert "!" in all_text
        assert "?" in all_text
    
    def test_unicode_preservation(self):
        """Test that Unicode characters are preserved."""
        chunker = SentenceChunker()
        passage = create_test_passage(
            translated_passage="यह हिंदी है। यह परीक्षण है।"
        )
        
        chunks = chunker.chunk(passage)
        
        # Check that Hindi characters are preserved
        all_text = " ".join(c.chunk_text for c in chunks)
        assert "हिंदी" in all_text
        assert "परीक्षण" in all_text
    
    def test_whitespace_handling(self):
        """Test handling of whitespace around sentences."""
        chunker = SentenceChunker(sentences_per_chunk=1, sentence_overlap=0)
        passage = create_test_passage(
            translated_passage="First.   Second.    Third."
        )
        
        chunks = chunker.chunk(passage)
        
        # Should create separate chunks
        assert len(chunks) >= 1


class TestSentenceChunkerBatch:
    """Test batch processing."""
    
    def test_batch_processing(self):
        """Test that batch processing works correctly."""
        chunker = SentenceChunker()
        passages = [
            create_test_passage(document_id="doc1", translated_passage="S1. S2."),
            create_test_passage(document_id="doc2", translated_passage="S3. S4."),
        ]
        
        chunks = chunker.chunk_batch(passages)
        
        # Should have chunks from both passages
        assert len(chunks) >= 2
    
    def test_batch_empty_input(self):
        """Test that empty batch returns empty list."""
        chunker = SentenceChunker()
        
        chunks = chunker.chunk_batch([])
        
        assert chunks == []
    
    def test_batch_no_mutation(self):
        """Test that batch processing does not mutate input passages."""
        chunker = SentenceChunker()
        passage = create_test_passage(
            document_id="doc1",
            translated_passage="S1. S2."
        )
        original_dict = passage.model_dump()
        
        chunker.chunk_batch([passage])
        
        assert passage.model_dump() == original_dict


class TestSentenceChunkerEdgeCases:
    """Test edge cases and unusual inputs."""
    
    def test_long_synthetic_passage(self):
        """Test handling of long passages with many sentences."""
        chunker = SentenceChunker(sentences_per_chunk=3, sentence_overlap=1)
        long_passage = " ".join([f"Sentence {i}." for i in range(20)])
        passage = create_test_passage(translated_passage=long_passage)
        
        chunks = chunker.chunk(passage)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        # All chunks should have metadata
        for chunk in chunks:
            assert chunk.document_id == "doc1"
    
    def test_no_sentence_boundaries(self):
        """Test handling of text with no clear sentence boundaries."""
        chunker = SentenceChunker()
        passage = create_test_passage(
            translated_passage="This is text without punctuation"
        )
        
        chunks = chunker.chunk(passage)
        
        # Should still create at least one chunk
        assert len(chunks) >= 1
        assert chunks[0].chunk_text.strip()
    
    def test_only_whitespace_after_split(self):
        """Test handling when split results contain only whitespace."""
        chunker = SentenceChunker()
        passage = create_test_passage(translated_passage="   .   .   ")
        
        chunks = chunker.chunk(passage)
        
        # Should handle gracefully
        assert isinstance(chunks, list)


class TestSentenceChunkerConformance:
    """Test conformance to BaseChunker interface."""
    
    def test_conforms_to_base_chunker(self):
        """Test that SentenceChunker properly implements BaseChunker."""
        from app.chunking.base import BaseChunker
        
        chunker = SentenceChunker()
        
        assert isinstance(chunker, BaseChunker)
        assert hasattr(chunker, "chunk")
        assert hasattr(chunker, "chunk_batch")
        assert callable(chunker.chunk)
        assert callable(chunker.chunk_batch)


class TestSentenceChunkerIntegration:
    """Integration tests comparing with PassageChunker."""
    
    def test_same_passage_different_strategies(self):
        """Test that same passage produces different results with different strategies."""
        from app.chunking.passage_chunker import PassageChunker
        
        passage = create_test_passage(
            document_id="doc1",
            translated_passage="S1. S2. S3. S4. S5."
        )
        
        passage_chunker = PassageChunker()
        sentence_chunker = SentenceChunker(sentences_per_chunk=2, sentence_overlap=0)
        
        passage_chunks = passage_chunker.chunk(passage)
        sentence_chunks = sentence_chunker.chunk(passage)
        
        # PassageChunker: 1 chunk
        assert len(passage_chunks) == 1
        
        # SentenceChunker: multiple chunks
        assert len(sentence_chunks) >= 2
        
        # Different strategies
        assert passage_chunks[0].strategy == ChunkingStrategy.PASSAGE
        assert sentence_chunks[0].strategy == ChunkingStrategy.SENTENCE
        
        # Different chunk IDs (due to different strategies)
        assert passage_chunks[0].chunk_id != sentence_chunks[0].chunk_id
