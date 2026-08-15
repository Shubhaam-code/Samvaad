"""
Tokenizer abstraction for chunking.

Provides a protocol-based interface for tokenization that decouples
chunking logic from specific tokenizer implementations.

Usage Examples:

1. Offline Testing & Development (with SimpleWhitespaceTokenizer):
    >>> from app.chunking.tokenizer import SimpleWhitespaceTokenizer
    >>> tokenizer = SimpleWhitespaceTokenizer()
    >>> ids = tokenizer.encode("Hello world")
    >>> text = tokenizer.decode(ids)

2. Production Usage (with pre-initialized HuggingFace Tokenizer):
    >>> from transformers import AutoTokenizer
    >>> from app.chunking.tokenizer import HuggingFaceTokenizerAdapter
    >>> # Caller must load/cache tokenizer locally beforehand (no automatic network download)
    >>> hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", local_files_only=True)
    >>> adapter = HuggingFaceTokenizerAdapter(hf_tokenizer)
"""

import hashlib
from typing import Protocol, Sequence


class TokenizerProtocol(Protocol):
    """
    Protocol defining the interface for tokenizers used in chunking.
    
    Any tokenizer implementation that provides these methods can be used
    with token-aware chunkers.
    """
    
    def encode(self, text: str) -> Sequence[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: The text to tokenize
            
        Returns:
            Sequence of token IDs
        """
        ...
    
    def decode(self, token_ids: Sequence[int]) -> str:
        """
        Decode token IDs back into text.
        
        Args:
            token_ids: Sequence of token IDs to decode
            
        Returns:
            Decoded text string
        """
        ...
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens
        """
        ...


class HuggingFaceTokenizerAdapter:
    """
    Adapter for HuggingFace transformers tokenizers.
    
    Wraps a HuggingFace PreTrainedTokenizer or PreTrainedTokenizerFast
    to provide the TokenizerProtocol interface.
    
    Note: This adapter does NOT download models automatically. The caller
    must provide a pre-initialized tokenizer instance.
    """
    
    def __init__(self, tokenizer):
        """
        Initialize the adapter with a HuggingFace tokenizer.
        
        Args:
            tokenizer: A HuggingFace tokenizer instance (PreTrainedTokenizer
                      or PreTrainedTokenizerFast)
                      
        Raises:
            ValueError: If tokenizer is None
                      
        Example:
            >>> from transformers import AutoTokenizer
            >>> hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
            >>> adapter = HuggingFaceTokenizerAdapter(hf_tokenizer)
        """
        if tokenizer is None:
            raise ValueError("HuggingFaceTokenizerAdapter requires a valid tokenizer instance")
        self._tokenizer = tokenizer
    
    def encode(self, text: str) -> Sequence[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: The text to tokenize
            
        Returns:
            List of token IDs
        """
        if not text or not text.strip():
            return []
        return self._tokenizer.encode(text, add_special_tokens=False)
    
    def decode(self, token_ids: Sequence[int]) -> str:
        """
        Decode token IDs back into text.
        
        Args:
            token_ids: Sequence of token IDs to decode
            
        Returns:
            Decoded text string
        """
        if not token_ids:
            return ""
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text or not text.strip():
            return 0
        return len(self.encode(text))
    
    def __repr__(self) -> str:
        return f"HuggingFaceTokenizerAdapter(tokenizer={self._tokenizer.__class__.__name__})"


class SimpleWhitespaceTokenizer:
    """
    Simple whitespace-based tokenizer for testing, algorithm verification, and fallback.
    
    This is NOT a production tokenizer. It is provided for:
    - Testing token-based chunking logic without HuggingFace models
    - Offline development, debugging, and synthetic benchmarks
    - Fallback when no production HuggingFace tokenizer is available
    
    Behavior:
    - Tokenization is based on whitespace splitting (`str.split()`).
    - Maintains a deterministic vocabulary mapping words to stable integer IDs.
    - `decode()` reconstructs space-separated words. Note that multiple spaces
      or original line breaks are normalized to single spaces upon decoding.
    """
    
    def __init__(self) -> None:
        self._word_to_id: dict[str, int] = {}
        self._id_to_word: dict[int, str] = {}
    
    def encode(self, text: str) -> Sequence[int]:
        """
        Encode text into pseudo-token IDs using whitespace splitting.
        
        Each unique word is assigned a deterministic integer ID.
        
        Args:
            text: The text to tokenize
            
        Returns:
            List of integer token IDs
        """
        if not text or not text.strip():
            return []
        
        words = text.split()
        token_ids = []
        for word in words:
            if word not in self._word_to_id:
                # Generate deterministic hash modulo 1,000,000
                digest = hashlib.md5(word.encode("utf-8")).hexdigest()
                word_id = int(digest[:6], 16) % 1000000
                while word_id in self._id_to_word and self._id_to_word[word_id] != word:
                    word_id = (word_id + 1) % 1000000
                self._word_to_id[word] = word_id
                self._id_to_word[word_id] = word
            token_ids.append(self._word_to_id[word])
        return token_ids
    
    def decode(self, token_ids: Sequence[int]) -> str:
        """
        Decode token IDs back into space-separated words.
        
        Args:
            token_ids: Sequence of token IDs
            
        Returns:
            Decoded text string (space-separated words)
        """
        if not token_ids:
            return ""
        words = [self._id_to_word.get(tid, f"[{tid}]") for tid in token_ids]
        return " ".join(words)
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using whitespace splitting.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of whitespace-separated words
        """
        if not text or not text.strip():
            return 0
        return len(text.split())
    
    def __repr__(self) -> str:
        return "SimpleWhitespaceTokenizer()"


def create_huggingface_tokenizer(
    model_name_or_path: str = "bert-base-multilingual-cased",
    local_files_only: bool = True,
) -> HuggingFaceTokenizerAdapter:
    """
    Create a HuggingFaceTokenizerAdapter for a local model/tokenizer.
    
    Args:
        model_name_or_path: HuggingFace model identifier or local directory path
        local_files_only: Force loading from local cache only (default True).
                          Prevents automatic downloading over the network.
                          
    Returns:
        HuggingFaceTokenizerAdapter wrapping the loaded tokenizer
        
    Raises:
        RuntimeError: If transformers is not installed or model is not cached locally
    """
    try:
        from transformers import AutoTokenizer
    except (ImportError, OSError) as e:
        raise RuntimeError(
            f"Failed to load HuggingFace tokenizer '{model_name_or_path}': "
            f"transformers package failed to load: {e}"
        ) from e

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only
        )
        return HuggingFaceTokenizerAdapter(tokenizer)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load HuggingFace tokenizer '{model_name_or_path}' locally. "
            f"Ensure model is cached locally (local_files_only={local_files_only}): {e}"
        ) from e


def create_test_tokenizer() -> SimpleWhitespaceTokenizer:
    """
    Create a SimpleWhitespaceTokenizer explicitly for testing and debugging.
    
    Returns:
        An instance of SimpleWhitespaceTokenizer
    """
    return SimpleWhitespaceTokenizer()


def create_fallback_tokenizer() -> SimpleWhitespaceTokenizer:
    """
    Create a fallback SimpleWhitespaceTokenizer when no production model is present.
    
    Returns:
        An instance of SimpleWhitespaceTokenizer
    """
    return SimpleWhitespaceTokenizer()


def create_default_tokenizer() -> TokenizerProtocol:
    """
    Create a default tokenizer for testing and development.
    
    Attempts to load a local HuggingFace multilingual tokenizer without downloading.
    If not cached locally, returns SimpleWhitespaceTokenizer as testing fallback.
    
    Returns:
        A tokenizer conforming to TokenizerProtocol
    """
    try:
        return create_huggingface_tokenizer("bert-base-multilingual-cased", local_files_only=True)
    except Exception:
        return SimpleWhitespaceTokenizer()
