"""Tests for the character-level tokenizer."""

import pytest

from discrete_diffusion import CharacterLevelTokenizer


@pytest.fixture
def tokenizer():
    return CharacterLevelTokenizer()


class TestCharacterLevelTokenizer:
    def test_creation(self, tokenizer):
        assert tokenizer is not None
        assert tokenizer.vocab_size > 90  # Flexible check for vocab size

    def test_encode_decode_roundtrip(self, tokenizer):
        text = "Hello, World!"
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)

        assert decoded == text

    def test_encode_returns_list(self, tokenizer):
        text = "test"
        encoded = tokenizer.encode(text, add_special_tokens=False)

        assert isinstance(encoded, list)
        assert len(encoded) == 4  # One per character

    def test_unknown_char_handling(self, tokenizer):
        text = "test\x00unknown"  # Null char not in vocab
        encoded = tokenizer.encode(text, add_special_tokens=False)

        # Should encode without error, unknown chars become UNK
        assert len(encoded) == len(text)

    def test_special_tokens(self, tokenizer):
        assert tokenizer.unk_token == "<UNK>"
        assert tokenizer.pad_token == "<PAD>"
        assert tokenizer.pad_token_id == 1

    def test_vocab_coverage(self, tokenizer):
        # Common characters should be in vocab
        for char in "abcdefghijklmnopqrstuvwxyz":
            assert char in tokenizer.get_vocab()

        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            assert char in tokenizer.get_vocab()

        for char in "0123456789":
            assert char in tokenizer.get_vocab()

        for char in " \t\n":
            assert char in tokenizer.get_vocab()

    def test_from_pretrained(self):
        # Should work without any path (no data to load)
        tokenizer = CharacterLevelTokenizer.from_pretrained()
        assert tokenizer.vocab_size > 90  # Flexible check for vocab size

    def test_tokenize(self, tokenizer):
        tokens = tokenizer._tokenize("abc")
        assert tokens == ["a", "b", "c"]

    def test_convert_tokens_to_string(self, tokenizer):
        tokens = ["H", "i", "!"]
        text = tokenizer.convert_tokens_to_string(tokens)
        assert text == "Hi!"
