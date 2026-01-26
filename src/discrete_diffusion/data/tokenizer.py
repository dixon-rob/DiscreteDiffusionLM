"""Tokenizers for discrete diffusion."""

from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer


class CharacterLevelTokenizer(PreTrainedTokenizer):
    """
    Character-level tokenizer compatible with HuggingFace Transformers.

    Vocabulary includes:
    - Special tokens: <UNK>, <PAD>
    - Digits: 0-9
    - Letters: a-z, A-Z
    - Common punctuation and whitespace
    - Some Unicode characters (curly quotes, em-dash, ellipsis)
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, **kwargs):
        """Initialize the tokenizer with a fixed character vocabulary."""
        self.unk_token = "<UNK>"
        self.vocab = [
            self.unk_token,
            "<PAD>",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "!",
            '"',
            "#",
            "$",
            "%",
            "&",
            "'",
            "(",
            ")",
            "*",
            "+",
            ",",
            "-",
            ".",
            "/",
            ":",
            ";",
            "<",
            "=",
            ">",
            "?",
            "@",
            "[",
            "\\",
            "]",
            "^",
            "_",
            "`",
            "{",
            "|",
            "}",
            "~",
            " ",
            "\t",
            "\n",
            "\u201c",  # "
            "\u201d",  # "
            "\u2018",  # '
            "\u2019",  # '
            "\u2014",  # —
            "\u2013",  # –
            "\u2026",  # …
        ]
        self._char_to_id = {ch: idx for idx, ch in enumerate(self.vocab)}
        self._id_to_char = {idx: ch for idx, ch in enumerate(self.vocab)}
        super().__init__(
            unk_token=self.unk_token,
            pad_token="<PAD>",
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return self._char_to_id.copy()

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return [ch if ch in self._char_to_id else self.unk_token for ch in text]

    def _convert_token_to_id(self, token: str) -> int:
        return self._char_to_id.get(token, self._char_to_id[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_char.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[()]:
        """
        Save the vocabulary to a file.
        Required by PreTrainedTokenizer.

        Args:
            save_directory: Directory to save to
            filename_prefix: Optional prefix for filename

        Returns:
            Tuple of saved file paths - empty since vocab is hardcoded
        """
        return ()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Create a fresh tokenizer instance.
        No data loading needed since vocabulary is hardcoded.
        """
        return cls(**kwargs)