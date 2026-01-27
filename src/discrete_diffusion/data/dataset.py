"""Dataset utilities for discrete diffusion training."""

from typing import Dict, List, Tuple

from datasets import load_dataset
from torch.utils.data import DataLoader

from .tokenizer import CharacterLevelTokenizer


def tokenize_and_chunk(
    tokenizer: CharacterLevelTokenizer,
    examples: Dict[str, List],
    context_len: int = 256,
) -> Dict[str, List]:
    """
    Tokenize texts and chunk into fixed-length sequences.

    Args:
        tokenizer: CharacterLevelTokenizer instance
        examples: Batch of examples from dataset, with 'text' column
        context_len: Length of each sequence (in tokens)

    Returns:
        Dictionary with 'input_ids' key containing list of chunked sequences
    """
    all_input_ids = []

    for text in examples["text"]:
        if len(text.strip()) == 0:
            continue
        encoded = tokenizer.encode(text, add_special_tokens=False)

        # Non-overlapping chunks
        for i in range(0, len(encoded) - context_len + 1, context_len):
            chunk = encoded[i : i + context_len]
            # Only keep full-length chunks
            if len(chunk) == context_len:
                all_input_ids.append(chunk)

    return {"input_ids": all_input_ids}


def setup_dataloaders(
    dataset_name: str = "roneneldan/TinyStories",
    train_samples: int = 50000,
    val_samples: int = 2000,
    context_len: int = 256,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, CharacterLevelTokenizer]:
    """
    Set up DataLoaders for training.

    Args:
        dataset_name: HuggingFace dataset name
        train_samples: Number of training samples to use
        val_samples: Number of validation samples to use
        context_len: Sequence length (number of tokens)
        batch_size: Batch size for DataLoaders
        num_workers: Number of data loading workers

    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    print(f"Loading {dataset_name} dataset...")
    dataset = load_dataset(dataset_name)

    # Select subsets
    # If train_samples/val_samples is None, use full dataset
    train_size = len(dataset["train"]) if train_samples is None else min(train_samples, len(dataset["train"]))
    val_size = len(dataset["validation"]) if val_samples is None else min(val_samples, len(dataset["validation"]))

    print(f"Using {train_size:,} training samples and {val_size:,} validation samples")

    train_subset = dataset["train"].select(range(train_size))
    val_subset = dataset["validation"].select(range(val_size))

    dataset_splits = {
        "train": train_subset,
        "validation": val_subset,
        "test": val_subset,  # Use validation as test too
    }

    # Initialize tokenizer
    print("Initializing CharacterLevelTokenizer...")
    tokenizer = CharacterLevelTokenizer()

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"PAD token ID: {tokenizer.pad_token_id}")
    print(f"UNK token ID: {tokenizer.convert_tokens_to_ids([tokenizer.unk_token])[0]}")

    print(f"\nProcessing datasets with context_len={context_len}...")

    # Process each split
    train_dataset = dataset_splits["train"].map(
        lambda x: tokenize_and_chunk(tokenizer, x, context_len=context_len),
        batched=True,
        remove_columns=dataset_splits["train"].column_names,
        desc="Tokenizing train set",
    )

    val_dataset = dataset_splits["validation"].map(
        lambda x: tokenize_and_chunk(tokenizer, x, context_len=context_len),
        batched=True,
        remove_columns=dataset_splits["validation"].column_names,
        desc="Tokenizing validation set",
    )

    test_dataset = dataset_splits["test"].map(
        lambda x: tokenize_and_chunk(tokenizer, x, context_len=context_len),
        batched=True,
        remove_columns=dataset_splits["test"].column_names,
        desc="Tokenizing test set",
    )

    print(f"\n{'='*50}")
    print("Dataset Statistics:")
    print(f"{'='*50}")
    print(f"Train sequences:      {len(train_dataset):,}")
    print(f"Validation sequences: {len(val_dataset):,}")
    print(f"Test sequences:       {len(test_dataset):,}")
    print(f"Sequence length:      {context_len}")
    print(f"Vocabulary size:      {tokenizer.vocab_size}")
    print(f"{'='*50}\n")

    # Set format to PyTorch tensors
    train_dataset.set_format(type="torch", columns=["input_ids"])
    val_dataset.set_format(type="torch", columns=["input_ids"])
    test_dataset.set_format(type="torch", columns=["input_ids"])

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, tokenizer
