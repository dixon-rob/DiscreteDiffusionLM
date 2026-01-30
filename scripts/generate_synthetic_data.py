#!/usr/bin/env python
"""
Synthetic training data generator using Claude API.

Takes paraloq/json_data_extraction (484 records) and generates 100 variants
per record, creating ~48,400 training examples.

Supports:
- Parallel async requests for speed
- Batch API for cost savings (50% cheaper)
- Haiku model for faster generation

Usage:
    export ANTHROPIC_API_KEY="your-key-here"

    # Fast parallel mode (default)
    python scripts/generate_synthetic_data.py --parallel 10

    # Batch mode (cheaper, slower turnaround)
    python scripts/generate_synthetic_data.py --batch

    # Test run
    python scripts/generate_synthetic_data.py --variants_per_record 10 --max_records 5 --parallel 5
"""

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
from anthropic import AsyncAnthropic
from datasets import Dataset, DatasetDict, load_dataset
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm

# Output constraints
MAX_TEXT_CHARS = 1024
MODEL = "claude-3-haiku-20240307"  # Cheapest option

# Rotating constraints for variety
DOC_STYLES = [
    "formal business letter",
    "casual email",
    "bullet-point notes",
    "narrative paragraph",
    "technical specification",
    "chat/message transcript",
    "form or survey response",
    "news article excerpt",
    "internal memo",
    "customer review",
]

NAMING_CONVENTIONS = [
    "camelCase",
    "snake_case",
    "PascalCase",
    "kebab-case",
]

TONES = [
    "professional and formal",
    "casual and conversational",
    "terse and minimal",
    "detailed and thorough",
]

# How many recent examples to show as negative examples
RECENT_EXAMPLES_WINDOW = 3
# Max banned words to track
MAX_BANNED_WORDS = 50


def extract_names_from_json(json_str: str) -> List[str]:
    """Extract likely names/identifiers from JSON values for banned words list."""
    names = []
    try:
        obj = json.loads(json_str)
        _extract_string_values(obj, names)
    except json.JSONDecodeError:
        pass
    filtered = []
    for name in names:
        words = name.split()
        for word in words:
            if len(word) >= 3 and word[0].isupper() and word.isalpha():
                filtered.append(word)
    return filtered[:10]


def _extract_string_values(obj: Any, results: List[str]) -> None:
    """Recursively extract string values from JSON."""
    if isinstance(obj, str) and len(obj) < 100:
        results.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _extract_string_values(v, results)
    elif isinstance(obj, list):
        for item in obj:
            _extract_string_values(item, results)


def count_fields(obj: Any) -> int:
    """Count total fields in a JSON object (recursively)."""
    if isinstance(obj, dict):
        count = len(obj)
        for v in obj.values():
            count += count_fields(v)
        return count
    elif isinstance(obj, list):
        return sum(count_fields(item) for item in obj)
    return 0


def trim_json_fields(obj: Any, target_fields: int) -> Any:
    """Randomly remove fields from a JSON object to reach target field count."""
    if not isinstance(obj, dict) or not obj:
        return obj

    current_count = count_fields(obj)
    if current_count <= target_fields:
        return obj

    result = obj.copy()

    while count_fields(result) > target_fields and len(result) > 1:
        key_to_remove = random.choice(list(result.keys()))
        del result[key_to_remove]

    for key in list(result.keys()):
        if isinstance(result[key], dict) and count_fields(result) > target_fields:
            result[key] = trim_json_fields(result[key], max(1, target_fields // 2))

    return result


def create_variant_prompt(
    example_item: str,
    topic: str,
    title: str,
    target_fields: int,
    doc_style: str,
    naming_convention: str,
    tone: str,
    recent_examples: List[Dict[str, str]],
    banned_words: List[str],
) -> str:
    """Create the prompt for generating a single variant."""

    negative_section = ""
    if recent_examples:
        negative_section = "\n\nAVOID generating anything similar to these recent examples:\n"
        for i, ex in enumerate(recent_examples, 1):
            negative_section += f"\n--- Recent Example {i} ---\n"
            negative_section += f"INSTRUCTION: {ex.get('instruction', '')[:100]}...\n"
            negative_section += f"TEXT: {ex.get('text', '')[:150]}...\n"
            negative_section += f"JSON: {ex.get('json', '')[:150]}...\n"

    banned_section = ""
    if banned_words:
        banned_section = f"\n\nDO NOT use these names/words (already used): {', '.join(banned_words[:30])}\n"

    return f"""You are generating synthetic training data for a JSON extraction model.

Here's an example of the kind of JSON structure:
```json
{example_item}
```

Topic: {topic}
Subject: {title}

CONSTRAINTS FOR THIS VARIANT:
- Target field count: {target_fields}
- Document style: {doc_style}
- JSON naming convention: {naming_convention}
- Tone: {tone}
{negative_section}{banned_section}
Generate ONE new training example following the constraints above.

1. INSTRUCTION: A {tone} task instruction asking to extract information into JSON.

2. TEXT: A {doc_style} (max {MAX_TEXT_CHARS} characters) containing information to extract.
   Use fresh, creative names and values not seen in recent examples.

3. JSON: The extracted JSON with approximately {target_fields} fields.
   Use {naming_convention} for all field names.

Respond exactly as:
INSTRUCTION: <instruction>
TEXT: <document, max {MAX_TEXT_CHARS} chars>
JSON: <valid JSON>"""


def parse_variant_response(response_text: str) -> Optional[Dict[str, str]]:
    """Parse Claude's response into instruction, text, and json components."""
    try:
        instruction_start = response_text.find("INSTRUCTION:") + len("INSTRUCTION:")
        text_start = response_text.find("TEXT:")
        json_start = response_text.find("JSON:")

        if any(x == -1 for x in [instruction_start, text_start, json_start]):
            return None

        instruction = response_text[instruction_start:text_start].strip()
        text = response_text[text_start + len("TEXT:"):json_start].strip()
        json_str = response_text[json_start + len("JSON:"):].strip()

        json.loads(json_str)

        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS]

        return {
            "instruction": instruction,
            "text": text,
            "json": json_str,
        }
    except (json.JSONDecodeError, ValueError):
        return None


# =============================================================================
# ASYNC PARALLEL GENERATION
# =============================================================================

async def generate_single_variant_async(
    client: AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    prompt: str,
    metadata: Dict[str, Any],
) -> Optional[Dict[str, str]]:
    """Generate a single variant asynchronously."""
    async with semaphore:
        try:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
            )

            response_text = response.content[0].text
            parsed = parse_variant_response(response_text)

            if parsed:
                parsed.update(metadata)
                return parsed
            return None

        except anthropic.RateLimitError:
            await asyncio.sleep(10)
            return None
        except Exception as e:
            return None


async def generate_variants_async(
    client: AsyncAnthropic,
    record: Dict[str, Any],
    num_variants: int,
    parallel: int,
    pbar: Optional[tqdm] = None,
) -> List[Dict[str, str]]:
    """Generate multiple variants for a single record using async parallelism."""

    semaphore = asyncio.Semaphore(parallel)

    # Parse the example JSON
    example_item_str = record["item"]
    try:
        example_item = json.loads(example_item_str)
    except json.JSONDecodeError:
        example_item = example_item_str

    topic = record.get("topic", "general")
    title = record.get("title", "Data Extraction")

    # Prepare all prompts and metadata upfront
    tasks = []
    # We can't use negative examples effectively in parallel, so we skip them for speed
    # Banned words also don't work well in parallel - trade-off for speed

    for i in range(num_variants):
        target_fields = random.randint(1, 5)
        doc_style = DOC_STYLES[i % len(DOC_STYLES)]
        naming_convention = NAMING_CONVENTIONS[i % len(NAMING_CONVENTIONS)]
        tone = TONES[i % len(TONES)]

        if isinstance(example_item, dict):
            trimmed_item = trim_json_fields(example_item, target_fields)
            trimmed_item_str = json.dumps(trimmed_item, indent=2)
        else:
            trimmed_item_str = example_item_str

        # No recent examples or banned words in parallel mode (trade-off for speed)
        prompt = create_variant_prompt(
            trimmed_item_str, topic, title, target_fields,
            doc_style, naming_convention, tone,
            recent_examples=[],
            banned_words=[],
        )

        metadata = {
            "topic": topic,
            "title": title,
            "target_fields": target_fields,
            "doc_style": doc_style,
            "naming_convention": naming_convention,
            "tone": tone,
        }

        tasks.append(generate_single_variant_async(client, semaphore, prompt, metadata))

    # Run all tasks with progress updates
    variants = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            variants.append(result)
        if pbar:
            pbar.update(1)

    return variants


# =============================================================================
# BATCH API GENERATION
# =============================================================================

def create_batch_requests(
    records,
    variants_per_record: int,
    output_path: Path,
) -> str:
    """Create a JSONL file of batch requests."""

    requests_path = output_path / "batch_requests.jsonl"

    with open(requests_path, "w") as f:
        request_id = 0
        for record_idx, record in enumerate(records):
            example_item_str = record["item"]
            try:
                example_item = json.loads(example_item_str)
            except json.JSONDecodeError:
                example_item = example_item_str

            topic = record.get("topic", "general")
            title = record.get("title", "Data Extraction")

            for i in range(variants_per_record):
                target_fields = random.randint(1, 5)
                doc_style = DOC_STYLES[i % len(DOC_STYLES)]
                naming_convention = NAMING_CONVENTIONS[i % len(NAMING_CONVENTIONS)]
                tone = TONES[i % len(TONES)]

                if isinstance(example_item, dict):
                    trimmed_item = trim_json_fields(example_item, target_fields)
                    trimmed_item_str = json.dumps(trimmed_item, indent=2)
                else:
                    trimmed_item_str = example_item_str

                prompt = create_variant_prompt(
                    trimmed_item_str, topic, title, target_fields,
                    doc_style, naming_convention, tone,
                    recent_examples=[],
                    banned_words=[],
                )

                request = {
                    "custom_id": f"req_{record_idx}_{i}",
                    "params": {
                        "model": MODEL,
                        "max_tokens": 2048,
                        "temperature": 0.9,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    "metadata": {
                        "topic": topic,
                        "title": title,
                        "target_fields": target_fields,
                        "doc_style": doc_style,
                        "naming_convention": naming_convention,
                        "tone": tone,
                    }
                }
                f.write(json.dumps(request) + "\n")
                request_id += 1

    print(f"Created {request_id} batch requests at {requests_path}")
    return str(requests_path)


def submit_batch(client: anthropic.Anthropic, requests_path: str) -> str:
    """Submit a batch job and return the batch ID."""

    with open(requests_path, "r") as f:
        requests = [json.loads(line) for line in f]

    # Create the batch
    batch = client.batches.create(
        requests=[
            {
                "custom_id": req["custom_id"],
                "params": req["params"],
            }
            for req in requests
        ]
    )

    print(f"Submitted batch: {batch.id}")
    print(f"Status: {batch.processing_status}")
    return batch.id


def poll_batch(
    client: anthropic.Anthropic,
    batch_id: str,
    output_path: Path,
) -> Optional[List[Dict]]:
    """Poll for batch completion and return results."""

    print(f"Polling batch {batch_id}...")

    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.processing_status

        print(f"  Status: {status}, "
              f"Completed: {batch.request_counts.succeeded}/{batch.request_counts.processing + batch.request_counts.succeeded}")

        if status == "ended":
            break

        time.sleep(30)  # Poll every 30 seconds

    # Load metadata from requests file
    requests_path = output_path / "batch_requests.jsonl"
    metadata_by_id = {}
    if requests_path.exists():
        with open(requests_path, "r") as f:
            for line in f:
                req = json.loads(line)
                metadata_by_id[req["custom_id"]] = req.get("metadata", {})

    # Get results
    results = []
    for result in client.batches.results(batch_id):
        if result.result.type == "succeeded":
            response_text = result.result.message.content[0].text
            parsed = parse_variant_response(response_text)
            if parsed:
                # Recover metadata from the requests file
                metadata = metadata_by_id.get(result.custom_id, {})
                parsed.update(metadata)
                results.append(parsed)

    return results


# =============================================================================
# MAIN GENERATION PIPELINE
# =============================================================================

async def generate_dataset_async(
    output_dir: str = "./data/synthetic_json_extraction",
    variants_per_record: int = 100,
    max_records: Optional[int] = None,
    resume_from: int = 0,
    parallel: int = 10,
) -> DatasetDict:
    """Main async generation pipeline."""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = AsyncAnthropic(api_key=api_key)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading paraloq/json_data_extraction...")
    source = load_dataset("paraloq/json_data_extraction")
    records = source["train"]

    if max_records:
        records = records.select(range(min(max_records, len(records))))

    total_records = len(records)

    # Check for existing checkpoint
    checkpoint_path = output_path / "checkpoint.json"
    jsonl_path = output_path / "generated.jsonl"
    all_variants = []

    if checkpoint_path.exists() and resume_from == 0:
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        resume_from = checkpoint.get("next_record_idx", 0)
        print(f"\nFound checkpoint - resuming from record {resume_from}")

    if resume_from > 0 and jsonl_path.exists():
        print(f"Loading existing data from {jsonl_path}...")
        with open(jsonl_path, "r") as f:
            all_variants = [json.loads(line) for line in f]
        print(f"  Loaded {len(all_variants)} existing variants")

    total_variants = (total_records - resume_from) * variants_per_record

    print(f"\nGeneration plan:")
    print(f"  Model: {MODEL}")
    print(f"  Source records: {total_records}")
    print(f"  Variants per record: {variants_per_record}")
    print(f"  Parallel requests: {parallel}")
    print(f"  Resume from record: {resume_from}")
    print(f"  Total variants to generate: {total_variants}")

    with tqdm(total=total_variants, desc="Generating variants") as pbar:
        for idx in range(resume_from, total_records):
            record = records[idx]

            variants = await generate_variants_async(
                client=client,
                record=record,
                num_variants=variants_per_record,
                parallel=parallel,
                pbar=pbar,
            )

            all_variants.extend(variants)

            # Append to JSONL
            with open(jsonl_path, "a") as f:
                for v in variants:
                    f.write(json.dumps(v, ensure_ascii=False) + "\n")

            # Save checkpoint after each record
            with open(checkpoint_path, "w") as f:
                json.dump({
                    "next_record_idx": idx + 1,
                    "total_records": total_records,
                    "variants_generated": len(all_variants),
                }, f)

            if (idx + 1) % 10 == 0:
                print(f"\n  Checkpoint: {idx + 1}/{total_records} records, "
                      f"{len(all_variants)} total variants")

    print(f"\nGeneration complete! {len(all_variants)} variants generated")

    # Remove checkpoint file since we're done
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Create train/validation split
    random.seed(42)
    random.shuffle(all_variants)

    split_idx = int(len(all_variants) * 0.9)
    train_data = all_variants[:split_idx]
    val_data = all_variants[split_idx:]

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
    })

    hf_path = output_path / "hf_dataset"
    print(f"\nSaving HuggingFace dataset to {hf_path}...")
    dataset_dict.save_to_disk(str(hf_path))

    stats = {
        "total_variants": len(all_variants),
        "train_size": len(train_data),
        "val_size": len(val_data),
        "source_records": total_records,
        "variants_per_record": variants_per_record,
        "model": MODEL,
    }
    with open(output_path / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset saved!")
    print(f"  Train: {len(train_data):,}")
    print(f"  Validation: {len(val_data):,}")

    return dataset_dict


def generate_dataset_batch(
    output_dir: str = "./data/synthetic_json_extraction",
    variants_per_record: int = 100,
    max_records: Optional[int] = None,
) -> None:
    """Generate using batch API (submit and poll)."""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading paraloq/json_data_extraction...")
    source = load_dataset("paraloq/json_data_extraction")
    records = source["train"]

    if max_records:
        records = records.select(range(min(max_records, len(records))))

    print(f"\nBatch generation plan:")
    print(f"  Model: {MODEL}")
    print(f"  Source records: {len(records)}")
    print(f"  Variants per record: {variants_per_record}")
    print(f"  Total requests: {len(records) * variants_per_record}")

    # Create batch request file
    requests_path = create_batch_requests(records, variants_per_record, output_path)

    # Submit batch
    batch_id = submit_batch(client, requests_path)

    # Save batch ID for later retrieval
    with open(output_path / "batch_id.txt", "w") as f:
        f.write(batch_id)

    print(f"\nBatch submitted! ID saved to {output_path / 'batch_id.txt'}")
    print(f"Run with --poll {batch_id} to check status and retrieve results")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic JSON extraction training data using Claude"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/synthetic_json_extraction",
        help="Output directory",
    )
    parser.add_argument(
        "--variants_per_record",
        type=int,
        default=100,
        help="Variants to generate per source record (default: 100)",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Limit source records (for testing)",
    )
    parser.add_argument(
        "--resume_from",
        type=int,
        default=0,
        help="Resume from this record index",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=10,
        help="Number of parallel requests (default: 10)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch API (50%% cheaper, slower turnaround)",
    )
    parser.add_argument(
        "--poll",
        type=str,
        default=None,
        help="Poll a batch job by ID and retrieve results",
    )

    args = parser.parse_args()

    if args.poll:
        # Poll existing batch
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=api_key)
        output_path = Path(args.output_dir)
        results = poll_batch(client, args.poll, output_path)
        if results:
            with open(output_path / "generated.jsonl", "w") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Saved {len(results)} results")
    elif args.batch:
        generate_dataset_batch(
            output_dir=args.output_dir,
            variants_per_record=args.variants_per_record,
            max_records=args.max_records,
        )
    else:
        asyncio.run(generate_dataset_async(
            output_dir=args.output_dir,
            variants_per_record=args.variants_per_record,
            max_records=args.max_records,
            resume_from=args.resume_from,
            parallel=args.parallel,
        ))


if __name__ == "__main__":
    main()
