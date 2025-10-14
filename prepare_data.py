"""
Prepare the text classification dataset for fine-tuning.
Converts data.py format to instruction-tuning JSONL format and splits into train/val/test.
"""

import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from data import TEST_DATASET

# Configuration
CATEGORIES = ["HR", "IT", "Sales", "Finance", "Operations", "Legal", "Marketing", "Customer Support"]
OUTPUT_DIR = Path("./training_data")
SEED = 42

# Set random seed for reproducibility
random.seed(SEED)

def format_instruction(question: str, category: str = None) -> dict:
    """
    Format a single example as an instruction-following conversation.

    For training, includes both instruction and response.
    For inference, only includes instruction.
    """
    instruction = f"""Classify the following question into one of these categories: {', '.join(CATEGORIES)}

Question: {question}
Category:"""

    if category is not None:
        # Training format with response
        return {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": category}
            ]
        }
    else:
        # Inference format
        return {
            "messages": [
                {"role": "user", "content": instruction}
            ]
        }

def augment_data(dataset: list) -> list:
    """
    Simple data augmentation by adding variations.
    This helps with the small dataset size.
    """
    augmented = dataset.copy()

    # Add paraphrased versions (you can expand this)
    augmentation_templates = [
        "I have a question: {question}",
        "Can you help with this: {question}",
        "Need assistance: {question}",
    ]

    for item in dataset:
        # Add 1-2 augmented versions per original example
        for template in random.sample(augmentation_templates, k=min(2, len(augmentation_templates))):
            augmented.append({
                "question": template.format(question=item["question"]),
                "expected": item["expected"]
            })

    return augmented

def save_jsonl(data: list, filepath: Path):
    """Save data in JSONL format."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    print("=== Preparing Text Classification Dataset ===\n")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load and optionally augment data
    print(f"Original dataset size: {len(TEST_DATASET)} examples")

    # Uncomment to enable augmentation (recommended for small datasets)
    augmented_dataset = augment_data(TEST_DATASET)
    print(f"After augmentation: {len(augmented_dataset)} examples")
    dataset = augmented_dataset

    # If you want to use original data only:
    # dataset = TEST_DATASET

    # Split into train/val/test (70/15/15)
    train_data, temp_data = train_test_split(
        dataset, test_size=0.3, random_state=SEED, stratify=[d["expected"] for d in dataset]
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=SEED, stratify=[d["expected"] for d in temp_data]
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Val:   {len(val_data)} examples")
    print(f"  Test:  {len(test_data)} examples")

    # Format as instruction-tuning data
    train_formatted = [format_instruction(d["question"], d["expected"]) for d in train_data]
    val_formatted = [format_instruction(d["question"], d["expected"]) for d in val_data]
    test_formatted = [format_instruction(d["question"], d["expected"]) for d in test_data]

    # Save to JSONL files
    save_jsonl(train_formatted, OUTPUT_DIR / "train.jsonl")
    save_jsonl(val_formatted, OUTPUT_DIR / "val.jsonl")
    save_jsonl(test_formatted, OUTPUT_DIR / "test.jsonl")

    print(f"\n✓ Saved training data to {OUTPUT_DIR}/")
    print(f"  - train.jsonl")
    print(f"  - val.jsonl")
    print(f"  - test.jsonl")

    # Print example
    print("\n=== Example Training Instance ===")
    print(json.dumps(train_formatted[0], indent=2, ensure_ascii=False))

    # Print category distribution
    print("\n=== Category Distribution (Train) ===")
    category_counts = {}
    for item in train_data:
        cat = item["expected"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count} examples")

if __name__ == "__main__":
    main()
