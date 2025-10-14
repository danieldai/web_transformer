"""
Fine-tune Qwen2.5-0.5B for text classification on M1 Mac.
Full parameter fine-tuning optimized for Apple Silicon.
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Configuration
@dataclass
class Config:
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # Smallest Qwen model

    # Data
    train_file: str = "training_data/train.jsonl"
    val_file: str = "training_data/val.jsonl"
    max_length: int = 256  # Max sequence length

    # Training
    output_dir: str = "./qwen_classifier"
    num_epochs: int = 10
    batch_size: int = 4  # Small batch for M1
    learning_rate: float = 2e-5
    warmup_steps: int = 50
    weight_decay: float = 0.01

    # Optimization for M1
    fp16: bool = False  # M1 doesn't support fp16
    bf16: bool = False  # Use bf16 if available
    gradient_accumulation_steps: int = 4  # Effective batch size = 16

    # Evaluation
    eval_steps: int = 50
    save_steps: int = 50
    logging_steps: int = 10

    # Early stopping
    early_stopping_patience: int = 3

    seed: int = 42

config = Config()
tokenizer = None  # Will be initialized in main()

def load_jsonl(filepath: str) -> list:
    """Load JSONL file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def format_conversation(messages: list) -> str:
    """
    Format messages into a single text string.
    Qwen2.5-Instruct uses a specific chat format.
    """
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    return formatted

def prepare_dataset(filepath: str, tokenizer, max_length: int):
    """Prepare and tokenize dataset."""
    data = load_jsonl(filepath)

    texts = []
    for item in data:
        text = format_conversation(item["messages"])
        texts.append(text)

    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None,
    )

    # Create dataset
    class TextDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            # Return as lists, not tensors - data collator will handle conversion
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.encodings["input_ids"][idx],  # For causal LM
            }

    return TextDataset(encodings)

def data_collator_fn(features):
    """Custom data collator for padding sequences."""
    # Get max length in batch
    max_length = max(len(f["input_ids"]) for f in features)

    # Pad sequences
    batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    for f in features:
        input_ids = f["input_ids"]
        attention_mask = f["attention_mask"]
        labels = f["labels"]

        # Calculate padding needed
        padding_length = max_length - len(input_ids)

        # Pad (pad_token_id will be set in main)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        labels = labels + [-100] * padding_length  # -100 is ignored by loss

        batch["input_ids"].append(input_ids)
        batch["attention_mask"].append(attention_mask)
        batch["labels"].append(labels)

    # Convert to tensors
    return {
        "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
        "labels": torch.tensor(batch["labels"], dtype=torch.long),
    }

def compute_metrics(eval_pred):
    """Compute perplexity as a metric."""
    predictions, labels = eval_pred
    # Calculate loss/perplexity
    # Note: For classification accuracy, use test_inference.py
    return {}

def main():
    global tokenizer  # Make tokenizer accessible to data_collator_fn

    print("=== Qwen2.5 Fine-tuning for Text Classification ===\n")

    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Check for MPS (Metal Performance Shaders) on M1
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Apple Silicon MPS acceleration")
    else:
        device = torch.device("cpu")
        print("⚠ MPS not available, using CPU")

    print(f"Model: {config.model_name}")
    print(f"Device: {device}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",  # Important for causal LM
    )

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # M1 optimization
    )

    # Ensure model uses padding token
    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Model parameters: {model.num_parameters():,}")

    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = prepare_dataset(config.train_file, tokenizer, config.max_length)
    val_dataset = prepare_dataset(config.val_file, tokenizer, config.max_length)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,

        # Evaluation
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,

        # Optimization
        fp16=config.fp16,
        bf16=config.bf16,

        # Saving
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Other
        seed=config.seed,
        dataloader_num_workers=0,  # M1 optimization
        remove_unused_columns=False,
        report_to="none",  # Disable wandb/tensorboard
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator_fn,  # Use our custom collator
        tokenizer=tokenizer,
    )

    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")

    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(config.output_dir + "/final")
    tokenizer.save_pretrained(config.output_dir + "/final")

    print(f"\n✓ Training complete!")
    print(f"✓ Model saved to: {config.output_dir}/final")
    print(f"\nNext steps:")
    print(f"  1. Run 'python test_inference.py' to evaluate the model")
    print(f"  2. Run 'python export_to_onnx.py' to convert for browser deployment")

if __name__ == "__main__":
    main()
