"""
Test fine-tuned Qwen model on the test set.
Supports both PyTorch and ONNX models.
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm

# Configuration
PYTORCH_MODEL_PATH = "./qwen_classifier/final"
ONNX_MODEL_PATH = "./qwen_classifier_onnx_quantized"
TEST_FILE = "training_data/test.jsonl"
CATEGORIES = ["HR", "IT", "Sales", "Finance", "Operations", "Legal", "Marketing", "Customer Support"]

def load_test_data(filepath: str) -> list:
    """Load test data from JSONL file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def extract_question_and_label(item: dict) -> tuple:
    """Extract question and label from test item."""
    # Find user message (question)
    user_msg = next((msg for msg in item["messages"] if msg["role"] == "user"), None)
    # Find assistant message (label)
    asst_msg = next((msg for msg in item["messages"] if msg["role"] == "assistant"), None)

    if user_msg and asst_msg:
        # Extract question from instruction
        question = user_msg["content"].split("Question: ")[-1].split("\nCategory:")[0].strip()
        label = asst_msg["content"].strip()
        return question, label
    return None, None

def format_prompt(question: str) -> str:
    """Format question as model prompt."""
    instruction = f"""Classify the following question into one of these categories: {', '.join(CATEGORIES)}

Question: {question}
Category:"""
    return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

def predict_pytorch(model, tokenizer, question: str, device) -> str:
    """Make prediction using PyTorch model."""
    prompt = format_prompt(question)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Extract first word as prediction
    prediction = response.strip().split()[0] if response.strip() else "Unknown"
    return prediction

def predict_onnx(model, tokenizer, question: str) -> str:
    """Make prediction using ONNX model."""
    prompt = format_prompt(question)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        temperature=0.1,
        do_sample=False,
    )

    # Decode
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Extract first word as prediction
    prediction = response.strip().split()[0] if response.strip() else "Unknown"
    return prediction

def evaluate_model(model, tokenizer, test_data: list, model_type: str = "pytorch", device=None):
    """Evaluate model on test set."""
    print(f"\n=== Evaluating {model_type.upper()} Model ===\n")

    predictions = []
    true_labels = []

    print("Making predictions...")
    for item in tqdm(test_data):
        question, label = extract_question_and_label(item)
        if question and label:
            if model_type == "pytorch":
                pred = predict_pytorch(model, tokenizer, question, device)
            else:
                pred = predict_onnx(model, tokenizer, question)

            predictions.append(pred)
            true_labels.append(label)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)

    print(f"\n{'='*60}")
    print(f"Results - {model_type.upper()} Model")
    print(f"{'='*60}")
    print(f"\nAccuracy: {accuracy:.2%} ({sum(p == t for p, t in zip(predictions, true_labels))}/{len(predictions)})")

    # Classification report
    print("\n--- Classification Report ---")
    print(classification_report(true_labels, predictions, zero_division=0))

    # Confusion matrix
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(true_labels, predictions, labels=CATEGORIES)
    print("\nTrue \\ Pred", end="")
    for cat in CATEGORIES:
        print(f"\t{cat[:4]}", end="")
    print()
    for i, cat in enumerate(CATEGORIES):
        print(f"{cat[:12]:<12}", end="")
        for j in range(len(CATEGORIES)):
            print(f"\t{cm[i][j]}", end="")
        print()

    # Show some examples
    print("\n--- Sample Predictions ---")
    for i in range(min(5, len(predictions))):
        question, _ = extract_question_and_label(test_data[i])
        status = "✓" if predictions[i] == true_labels[i] else "✗"
        print(f"{status} Q: {question}")
        print(f"  Pred: {predictions[i]} | True: {true_labels[i]}\n")

    return accuracy, predictions, true_labels

def interactive_test(model, tokenizer, model_type: str = "pytorch", device=None):
    """Interactive testing mode."""
    print("\n=== Interactive Testing Mode ===")
    print("Enter questions to classify (or 'quit' to exit)\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break

        if not question:
            continue

        if model_type == "pytorch":
            prediction = predict_pytorch(model, tokenizer, question, device)
        else:
            prediction = predict_onnx(model, tokenizer, question)

        print(f"→ Category: {prediction}\n")

def main():
    # Check if test file exists
    if not Path(TEST_FILE).exists():
        print(f"Error: Test file not found at {TEST_FILE}")
        print("Please run 'python prepare_data.py' first")
        return

    # Load test data
    test_data = load_test_data(TEST_FILE)
    print(f"Loaded {len(test_data)} test examples")

    # Test PyTorch model
    if Path(PYTORCH_MODEL_PATH).exists():
        print(f"\nLoading PyTorch model from {PYTORCH_MODEL_PATH}...")

        # Check for MPS
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device")
        else:
            device = torch.device("cpu")
            print("Using CPU device")

        tokenizer = AutoTokenizer.from_pretrained(PYTORCH_MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            PYTORCH_MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(device)

        pytorch_acc, _, _ = evaluate_model(model, tokenizer, test_data, "pytorch", device)

        # Interactive mode
        choice = input("\nTest interactively? (y/n): ").strip().lower()
        if choice == 'y':
            interactive_test(model, tokenizer, "pytorch", device)

    else:
        print(f"PyTorch model not found at {PYTORCH_MODEL_PATH}")

    # Test ONNX model
    if Path(ONNX_MODEL_PATH).exists():
        try:
            from optimum.onnxruntime import ORTModelForCausalLM

            print(f"\nLoading ONNX model from {ONNX_MODEL_PATH}...")
            tokenizer = AutoTokenizer.from_pretrained(ONNX_MODEL_PATH, trust_remote_code=True)
            model = ORTModelForCausalLM.from_pretrained(ONNX_MODEL_PATH)

            onnx_acc, _, _ = evaluate_model(model, tokenizer, test_data, "onnx")

            # Interactive mode
            choice = input("\nTest ONNX model interactively? (y/n): ").strip().lower()
            if choice == 'y':
                interactive_test(model, tokenizer, "onnx")

        except Exception as e:
            print(f"\nCouldn't load ONNX model: {e}")
    else:
        print(f"\nONNX model not found at {ONNX_MODEL_PATH}")
        print("Run 'python export_to_onnx.py' to create it")

    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)

if __name__ == "__main__":
    main()
