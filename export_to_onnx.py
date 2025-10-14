"""
Export fine-tuned Qwen model to ONNX format for browser deployment with transformers.js.
Includes quantization for smaller file size and faster inference.
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
import shutil

# Configuration
MODEL_PATH = "./qwen_classifier/final"  # Path to fine-tuned model
OUTPUT_DIR = "./qwen_classifier_onnx"  # ONNX export directory
QUANTIZE = True  # Apply INT8 quantization for browser
TEST_INFERENCE = True  # Test ONNX model after export

def export_to_onnx(model_path: str, output_dir: str, quantize: bool = True):
    """
    Export PyTorch model to ONNX format.

    Args:
        model_path: Path to fine-tuned model
        output_dir: Directory to save ONNX model
        quantize: Whether to apply INT8 quantization
    """
    print("=== Exporting Model to ONNX ===\n")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    # Export to ONNX
    print(f"\nExporting model from {model_path}...")
    print("This may take a few minutes...\n")

    try:
        # Use optimum to export
        ort_model = ORTModelForCausalLM.from_pretrained(
            model_path,
            export=True,
            trust_remote_code=True,
        )

        # Save ONNX model
        ort_model.save_pretrained(output_dir)
        print(f"✓ ONNX model exported to {output_dir}")

        # Get model size
        onnx_files = list(Path(output_dir).glob("*.onnx"))
        if onnx_files:
            size_mb = sum(f.stat().st_size for f in onnx_files) / (1024 * 1024)
            print(f"  Model size: {size_mb:.2f} MB")

        # Quantize if requested
        if quantize:
            print("\n--- Applying INT8 Quantization ---")
            quantized_dir = output_dir + "_quantized"
            Path(quantized_dir).mkdir(exist_ok=True, parents=True)

            # Dynamic quantization (faster, smaller)
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
            quantizer = ORTQuantizer.from_pretrained(output_dir)

            print("Quantizing model...")
            quantizer.quantize(
                save_dir=quantized_dir,
                quantization_config=qconfig,
            )

            # Copy tokenizer to quantized directory
            for file in Path(output_dir).glob("*.json"):
                shutil.copy(file, quantized_dir)
            for file in Path(output_dir).glob("*.txt"):
                shutil.copy(file, quantized_dir)

            # Get quantized size
            quantized_files = list(Path(quantized_dir).glob("*.onnx"))
            if quantized_files:
                quant_size_mb = sum(f.stat().st_size for f in quantized_files) / (1024 * 1024)
                print(f"✓ Quantized model saved to {quantized_dir}")
                print(f"  Model size: {quant_size_mb:.2f} MB")
                print(f"  Compression: {(1 - quant_size_mb/size_mb)*100:.1f}%")

            return quantized_dir

        return output_dir

    except Exception as e:
        print(f"\n⚠ Error during ONNX export: {e}")
        print("\nTrying alternative export method...")

        # Alternative: Manual ONNX export
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

        # Create dummy input
        dummy_input = tokenizer("Test input", return_tensors="pt")

        # Export
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            f"{output_dir}/model.onnx",
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=14,
        )

        print(f"✓ Model exported to {output_dir}/model.onnx")
        return output_dir

def test_onnx_inference(model_dir: str):
    """Test ONNX model inference."""
    print("\n=== Testing ONNX Model ===\n")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    try:
        # Load ONNX model
        model = ORTModelForCausalLM.from_pretrained(model_dir)

        # Test input
        test_question = "How many vacation days do I have left?"
        categories = ["HR", "IT", "Sales", "Finance", "Operations", "Legal", "Marketing", "Customer Support"]

        instruction = f"""Classify the following question into one of these categories: {', '.join(categories)}

Question: {test_question}
Category:"""

        # Format with chat template
        formatted = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

        # Tokenize
        inputs = tokenizer(formatted, return_tensors="pt")

        # Generate
        print(f"Test question: '{test_question}'")
        print("Generating prediction...\n")

        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
        )

        # Decode
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        prediction = response.strip().split()[0] if response.strip() else "Unknown"

        print(f"Prediction: {prediction}")
        print(f"Expected: HR")

        print("\n✓ ONNX model inference successful!")

    except Exception as e:
        print(f"⚠ Error during ONNX inference test: {e}")
        print("You can still use the model, but test it with test_inference.py")

def main():
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first using: python train.py")
        return

    # Export to ONNX
    final_dir = export_to_onnx(MODEL_PATH, OUTPUT_DIR, quantize=QUANTIZE)

    # Test inference
    if TEST_INFERENCE:
        test_onnx_inference(final_dir)

    print("\n" + "="*60)
    print("Export complete!")
    print("="*60)
    print(f"\nONNX model location: {final_dir}")
    print("\nFor browser deployment with transformers.js:")
    print("  1. Upload the model to Hugging Face Hub, or")
    print("  2. Host the model files and use custom model loading")
    print("\nNext step: Test with test_inference.py or use browser_demo.html")

if __name__ == "__main__":
    main()
