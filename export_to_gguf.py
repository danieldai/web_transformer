"""
Export fine-tuned Qwen model to GGUF format for LM Studio deployment.
GGUF is the format used by llama.cpp and LM Studio for optimal CPU/GPU inference.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Check dependencies
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError as e:
    print("\n" + "="*60)
    print("ERROR: Missing dependencies")
    print("="*60)
    print(f"\n{e}\n")
    print("Please activate your virtual environment and install dependencies:")
    print("\n  source venv/bin/activate")
    print("  pip install -r requirements.txt  # or requirements-linux.txt\n")
    print("Then run this script again.")
    print("="*60 + "\n")
    sys.exit(1)

# Configuration
MODEL_PATH = "./qwen_classifier/final"  # Path to fine-tuned model
OUTPUT_DIR = "./qwen_classifier_gguf"  # GGUF export directory
QUANTIZATION_TYPES = ["f16", "q4_k_m", "q5_k_m", "q8_0"]  # Available quantization types
DEFAULT_QUANT = "q4_k_m"  # Default quantization (good balance of size/quality)

def check_llama_cpp():
    """Check if llama.cpp is available."""
    print("=== Checking llama.cpp Installation ===\n")

    # Check if convert script exists (llama.cpp)
    llama_cpp_dirs = [
        Path.home() / "llama.cpp",
        Path("/opt/llama.cpp"),
        Path("/usr/local/llama.cpp"),
    ]

    for llama_dir in llama_cpp_dirs:
        # Try newer convert script first (llama.cpp >= v3)
        convert_script_new = llama_dir / "convert_hf_to_gguf.py"
        convert_script_old = llama_dir / "convert.py"

        # Try newer quantize binary name first
        llama_quantize_bin = llama_dir / "llama-quantize"
        quantize_bin = llama_dir / "quantize"
        llama_quantize_build = llama_dir / "build" / "bin" / "llama-quantize"
        quantize_build = llama_dir / "build" / "bin" / "quantize"

        # Check for either new or old convert script
        convert_script = None
        if convert_script_new.exists():
            convert_script = convert_script_new
        elif convert_script_old.exists():
            convert_script = convert_script_old

        # Check for quantize binary in various locations
        quant_bin = None
        for qbin in [llama_quantize_bin, llama_quantize_build, quantize_bin, quantize_build]:
            if qbin.exists():
                quant_bin = qbin
                break

        if convert_script and quant_bin:
            print(f"✓ Found llama.cpp at {llama_dir}")
            print(f"  Convert script: {convert_script.name}")
            print(f"  Quantize binary: {quant_bin.name}")

            # Check and install llama.cpp Python dependencies
            requirements_file = llama_dir / "requirements.txt"
            if requirements_file.exists():
                print(f"\n  Installing llama.cpp Python dependencies...")
                try:
                    # Try to install requirements (will skip if already installed)
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_file)],
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    if result.returncode == 0:
                        print(f"  ✓ Python dependencies installed")
                    else:
                        print(f"  ⚠ Warning: Some dependencies may not have installed")
                        print(f"     Run manually if needed: pip install -r {requirements_file}")
                except Exception as e:
                    print(f"  ⚠ Warning: Could not install dependencies automatically")
                    print(f"     Please run: pip install -r {requirements_file}")

            return {"path": llama_dir, "convert": convert_script, "quantize": quant_bin}

    print("⚠ llama.cpp not found!")
    print("\nTo export to GGUF format, you need llama.cpp:")
    print("\n1. Clone llama.cpp:")
    print("   git clone https://github.com/ggerganov/llama.cpp.git ~/llama.cpp")
    print("\n2. Build llama.cpp (choose one):")
    print("   cd ~/llama.cpp")
    print("   # Option A: CMake (recommended)")
    print("   cmake -B build && cmake --build build --config Release")
    print("   # Option B: Make (simpler)")
    print("   make")
    print("\n3. Install Python dependencies:")
    print("   pip install -r requirements.txt")
    print("\nThen run this script again.")
    return None

def convert_to_gguf_fp16(model_path: str, output_dir: str, llama_cpp_info: dict):
    """
    Convert PyTorch model to GGUF FP16 format.

    Args:
        model_path: Path to fine-tuned model
        output_dir: Directory to save GGUF model
        llama_cpp_info: Dictionary with llama.cpp paths (convert script, quantize binary)
    """
    print("\n=== Converting to GGUF (FP16) ===\n")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Output file
    output_file = output_path / "qwen-classifier-f16.gguf"

    # Use llama.cpp convert script
    convert_script = llama_cpp_info["convert"]

    print(f"Converting model from {model_path}...")
    print("This may take 2-5 minutes...\n")

    try:
        # Run conversion
        cmd = [
            sys.executable,
            str(convert_script),
            model_path,
            "--outfile", str(output_file),
            "--outtype", "f16",
        ]

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"✓ FP16 GGUF model created: {output_file}")
            print(f"  Model size: {size_mb:.2f} MB")
            return output_file
        else:
            print(f"⚠ Conversion completed but file not found: {output_file}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"⚠ Error during conversion: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None
    except Exception as e:
        print(f"⚠ Unexpected error: {e}")
        return None

def quantize_gguf(fp16_file: Path, output_dir: str, quant_type: str, llama_cpp_info: dict):
    """
    Quantize GGUF FP16 model to smaller quantized format.

    Args:
        fp16_file: Path to FP16 GGUF model
        output_dir: Directory to save quantized model
        quant_type: Quantization type (q4_k_m, q5_k_m, q8_0, etc.)
        llama_cpp_info: Dictionary with llama.cpp paths (convert script, quantize binary)
    """
    print(f"\n=== Quantizing to {quant_type.upper()} ===\n")

    output_path = Path(output_dir)
    output_file = output_path / f"qwen-classifier-{quant_type}.gguf"

    # Use llama.cpp quantize binary
    quantize_bin = llama_cpp_info["quantize"]

    print(f"Quantizing model to {quant_type}...")
    print("This may take 1-3 minutes...\n")

    try:
        # Run quantization
        cmd = [
            str(quantize_bin),
            str(fp16_file),
            str(output_file),
            quant_type,
        ]

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            original_size_mb = fp16_file.stat().st_size / (1024 * 1024)
            compression = (1 - size_mb / original_size_mb) * 100

            print(f"✓ Quantized GGUF model created: {output_file}")
            print(f"  Model size: {size_mb:.2f} MB")
            print(f"  Compression: {compression:.1f}%")
            return output_file
        else:
            print(f"⚠ Quantization completed but file not found: {output_file}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"⚠ Error during quantization: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None
    except Exception as e:
        print(f"⚠ Unexpected error: {e}")
        return None

def export_to_gguf(model_path: str, output_dir: str, quantize: bool = True, quant_types: list = None):
    """
    Export PyTorch model to GGUF format with optional quantization.

    Args:
        model_path: Path to fine-tuned model
        output_dir: Directory to save GGUF models
        quantize: Whether to create quantized versions
        quant_types: List of quantization types to create
    """
    print("=== Exporting Model to GGUF Format ===\n")
    print("GGUF format is optimized for llama.cpp and LM Studio\n")

    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using: python train.py")
        return

    # Check llama.cpp
    llama_cpp_info = check_llama_cpp()
    if llama_cpp_info is None:
        return

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Step 1: Convert to FP16 GGUF
    fp16_file = convert_to_gguf_fp16(model_path, output_dir, llama_cpp_info)

    if fp16_file is None:
        print("\n⚠ Failed to convert model to GGUF format")
        return

    # Step 2: Create quantized versions
    if quantize:
        if quant_types is None:
            quant_types = [DEFAULT_QUANT]

        print(f"\nCreating quantized versions: {', '.join(quant_types)}")

        quantized_files = []
        for quant_type in quant_types:
            quant_file = quantize_gguf(fp16_file, output_dir, quant_type, llama_cpp_info)
            if quant_file:
                quantized_files.append(quant_file)

        if quantized_files:
            print("\n" + "="*60)
            print("Export Complete!")
            print("="*60)
            print(f"\nGGUF models saved to: {output_dir}")
            print("\nAvailable models:")
            print(f"  - FP16 (highest quality): {fp16_file.name}")
            for qf in quantized_files:
                size_mb = qf.stat().st_size / (1024 * 1024)
                print(f"  - {qf.name.split('-')[-1].replace('.gguf', '').upper()}: {qf.name} ({size_mb:.0f}MB)")

            print("\n" + "="*60)
            print("Using with LM Studio (macOS M1)")
            print("="*60)
            print("\n1. Open LM Studio")
            print("2. Click 'Import Model'")
            print(f"3. Select: {output_dir}/qwen-classifier-{DEFAULT_QUANT}.gguf")
            print("4. Start chatting with your fine-tuned classifier!")

            print("\n" + "="*60)
            print("Recommended Models by Use Case")
            print("="*60)
            print("  - Best quality: qwen-classifier-f16.gguf (largest)")
            print("  - Balanced: qwen-classifier-q4_k_m.gguf (recommended)")
            print("  - Fast: qwen-classifier-q4_0.gguf (smallest)")

    else:
        print("\n" + "="*60)
        print("Export Complete!")
        print("="*60)
        print(f"\nGGUF model: {fp16_file}")
        print("\nFor LM Studio, use the FP16 version or create quantized versions.")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export Qwen model to GGUF format for LM Studio")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to fine-tuned model")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory for GGUF models")
    parser.add_argument("--no-quantize", action="store_true", help="Skip quantization (FP16 only)")
    parser.add_argument("--quant-types", nargs="+", default=["q4_k_m"],
                       choices=QUANTIZATION_TYPES,
                       help="Quantization types to create")

    args = parser.parse_args()

    # Export to GGUF
    export_to_gguf(
        model_path=args.model_path,
        output_dir=args.output_dir,
        quantize=not args.no_quantize,
        quant_types=args.quant_types,
    )

if __name__ == "__main__":
    main()
