# Training Workflow Summary

## Overview

This project now supports two optimized training pipelines:

1. **Linux RTX 4090**: Fast training with CUDA acceleration
2. **macOS M1**: Training with MPS acceleration

Both pipelines export to **GGUF format** for optimal inference in **LM Studio** on macOS M1.

## Quick Start Commands

### Linux RTX 4090 (Recommended for Training)

```bash
bash train_pipeline_linux.sh
```

**What it does:**
- ✓ Verifies CUDA and GPU
- ✓ Installs dependencies
- ✓ Trains model with FP16 (5-10 min for 0.5B model)
- ✓ Tests accuracy
- ✓ Exports to GGUF format
- ✓ Optional: Exports to ONNX

### macOS M1/M2/M3

```bash
bash train_pipeline_macos.sh
```

**What it does:**
- ✓ Verifies MPS availability
- ✓ Installs dependencies
- ✓ Trains model with MPS (20-40 min for 0.5B model)
- ✓ Tests accuracy
- ✓ Exports to GGUF format
- ✓ Optional: Exports to ONNX

## Recommended Workflow

### For Best Performance:

1. **Train on Linux RTX 4090** (3-5x faster)
   ```bash
   bash train_pipeline_linux.sh
   ```

2. **Transfer GGUF model to macOS**
   ```bash
   # On Linux
   tar -czf qwen_classifier_gguf.tar.gz qwen_classifier_gguf/
   scp qwen_classifier_gguf.tar.gz user@mac:/destination/

   # On macOS
   tar -xzf qwen_classifier_gguf.tar.gz
   ```

3. **Use in LM Studio on macOS M1**
   - Open LM Studio
   - Import Model → Select `qwen-classifier-q4_k_m.gguf`
   - Start chatting!

## Model Formats

### GGUF (for LM Studio - Recommended)

**Location:** `./qwen_classifier_gguf/`

**Files:**
- `qwen-classifier-f16.gguf` - Full precision (500MB)
- `qwen-classifier-q4_k_m.gguf` - **Recommended** (150-200MB)
- `qwen-classifier-q5_k_m.gguf` - Higher quality (200-250MB)
- `qwen-classifier-q8_0.gguf` - Near-lossless (300-350MB)

**Best for:**
- LM Studio on macOS M1
- Local inference
- API access via LM Studio server

### ONNX (for Browser - Optional)

**Location:** `./qwen_classifier_onnx_quantized/`

**Best for:**
- Web browsers
- transformers.js deployment
- Cross-platform web apps

### PyTorch (Native Format)

**Location:** `./qwen_classifier/final/`

**Best for:**
- Python applications
- Fine-tuning
- Research

## Performance Comparison

| Platform | Training Time (0.5B) | Inference Speed (tokens/sec) |
|----------|---------------------|------------------------------|
| Linux RTX 4090 | 5-10 min | 100-200 (CUDA) |
| macOS M1 | 20-40 min | 50-100 (Q4_K_M in LM Studio) |

## Manual Export Commands

If you need to export models separately:

```bash
# Export to GGUF (for LM Studio)
python export_to_gguf.py

# Export to ONNX (for browser)
python export_to_onnx.py

# Create multiple quantizations
python export_to_gguf.py --quant-types q4_k_m q5_k_m q8_0
```

## Files Created

### New Pipeline Scripts
- `train_pipeline_linux.sh` - Linux RTX 4090 pipeline
- `train_pipeline_macos.sh` - macOS M1 pipeline
- `export_to_gguf.py` - GGUF export script

### Updated Files
- `README.md` - Complete workflow documentation
- `LINUX_SETUP.md` - Linux setup with GGUF export
- `requirements.txt` - macOS dependencies
- `requirements-linux.txt` - Linux dependencies

### Generated Directories
- `qwen_classifier/` - PyTorch models
- `qwen_classifier_gguf/` - GGUF models (LM Studio)
- `qwen_classifier_onnx_quantized/` - ONNX models (Browser)

## Using Your Model in LM Studio

### Step 1: Import Model

1. Open LM Studio
2. Click "Import Model" or drag-and-drop
3. Navigate to `qwen_classifier_gguf/`
4. Select `qwen-classifier-q4_k_m.gguf` (recommended)

### Step 2: Chat with Your Model

**Example prompt:**
```
Classify the following question into one of these categories: HR, IT, Sales, Finance, Operations

Question: How do I reset my password?
Category:
```

**Expected response:**
```
IT
```

### Step 3: Enable API (Optional)

1. In LM Studio, go to "Developer" tab
2. Enable "Local Server"
3. Use API endpoint: `http://localhost:1234/v1/chat/completions`

## Troubleshooting

### CUDA not available (Linux)
```bash
# Verify GPU
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu121
```

### MPS not available (macOS)
```bash
# Check MPS
python -c "import torch; print(torch.backends.mps.is_available())"

# Update PyTorch
pip install --upgrade torch
```

### llama.cpp not found
The pipeline scripts will automatically install llama.cpp. If manual installation is needed:

```bash
# Clone and build
git clone https://github.com/ggerganov/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp
make  # or 'LLAMA_METAL=1 make' on macOS for Metal support
pip install -r requirements.txt
```

## Support

For issues or questions:
- Check [README.md](README.md) for detailed documentation
- Check [LINUX_SETUP.md](LINUX_SETUP.md) for Linux-specific setup
- Review training logs in `qwen_classifier/`

## Next Steps

1. **Improve dataset**: Add more training examples
2. **Try larger models**: Qwen2.5-1.5B or 3B
3. **Optimize prompts**: Experiment with different classification formats
4. **Deploy to production**: Use LM Studio API for applications
