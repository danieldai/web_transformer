# Linux Setup Guide (RTX 4090)

Train Qwen text classifier on Linux with NVIDIA RTX 4090 GPU, export to GGUF for LM Studio on macOS M1.

## Quick Start

```bash
bash train_pipeline_linux.sh
```

This automated script will:
1. Check GPU and CUDA installation
2. Install dependencies
3. Train the model with FP16 optimization
4. Test accuracy
5. Export to GGUF format for LM Studio
6. Optionally export to ONNX for browser deployment

## Prerequisites

- Linux (Ubuntu 20.04+ recommended)
- NVIDIA RTX 4090 (or other CUDA-compatible GPU)
- NVIDIA drivers installed (525+ recommended)
- CUDA 12.1+ installed
- Python 3.8+
- At least 16GB RAM
- ~10GB disk space for models

## Setup Instructions

### 1. Verify NVIDIA Drivers and CUDA

Check that your GPU is detected:

```bash
nvidia-smi
```

You should see your RTX 4090 listed with driver version and CUDA version.

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.1   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        ...         | Bus-Id        ...    | ...                 |
|   0  NVIDIA GeForce RTX 4090 | ...                  | ...                 |
+-----------------------------------------------------------------------------+
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch with CUDA Support

**IMPORTANT:** Install PyTorch with CUDA support FIRST:

```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (if you have older CUDA)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify CUDA is available:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 4090
```

### 4. Install Other Dependencies

```bash
pip install -r requirements-linux.txt
```

This installs:
- Transformers, datasets, accelerate
- ONNX Runtime with GPU support
- scikit-learn, pandas, numpy
- Other utilities

**Note:** Installation takes 5-10 minutes.

### 5. Verify Installation

```bash
python -c "import torch; import transformers; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ Transformers {transformers.__version__}'); print(f'✓ CUDA: {torch.cuda.is_available()}')"
```

### 6. Install GGUF Export Dependencies

GGUF format is required for LM Studio deployment on macOS M1. The pipeline script will automatically install llama.cpp if not found.

**Manual installation (optional):**

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git ~/llama.cpp

# Build llama.cpp
cd ~/llama.cpp
make -j$(nproc)

# Install Python dependencies
pip install -r requirements.txt
```

Verify the installation:

```bash
ls ~/llama.cpp/convert.py ~/llama.cpp/quantize
```

Both files should exist for GGUF export to work.

### 7. Install ONNX Export Dependencies (Optional)

ONNX export is only needed if you want to deploy your model to browsers using transformers.js. If you only need LM Studio deployment, you can skip this step.

Install optimum with ONNX runtime support:

```bash
# Install optimum with ONNX runtime support
pip install optimum[onnxruntime]

# For GPU-accelerated ONNX inference (recommended for RTX 4090)
pip install onnxruntime-gpu
```

Verify the installation:

```bash
python -c "from optimum.onnxruntime import ORTModelForCausalLM; print('✓ Optimum ONNX Runtime installed')"
```

Verify GPU support is available:

```bash
python -c "import onnxruntime as ort; print('Available providers:', ort.get_available_providers())"
```

You should see `['CUDAExecutionProvider', 'CPUExecutionProvider']` if GPU support is working.

This enables the `export_to_onnx.py` script for browser deployment.

## Training on RTX 4090

### Automated Pipeline (Recommended)

```bash
bash train_pipeline_linux.sh
```

This will:
- Verify GPU and CUDA
- Install all dependencies
- Train the model
- Test accuracy
- Export to GGUF for LM Studio
- Optionally export to ONNX for browser

### Manual Step-by-Step

```bash
# 1. Prepare dataset
python prepare_data.py

# 2. Train model (will auto-detect GPU)
python train.py

# 3. Test accuracy
python test_inference.py

# 4. Export to GGUF (for LM Studio on macOS M1)
python export_to_gguf.py

# 5. Export to ONNX (optional, for browser)
python export_to_onnx.py
```

### Performance Optimizations

The training script automatically optimizes for NVIDIA GPUs:

1. **FP16 Training:** Enabled automatically on CUDA (2x faster, less memory)
2. **Larger Batch Size:** You can increase batch size for RTX 4090
3. **Gradient Checkpointing:** For training larger models

### Recommended Training Settings for RTX 4090

Edit `train.py` config for better performance:

```python
@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"  # Use larger model

    # Training - optimized for RTX 4090
    num_epochs: int = 10
    batch_size: int = 8           # Increase from 4 (24GB VRAM allows this)
    learning_rate: float = 2e-5
    gradient_accumulation_steps: int = 2  # Reduce from 4 (effective batch = 16)

    # Will auto-enable FP16
    fp16: bool = False  # Set to True by script on CUDA
```

### Training Time Comparison

| Model Size | M1 Mac (MPS) | RTX 4090 (CUDA) |
|------------|--------------|------------------|
| 0.5B       | 20-30 min    | 5-10 min         |
| 1.5B       | 60-90 min    | 15-20 min        |
| 3B         | N/A (OOM)    | 30-40 min        |

### Memory Usage

RTX 4090 has 24GB VRAM. Approximate memory usage:

| Model | Batch Size | FP16 | Memory Usage |
|-------|------------|------|--------------|
| 0.5B  | 4          | Yes  | ~4GB         |
| 0.5B  | 8          | Yes  | ~6GB         |
| 1.5B  | 4          | Yes  | ~8GB         |
| 1.5B  | 8          | Yes  | ~12GB        |
| 3B    | 4          | Yes  | ~14GB        |
| 3B    | 8          | Yes  | ~20GB        |

### Monitor GPU Usage

While training, open another terminal and run:

```bash
watch -n 1 nvidia-smi
```

This shows real-time GPU utilization, memory usage, and temperature.

## Training Larger Models

With RTX 4090's 24GB VRAM, you can train larger models:

```python
# In train.py, update Config:
model_name: str = "Qwen/Qwen2.5-3B-Instruct"  # Larger model
batch_size: int = 4
gradient_accumulation_steps: int = 4
```

Larger models provide:
- Better accuracy (90-98%)
- Better generalization
- Better understanding of nuanced text

## Troubleshooting

### CUDA Out of Memory

If you see "CUDA out of memory" errors:

1. Reduce batch size:
```python
batch_size: int = 4  # or even 2
```

2. Increase gradient accumulation:
```python
gradient_accumulation_steps: int = 4  # or 8
```

3. Use gradient checkpointing (add to train.py):
```python
model.gradient_checkpointing_enable()
```

### CUDA Not Available

If `torch.cuda.is_available()` returns False:

1. Check NVIDIA drivers: `nvidia-smi`
2. Reinstall PyTorch with CUDA: `pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu121`
3. Check CUDA installation: `nvcc --version`
4. Verify compatibility: PyTorch CUDA version must match system CUDA

### Slow Training

If training is slower than expected:

1. Verify GPU is being used: Check "Device: cuda" in training output
2. Monitor GPU usage: `nvidia-smi` should show ~95-100% GPU utilization
3. Check FP16 is enabled: Should see "FP16 enabled: True"
4. Disable other GPU processes: Close browsers, other apps using GPU

### Driver/CUDA Version Mismatch

If you see CUDA version warnings:

```bash
# Check your CUDA version
nvidia-smi  # Shows CUDA version in header

# Install matching PyTorch version
# For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Multi-GPU Training (Optional)

If you have multiple GPUs, you can use distributed training:

```python
# In train.py, add to TrainingArguments:
training_args = TrainingArguments(
    ...
    ddp_find_unused_parameters=False,
    dataloader_num_workers=4,
)
```

Then run with:
```bash
torchrun --nproc_per_node=2 train.py  # For 2 GPUs
```

## Advanced Optimization

### Enable Flash Attention 2 (Faster Training)

```bash
pip install flash-attn --no-build-isolation
```

Then in train.py:
```python
model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    trust_remote_code=True,
    torch_dtype=dtype,
    attn_implementation="flash_attention_2",  # Add this
)
```

### Enable BFloat16 (Alternative to FP16)

If your GPU supports it:
```python
# In Config:
fp16: bool = False
bf16: bool = True  # Better numerical stability
```

### Use Gradient Checkpointing (Save Memory)

For larger models:
```python
# After loading model in train.py:
model.gradient_checkpointing_enable()
```

This trades compute for memory (slower but uses less VRAM).

## Performance Tips

1. **Use FP16:** Enabled automatically on CUDA (2x faster)
2. **Increase batch size:** RTX 4090 can handle larger batches
3. **Monitor GPU:** Keep utilization >90%
4. **Close other apps:** Free up VRAM and system RAM
5. **Use SSD:** Faster data loading
6. **Pin memory:** Already enabled in training script

## Comparison: M1 Mac vs RTX 4090

| Feature | M1 Mac | RTX 4090 |
|---------|--------|----------|
| Memory | 8-16GB unified | 24GB VRAM |
| FP16 Support | Limited | Excellent |
| Training Speed | 1x | 3-5x faster |
| Max Model Size | 1.5B | 7B+ |
| Power Usage | 15-30W | 300-450W |
| Cost | ~$1500+ | ~$1600 |

## Transferring Model to macOS M1

After training on Linux, transfer your GGUF model to macOS for use in LM Studio:

### Method 1: SCP (Secure Copy)

```bash
# On Linux (after training)
cd /path/to/web_transformer
tar -czf qwen_classifier_gguf.tar.gz qwen_classifier_gguf/

# Transfer to Mac
scp qwen_classifier_gguf.tar.gz user@mac-hostname:/path/to/destination/

# On macOS
tar -xzf qwen_classifier_gguf.tar.gz
```

### Method 2: Cloud Storage

```bash
# On Linux
tar -czf qwen_classifier_gguf.tar.gz qwen_classifier_gguf/

# Upload to cloud storage (Dropbox, Google Drive, etc.)
# Download on macOS and extract
```

### Method 3: USB Drive

```bash
# Copy to USB drive
cp -r qwen_classifier_gguf /media/usb-drive/

# Transfer USB drive to Mac
# Copy from USB to Mac
```

### Using Model in LM Studio

1. Open LM Studio on macOS
2. Click "Import Model" or drag-and-drop
3. Select: `qwen_classifier_gguf/qwen-classifier-q4_k_m.gguf`
4. Start using your fine-tuned classifier!

**Recommended quantization for M1:**
- **Q4_K_M** (best balance): ~150-200MB, 50-100 tokens/sec
- **Q5_K_M** (higher quality): ~200-250MB, 40-80 tokens/sec
- **Q8_0** (near-lossless): ~300-350MB, 30-50 tokens/sec

## Next Steps

1. **Train with larger model:** Try Qwen2.5-1.5B or 3B
2. **Optimize hyperparameters:** Experiment with learning rate, batch size
3. **Use more data:** Collect more training examples
4. **Create multiple quantizations:** `python export_to_gguf.py --quant-types q4_k_m q5_k_m q8_0`
5. **Deploy with LM Studio API:** Enable server mode for API access

## Resources

- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [RTX 4090 Specs](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)

## Support

For issues:
1. Check GPU is detected: `nvidia-smi`
2. Verify CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check training logs in `qwen_classifier/`
4. Monitor GPU usage during training
