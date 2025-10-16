# Qwen Text Classification Training Pipeline

Fine-tune Qwen2.5-0.5B for text classification with optimized pipelines for Linux RTX 4090 training and macOS M1 deployment.

## Overview

This pipeline allows you to:
1. Fine-tune a small Qwen LLM on your custom text classification dataset
2. Export the model to **GGUF format** for **LM Studio** (recommended for macOS M1)
3. Export the model to ONNX format for browser deployment with transformers.js (optional)

## 🚀 Quick Start

### Training on Linux RTX 4090 (Recommended)
```bash
bash train_pipeline_linux.sh
```

### Training on macOS M1/M2/M3
```bash
bash train_pipeline_macos.sh
```

### Use Trained Model in LM Studio (macOS M1)
1. Open LM Studio
2. Click "Import Model"
3. Select: `qwen_classifier_gguf/qwen-classifier-q4_k_m.gguf`
4. Start chatting with your fine-tuned classifier!

## Prerequisites

### macOS (M1/M2/M3)
- macOS with M1/M2/M3 chip
- Python 3.8+
- At least 8GB RAM
- ~5GB disk space for models

### Linux with NVIDIA GPU (RTX 4090)
- Linux (Ubuntu 20.04+)
- NVIDIA GPU with CUDA support (RTX 4090 recommended)
- NVIDIA drivers + CUDA 12.1+
- Python 3.8+
- At least 16GB RAM
- ~10GB disk space for models
- **See [LINUX_SETUP.md](LINUX_SETUP.md) for detailed Linux setup instructions**

## Setup

### 1. Install Dependencies

#### For macOS (M1/M2/M3):

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

**Note:** First installation may take 10-15 minutes as it downloads all dependencies.

#### For Linux with NVIDIA GPU:

See **[LINUX_SETUP.md](LINUX_SETUP.md)** for complete Linux setup instructions.

Quick start:
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support FIRST
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install other dependencies
pip install -r requirements-linux.txt
```

## Training Workflow

### Automated Pipeline (Recommended)

**For Linux RTX 4090:**
```bash
bash train_pipeline_linux.sh
```

**For macOS M1/M2/M3:**
```bash
bash train_pipeline_macos.sh
```

These scripts will automatically:
1. Check system requirements
2. Install dependencies
3. Prepare dataset
4. Train model
5. Test accuracy
6. Export to GGUF format for LM Studio
7. Optionally export to ONNX for browser deployment

### Manual Step-by-Step Workflow

#### Step 1: Prepare Dataset

Convert your dataset to training format and split into train/val/test sets:

```bash
python prepare_data.py
```

This will:
- Load data from `data.py`
- Augment the dataset (2-3x larger)
- Split into train/val/test (70/15/15)
- Save as JSONL files in `training_data/`

**Output:**
```
training_data/
├── train.jsonl
├── val.jsonl
└── test.jsonl
```

#### Step 2: Train Model

Fine-tune Qwen2.5-0.5B on your dataset:

```bash
python train.py
```

**Training details:**
- Model: Qwen2.5-0.5B-Instruct (~500MB)
- Full parameter fine-tuning (not LoRA)
- Auto-detects and uses best device: CUDA > MPS > CPU
- Training time: 5-10 min (RTX 4090), 20-40 min (M1 Mac)
- FP16 automatically enabled on CUDA for faster training
- Early stopping based on validation loss

**Output:**
```
qwen_classifier/
├── checkpoint-50/
├── checkpoint-100/
└── final/          # Best model
```

**Training tips:**
- Monitor the loss curves - training loss should decrease
- If you see overfitting (train loss << val loss), reduce epochs
- The model auto-saves checkpoints every 50 steps

#### Step 3: Test Model

Evaluate the trained model on test set:

```bash
python test_inference.py
```

This will:
- Load the fine-tuned model
- Run predictions on test set
- Show accuracy, classification report, confusion matrix
- Offer interactive testing mode

**Expected accuracy:** 85-95% (depending on dataset quality and size)

#### Step 4: Export to GGUF (for LM Studio)

Convert model for LM Studio deployment:

```bash
python export_to_gguf.py
```

This will:
- Convert PyTorch model to GGUF format
- Create quantized versions (Q4_K_M recommended)
- Optimize for llama.cpp/LM Studio

**Output:**
```
qwen_classifier_gguf/
├── qwen-classifier-f16.gguf      # Full precision (largest)
├── qwen-classifier-q4_k_m.gguf   # Recommended (balanced)
├── qwen-classifier-q5_k_m.gguf   # Higher quality
└── qwen-classifier-q8_0.gguf     # Near-lossless
```

**File sizes:**
- FP16: ~500MB (highest quality)
- Q4_K_M: ~150-200MB (recommended, best balance)
- Q5_K_M: ~200-250MB (higher quality)
- Q8_0: ~300-350MB (near-lossless)

**Note:** llama.cpp must be installed (handled automatically by pipeline scripts)

#### Step 5: Export to ONNX (Optional)

Convert model for browser deployment:

```bash
python export_to_onnx.py
```

This will:
- Export PyTorch model to ONNX format
- Apply INT8 quantization (~50-70% size reduction)
- Test ONNX inference
- Save quantized model

**Output:**
```
qwen_classifier_onnx_quantized/
├── model.onnx              # Quantized model
├── model_quantized.onnx    # Additional optimizations
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

**File sizes:**
- Original PyTorch: ~500MB
- ONNX (unquantized): ~500MB
- ONNX (quantized): ~150-200MB

## Deployment Options

### Option 1: LM Studio (Recommended for macOS M1)

**Best for:** Local inference, desktop applications, API access

1. **Download LM Studio:**
   - Visit: https://lmstudio.ai
   - Download and install for macOS

2. **Import your fine-tuned model:**
   ```
   Open LM Studio → Click "Import Model" → Select your GGUF file:
   ./qwen_classifier_gguf/qwen-classifier-q4_k_m.gguf
   ```

3. **Use your model:**
   - **Chat interface:** Start chatting immediately
   - **API mode:** Enable local server for API access
   - **Embeddings:** Generate embeddings for your text

4. **Example classification prompt:**
   ```
   Classify the following question into one of these categories: HR, IT, Sales, Finance

   Question: How do I reset my password?
   Category:
   ```

**Performance on M1:**
- Q4_K_M: ~50-100 tokens/sec
- Q5_K_M: ~40-80 tokens/sec
- F16: ~20-40 tokens/sec

### Option 2: Transfer Model from Linux to macOS

If you trained on Linux RTX 4090 and want to use on macOS M1:

1. **On Linux (after training):**
   ```bash
   # Compress the GGUF model
   tar -czf qwen_classifier_gguf.tar.gz qwen_classifier_gguf/

   # Transfer to Mac (via scp, rsync, or cloud storage)
   scp qwen_classifier_gguf.tar.gz user@mac:/path/to/destination/
   ```

2. **On macOS:**
   ```bash
   # Extract the model
   tar -xzf qwen_classifier_gguf.tar.gz

   # Import to LM Studio
   # Open LM Studio → Import Model → Select GGUF file
   ```

3. **Alternative transfer methods:**
   - Cloud storage (Dropbox, Google Drive, etc.)
   - USB drive
   - Network share
   - GitHub LFS (for version control)

### Option 3: Browser Deployment (ONNX)

**Best for:** Web applications, browser-based inference

Open `browser_demo.html` in a browser (needs a local server for file access):

```bash
# Using Python
python -m http.server 8000

# Then open http://localhost:8000/browser_demo.html
```

**Note:** By default, the demo uses the base Qwen model. Update `MODEL_PATH` in the HTML file to use your fine-tuned model.

#### Upload to Hugging Face Hub

1. Create account at https://huggingface.co
2. Install Hugging Face CLI:
   ```bash
   pip install huggingface-hub
   huggingface-cli login
   ```

3. Upload your model:
   ```bash
   # Upload ONNX model
   huggingface-cli upload username/qwen-classifier-onnx ./qwen_classifier_onnx_quantized
   ```

4. Update `browser_demo.html`:
   ```javascript
   const MODEL_PATH = 'username/qwen-classifier-onnx';
   ```

#### Self-hosted Model

Serve the ONNX model directory with proper CORS headers:

```bash
# Example with simple HTTP server
python -m http.server 8080 --directory qwen_classifier_onnx_quantized
```

Update `browser_demo.html`:
```javascript
const MODEL_PATH = 'http://localhost:8080';
```

## Project Structure

```
web_transformer/
├── data.py                              # Original dataset
├── requirements.txt                     # Python dependencies (macOS)
├── requirements-linux.txt               # Python dependencies (Linux)
├── prepare_data.py                      # Dataset preparation
├── train.py                             # Training script
├── test_inference.py                    # Model evaluation
├── export_to_gguf.py                    # GGUF conversion (LM Studio)
├── export_to_onnx.py                    # ONNX conversion (Browser)
├── train_pipeline_linux.sh              # Linux RTX 4090 pipeline
├── train_pipeline_macos.sh              # macOS M1 pipeline
├── run_pipeline.sh                      # Legacy pipeline script
├── browser_demo.html                    # Web demo
├── README.md                            # This file
├── LINUX_SETUP.md                       # Linux setup guide
├── training_data/                       # Prepared datasets (generated)
├── qwen_classifier/                     # Trained models (generated)
├── qwen_classifier_gguf/                # GGUF models for LM Studio (generated)
└── qwen_classifier_onnx_quantized/      # ONNX models for browser (generated)
```

## Customization

### Adjust Training Parameters

Edit `train.py`, modify the `Config` class:

```python
@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # Try 1.5B for better accuracy
    num_epochs: int = 10                             # Increase for more training
    batch_size: int = 4                              # Reduce if OOM errors
    learning_rate: float = 2e-5                      # Lower for stability
    # ... more options
```

### Use Different Model Size

Available Qwen2.5 models:
- `Qwen/Qwen2.5-0.5B-Instruct` (smallest, fastest)
- `Qwen/Qwen2.5-1.5B-Instruct` (better accuracy)
- `Qwen/Qwen2.5-3B-Instruct` (best accuracy, slower)

Larger models = better accuracy but slower training/inference.

### Disable Data Augmentation

Edit `prepare_data.py`, comment out augmentation:

```python
# augmented_dataset = augment_data(TEST_DATASET)
# dataset = augmented_dataset
dataset = TEST_DATASET  # Use original only
```

### Add Your Own Categories

1. Update `data.py` with your categories and examples
2. Update `CATEGORIES` list in:
   - `prepare_data.py`
   - `test_inference.py`
   - `browser_demo.html`

## Troubleshooting

### "Out of memory" errors
- Reduce `batch_size` in `train.py` (try 2 or 1)
- Increase `gradient_accumulation_steps` to compensate
- Close other applications

### Training is very slow
- Check if MPS is enabled (should see "Using Apple Silicon MPS acceleration")
- Ensure you're not using the model while training
- Try smaller model (0.5B instead of 1.5B)

### Low accuracy
- Increase training epochs (try 15-20)
- Add more training data or better augmentation
- Try larger model (1.5B or 3B)
- Check if categories are well-defined and distinguishable

### ONNX export fails
- Install optimum properly: `pip install optimum[exporters]`
- Try alternative export in `export_to_onnx.py` (it has fallback)
- Some models need specific ONNX opset versions

### Browser demo doesn't work
- Ensure model files are accessible (CORS headers)
- Check browser console for errors
- Verify MODEL_PATH is correct
- Try Chrome/Edge (better WebGPU support)

## Performance Benchmarks

### Training Performance

**M1 Mac (8GB RAM):**
- Data preparation: < 1 minute
- Training (10 epochs, 0.5B model): 20-40 minutes
- Training (10 epochs, 1.5B model): 60-90 minutes
- GGUF export: 2-5 minutes
- ONNX export: 2-5 minutes

**Linux RTX 4090 (24GB VRAM):**
- Data preparation: < 1 minute
- Training (10 epochs, 0.5B model): 5-10 minutes (3-5x faster than M1)
- Training (10 epochs, 1.5B model): 15-20 minutes
- Training (10 epochs, 3B model): 30-40 minutes
- GGUF export: 2-5 minutes
- ONNX export: 2-5 minutes

### Inference Performance

**LM Studio on M1 Mac:**
- Q4_K_M: ~50-100 tokens/sec (recommended)
- Q5_K_M: ~40-80 tokens/sec
- F16: ~20-40 tokens/sec

**Browser (transformers.js):**
- First load: 5-10 seconds (model loading)
- Subsequent queries: ~500ms per classification

### Model Accuracy

**Accuracy expectations:**
- 59 examples (no augmentation): 70-85%
- 177 examples (with augmentation): 85-95%
- 500+ examples: 90-98%

### Model Sizes

**GGUF (for LM Studio):**
- FP16: ~500MB
- Q4_K_M: ~150-200MB (recommended)
- Q5_K_M: ~200-250MB
- Q8_0: ~300-350MB

**ONNX (for browser):**
- Quantized: ~150-200MB

## Next Steps

1. **Improve dataset:** Add more diverse examples for each category
2. **Fine-tune prompts:** Experiment with different instruction formats
3. **Try few-shot:** Add example classifications in the prompt
4. **Ensemble models:** Train multiple models and vote
5. **Production deployment:** Add error handling, logging, monitoring

## Resources

- [Transformers.js Documentation](https://huggingface.co/docs/transformers.js)
- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [ONNX Runtime](https://onnxruntime.ai/)
- [MLX Framework](https://github.com/ml-explore/mlx)

## License

This pipeline is provided as-is. Model licenses follow the original Qwen2.5 license (Apache 2.0).

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review training logs in `qwen_classifier/`
3. Test with the base model first to isolate issues
