# Qwen Text Classification Training Pipeline

Fine-tune Qwen2.5-0.5B for text classification on M1 Mac, optimized for browser deployment with transformers.js.

## Overview

This pipeline allows you to:
1. Fine-tune a small Qwen LLM on your custom text classification dataset
2. Export the model to ONNX format with quantization
3. Deploy the model in web browsers using transformers.js

## Prerequisites

- macOS with M1/M2/M3 chip
- Python 3.8+
- At least 8GB RAM
- ~5GB disk space for models

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

**Note:** First installation may take 10-15 minutes as it downloads all dependencies.

## Training Workflow

### Step 1: Prepare Dataset

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

### Step 2: Train Model

Fine-tune Qwen2.5-0.5B on your dataset:

```bash
python train.py
```

**Training details:**
- Model: Qwen2.5-0.5B-Instruct (~500MB)
- Full parameter fine-tuning (not LoRA)
- Optimized for M1 with MPS acceleration
- ~20-40 minutes training time
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

### Step 3: Test Model

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

### Step 4: Export to ONNX

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

## Browser Deployment

### Option 1: Local Testing

Open `browser_demo.html` in a browser (needs a local server for file access):

```bash
# Using Python
python -m http.server 8000

# Then open http://localhost:8000/browser_demo.html
```

**Note:** By default, the demo uses the base Qwen model. Update `MODEL_PATH` in the HTML file to use your fine-tuned model.

### Option 2: Upload to Hugging Face Hub

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

### Option 3: Self-hosted Model

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
├── data.py                    # Original dataset
├── requirements.txt           # Python dependencies
├── prepare_data.py           # Dataset preparation
├── train.py                  # Training script
├── test_inference.py         # Model evaluation
├── export_to_onnx.py         # ONNX conversion
├── browser_demo.html         # Web demo
├── README.md                 # This file
├── training_data/            # Prepared datasets (generated)
├── qwen_classifier/          # Trained models (generated)
└── qwen_classifier_onnx_quantized/  # ONNX models (generated)
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

**M1 Mac (8GB RAM):**
- Data preparation: < 1 minute
- Training (10 epochs, 0.5B model): 20-40 minutes
- ONNX export: 2-5 minutes
- Browser inference: ~500ms per query (first load slower)

**Accuracy expectations:**
- 59 examples (no augmentation): 70-85%
- 177 examples (with augmentation): 85-95%
- 500+ examples: 90-98%

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
