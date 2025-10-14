# Quick Start Guide

Get your Qwen text classifier running in 3 steps!

## Prerequisites

- macOS with M1/M2/M3 chip
- Python 3.8+
- 20-40 minutes for training

## Option 1: Automated Pipeline (Recommended)

Run everything with one command:

```bash
bash run_pipeline.sh
```

This will guide you through:
1. Setting up virtual environment
2. Installing dependencies
3. Preparing data
4. Training model
5. Testing accuracy
6. Exporting to ONNX

## Option 2: Manual Steps

### 1. Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (takes 5-10 min)
pip install -r requirements.txt
```

### 2. Train

```bash
# Prepare dataset
python prepare_data.py

# Train model (takes 20-40 min)
python train.py

# Test accuracy
python test_inference.py
```

### 3. Deploy to Browser

```bash
# Export to ONNX
python export_to_onnx.py

# Test in browser
python -m http.server 8000
# Open http://localhost:8000/browser_demo.html
```

## What You'll Get

After training, you'll have:
- **Fine-tuned model** optimized for your 8 categories
- **ONNX model** (~150-200MB) ready for browser deployment
- **85-95% accuracy** on text classification
- **Interactive web demo** for testing

## Expected Timeline

| Step | Time | Output |
|------|------|--------|
| Setup | 5-10 min | Dependencies installed |
| Data prep | < 1 min | 177 training examples |
| Training | 20-40 min | Fine-tuned model |
| Testing | 1-2 min | Accuracy report |
| ONNX export | 2-5 min | Browser-ready model |

**Total:** ~30-60 minutes

## Quick Test

After training, test your model interactively:

```bash
python test_inference.py
```

Type `y` when prompted for interactive mode, then try:
- "How many vacation days do I have?" → Should predict **HR**
- "My laptop is broken" → Should predict **IT**
- "What's my sales quota?" → Should predict **Sales**

## Troubleshooting

**Import errors?**
```bash
pip install -r requirements.txt --upgrade
```

**Out of memory?**
- Edit `train.py`, set `batch_size = 2` (line 21)
- Close other apps

**Low accuracy?**
- Increase epochs: set `num_epochs = 15` in `train.py`
- Add more training examples to `data.py`

## Next Steps

1. **Customize categories** - Edit `data.py` with your own examples
2. **Deploy to web** - Upload ONNX model to Hugging Face
3. **Improve accuracy** - Add more training data, try larger model

See **README.md** for detailed documentation.

## Key Files

```
prepare_data.py     → Prepare dataset
train.py           → Train model
test_inference.py  → Test accuracy
export_to_onnx.py  → Export for browser
browser_demo.html  → Web interface
```

## Help

- **Full documentation:** See README.md
- **Model configs:** Edit Config class in train.py
- **Categories:** Update CATEGORIES lists in all .py files

Ready? Start with: `bash run_pipeline.sh`
