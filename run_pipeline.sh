#!/bin/bash

# Complete training pipeline for Qwen text classification
# Run with: bash run_pipeline.sh

set -e  # Exit on error

echo "=========================================="
echo "Qwen Text Classification Pipeline"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import transformers" 2>/dev/null; then
    echo -e "${YELLOW}Dependencies not installed. Installing...${NC}"
    pip install -q -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
    echo ""
fi

# Step 1: Prepare data
echo "=========================================="
echo "Step 1: Preparing Dataset"
echo "=========================================="
python prepare_data.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Data preparation complete${NC}"
else
    echo -e "${RED}✗ Data preparation failed${NC}"
    exit 1
fi
echo ""

# Step 2: Train model
echo "=========================================="
echo "Step 2: Training Model"
echo "=========================================="
echo -e "${YELLOW}This will take 20-40 minutes on M1 Mac${NC}"
read -p "Continue with training? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python train.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Training complete${NC}"
    else
        echo -e "${RED}✗ Training failed${NC}"
        exit 1
    fi
else
    echo "Training skipped. Run 'python train.py' when ready."
    exit 0
fi
echo ""

# Step 3: Test model
echo "=========================================="
echo "Step 3: Testing Model"
echo "=========================================="
read -p "Test the model now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python test_inference.py
else
    echo "Testing skipped. Run 'python test_inference.py' when ready."
fi
echo ""

# Step 4: Export to ONNX
echo "=========================================="
echo "Step 4: Exporting to ONNX"
echo "=========================================="
read -p "Export model to ONNX for browser deployment? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python export_to_onnx.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ ONNX export complete${NC}"
    else
        echo -e "${RED}✗ ONNX export failed${NC}"
        exit 1
    fi
else
    echo "ONNX export skipped. Run 'python export_to_onnx.py' when ready."
fi
echo ""

# Summary
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Your fine-tuned model is ready at:"
echo "  - PyTorch: ./qwen_classifier/final"
echo "  - ONNX: ./qwen_classifier_onnx_quantized"
echo ""
echo "Next steps:"
echo "  1. Test model: python test_inference.py"
echo "  2. Open browser demo: open browser_demo.html"
echo "  3. Deploy to web (see README.md)"
echo ""
echo "For help, see README.md"
