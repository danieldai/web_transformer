#!/bin/bash

# Complete training pipeline for Qwen text classification on macOS M1
# Optimized for Apple Silicon MPS, exports to GGUF format for LM Studio
# Run with: bash train_pipeline_macos.sh

set -e  # Exit on error

echo "=========================================="
echo "Qwen Training Pipeline - macOS M1"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 0: Check system
echo "=========================================="
echo "Step 0: Checking System"
echo "=========================================="

# Check if running on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    print_warning "Not running on Apple Silicon (arm64). This script is optimized for M1/M2/M3."
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
print_success "Python $PYTHON_VERSION detected"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_warning "Virtual environment not found. Creating one..."
    python3 -m venv venv
    print_success "Virtual environment created"
    echo ""
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Check if PyTorch is installed
echo "=========================================="
echo "Checking PyTorch Installation"
echo "=========================================="

if ! python -c "import torch" 2>/dev/null; then
    print_warning "PyTorch not found. Installing..."
    echo ""

    print_info "Installing PyTorch for Apple Silicon..."
    pip install torch torchvision torchaudio

    print_success "PyTorch installed"
    echo ""
fi

# Verify MPS
print_info "Verifying MPS (Metal Performance Shaders) availability..."
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'MPS built: {torch.backends.mps.is_built()}')"

if [ $? -ne 0 ]; then
    print_error "MPS verification failed"
    exit 1
fi

print_success "MPS verified successfully"
echo ""

# Check if other dependencies are installed
if ! python -c "import transformers" 2>/dev/null; then
    print_warning "Dependencies not installed. Installing from requirements.txt..."
    pip install -r requirements.txt
    print_success "Dependencies installed"
    echo ""
fi

# Step 1: Prepare data
echo "=========================================="
echo "Step 1: Preparing Dataset"
echo "=========================================="
python prepare_data.py
if [ $? -eq 0 ]; then
    print_success "Data preparation complete"
else
    print_error "Data preparation failed"
    exit 1
fi
echo ""

# Step 2: Train model
echo "=========================================="
echo "Step 2: Training Model (M1/M2/M3)"
echo "=========================================="
print_info "Training optimizations for Apple Silicon:"
echo "  - MPS (Metal) acceleration"
echo "  - FP32 precision (for stability)"
echo "  - Optimized batch sizes for unified memory"
echo ""
print_warning "Training will take 20-40 minutes for 0.5B model"
print_info "You can continue using your Mac during training"
echo ""

read -p "Continue with training? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Starting training..."
    echo ""

    python train.py
    TRAIN_STATUS=$?

    if [ $TRAIN_STATUS -eq 0 ]; then
        print_success "Training complete"
    else
        print_error "Training failed"
        exit 1
    fi
else
    print_warning "Training skipped. Run 'python train.py' when ready."
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
    if [ $? -eq 0 ]; then
        print_success "Testing complete"
    else
        print_warning "Testing had issues (non-critical)"
    fi
else
    print_warning "Testing skipped. Run 'python test_inference.py' when ready."
fi
echo ""

# Step 4: Export to GGUF (for LM Studio)
echo "=========================================="
echo "Step 4: Exporting to GGUF for LM Studio"
echo "=========================================="
print_info "GGUF format is required for LM Studio on macOS"
echo ""
read -p "Export model to GGUF? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check if llama.cpp is available
    if [ ! -d "$HOME/llama.cpp" ]; then
        print_warning "llama.cpp not found. Installing..."
        echo ""

        print_info "Cloning llama.cpp..."
        git clone https://github.com/ggerganov/llama.cpp.git "$HOME/llama.cpp"

        print_info "Building llama.cpp for Apple Silicon..."
        cd "$HOME/llama.cpp"

        # Build with Metal support for M1
        LLAMA_METAL=1 make -j$(sysctl -n hw.ncpu)

        print_info "Installing llama.cpp dependencies..."
        pip install -r requirements.txt

        cd - > /dev/null

        print_success "llama.cpp installed with Metal support"
        echo ""
    fi

    print_info "Exporting to GGUF format..."
    print_info "Creating q4_k_m quantization (recommended for M1)..."
    python export_to_gguf.py --quant-types q4_k_m

    if [ $? -eq 0 ]; then
        print_success "GGUF export complete"
    else
        print_error "GGUF export failed"
        echo ""
        print_info "You can manually export later with: python export_to_gguf.py"
    fi
else
    print_warning "GGUF export skipped. Run 'python export_to_gguf.py' when ready."
fi
echo ""

# Optional: Export to ONNX (for browser deployment)
echo "=========================================="
echo "Step 5: Export to ONNX (Optional)"
echo "=========================================="
print_info "ONNX format is for browser deployment with transformers.js"
echo ""
read -p "Export model to ONNX? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python export_to_onnx.py
    if [ $? -eq 0 ]; then
        print_success "ONNX export complete"
    else
        print_warning "ONNX export failed (non-critical)"
    fi
else
    print_warning "ONNX export skipped. Run 'python export_to_onnx.py' when ready."
fi
echo ""

# Summary
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
print_success "Your fine-tuned model is ready!"
echo ""
echo "Available model formats:"
echo "  - PyTorch: ./qwen_classifier/final"
echo "  - GGUF (LM Studio): ./qwen_classifier_gguf/"
echo "  - ONNX (Browser): ./qwen_classifier_onnx_quantized/"
echo ""
echo "=========================================="
echo "Using with LM Studio (macOS)"
echo "=========================================="
echo ""
echo "1. Open LM Studio"
echo ""
echo "2. Import your fine-tuned model:"
echo "   - Click 'Import Model' or drag-and-drop"
echo "   - Select: ./qwen_classifier_gguf/qwen-classifier-q4_k_m.gguf"
echo ""
echo "3. Start a new chat:"
echo "   - Your model will appear in the model list"
echo "   - Select it and start chatting!"
echo ""
echo "4. Example prompt for classification:"
echo "   'Classify this question into a category: How do I reset my password?'"
echo ""
echo "=========================================="
echo "Model Recommendations"
echo "=========================================="
echo ""
echo "For M1/M2/M3 Macs in LM Studio:"
echo "  - Recommended: q4_k_m (best balance of speed/quality)"
echo "  - Faster: q4_0 (smaller, faster, slightly lower quality)"
echo "  - Best quality: f16 (largest, slowest, best quality)"
echo ""
print_info "You can create additional quantizations with:"
echo "  python export_to_gguf.py --quant-types q4_0 q5_k_m q8_0"
echo ""
print_success "All done! Happy classifying!"
echo ""
