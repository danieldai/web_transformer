#!/bin/bash

# Complete training pipeline for Qwen text classification on Linux with RTX 4090
# Optimized for CUDA, exports to GGUF format for LM Studio on macOS M1
# Run with: bash train_pipeline_linux.sh

set -e  # Exit on error

echo "=========================================="
echo "Qwen Training Pipeline - Linux RTX 4090"
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

# Step 0: Check GPU
echo "=========================================="
echo "Step 0: Checking NVIDIA GPU"
echo "=========================================="

if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if [ $? -eq 0 ]; then
    print_success "NVIDIA GPU detected"
else
    print_error "Failed to query GPU"
    exit 1
fi
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

# Check if PyTorch with CUDA is installed
echo "=========================================="
echo "Checking PyTorch CUDA Installation"
echo "=========================================="

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    print_warning "PyTorch with CUDA not found. Installing..."
    echo ""

    print_info "Installing PyTorch with CUDA 12.1 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    print_success "PyTorch with CUDA installed"
    echo ""
fi

# Verify CUDA
print_info "Verifying CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

if [ $? -ne 0 ]; then
    print_error "CUDA verification failed"
    exit 1
fi

print_success "CUDA verified successfully"
echo ""

# Check if other dependencies are installed
if ! python -c "import transformers" 2>/dev/null; then
    print_warning "Dependencies not installed. Installing from requirements-linux.txt..."
    pip install -r requirements-linux.txt
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
echo "Step 2: Training Model (RTX 4090)"
echo "=========================================="
print_info "Training optimizations for RTX 4090:"
echo "  - FP16 precision (automatic)"
echo "  - CUDA acceleration"
echo "  - Larger batch sizes supported"
echo ""
print_warning "Training will take 5-10 minutes for 0.5B model, 15-20 min for 1.5B"
echo ""

read -p "Continue with training? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Starting training..."

    # Monitor GPU in background
    (
        sleep 5  # Wait for training to start
        print_info "GPU status during training (press Ctrl+C to stop monitoring):"
        watch -n 2 "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | awk -F',' '{print \"GPU: \"\$1\"% | Memory: \"\$2\"% (\"\$3\"MB/\"\$4\"MB) | Temp: \"\$5\"°C\"}'"
    ) &
    WATCH_PID=$!

    python train.py
    TRAIN_STATUS=$?

    # Kill GPU monitoring
    kill $WATCH_PID 2>/dev/null || true

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

# Step 4: Export to GGUF (for LM Studio on macOS)
echo "=========================================="
echo "Step 4: Exporting to GGUF for LM Studio"
echo "=========================================="
print_info "GGUF format is required for LM Studio on macOS M1"
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

        print_info "Building llama.cpp..."
        cd "$HOME/llama.cpp"
        make -j$(nproc)

        print_info "Installing llama.cpp dependencies..."
        pip install -r requirements.txt

        cd - > /dev/null

        print_success "llama.cpp installed"
        echo ""
    fi

    print_info "Exporting to GGUF format..."
    python export_to_gguf.py --quant-types q4_k_m q5_k_m

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
echo "Next Steps: Transfer to macOS M1"
echo "=========================================="
echo ""
echo "1. Transfer GGUF model to your Mac:"
echo "   scp -r ./qwen_classifier_gguf your-mac:/path/to/destination"
echo ""
echo "2. On macOS M1, open LM Studio"
echo ""
echo "3. Import model:"
echo "   - Click 'Import Model'"
echo "   - Select: qwen_classifier_gguf/qwen-classifier-q4_k_m.gguf"
echo ""
echo "4. Start using your fine-tuned classifier!"
echo ""
echo "=========================================="
echo "Performance Stats (RTX 4090)"
echo "=========================================="
echo ""
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader | \
    awk -F',' '{print "GPU: "$1"\nMemory Used: "$2" / "$3"\nUtilization: "$4}'
echo ""
print_success "All done! Happy classifying!"
echo ""
