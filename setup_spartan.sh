#!/bin/bash
# Setup script for Spartan HPC environment
# Run this script first to set up your environment

echo "Setting up environment for Qwen CoT + Few-Shot training on Spartan"
echo "=================================================================="

# Check if running on Spartan
if [[ $(hostname) == *"spartan"* ]]; then
    echo "✓ Running on Spartan HPC"
else
    echo "⚠ Warning: This script is designed for Spartan HPC"
fi

# Set your project directory - MODIFY THIS
PROJECT_ROOT="/data/scratch/projects/punim2103"  # Replace punim2103 with your project ID
REPO_DIR="$PROJECT_ROOT/PrimeVul"

echo "Project root: $PROJECT_ROOT"
echo "Repository directory: $REPO_DIR"

# Create necessary directories
echo "Creating directories..."
mkdir -p "$PROJECT_ROOT/data"
mkdir -p "$PROJECT_ROOT/checkpoints"
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/huggingface_cache"
mkdir -p "$PROJECT_ROOT/transformers_cache"
mkdir -p "$PROJECT_ROOT/datasets_cache"
mkdir -p "$PROJECT_ROOT/wandb_cache"

# Create virtual environment if it doesn't exist
VENV_DIR="$PROJECT_ROOT/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    module load Python/3.10.4-GCCcore-11.3.0
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    
    echo "Installing required packages..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers datasets accelerate peft
    pip install scikit-learn pandas numpy tqdm
    pip install wandb  # Optional
    pip install flash-attn --no-build-isolation  # Optional, may fail on some systems
    
    echo "Virtual environment created and packages installed"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Clone or update repository
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repository..."
    cd "$PROJECT_ROOT"
    git clone https://github.com/TheSyx04/PrimeVul.git
    echo "Repository cloned to $REPO_DIR"
else
    echo "Repository already exists. Pulling latest changes..."
    cd "$REPO_DIR"
    git pull
fi

# Make scripts executable
chmod +x "$REPO_DIR/run_qwen_cot_fewshot_spartan.sh"

# Create a sample data preparation script
cat > "$PROJECT_ROOT/prepare_data.py" << 'EOF'
#!/usr/bin/env python3
"""
Sample script to prepare data in the correct format for training
Modify this script according to your data source
"""

import json
import os

def prepare_sample_data():
    """Create sample data files for testing"""
    
    # Sample data - replace with your actual data loading logic
    sample_data = [
        {
            "commit_id": "sample_001",
            "code_change": "void unsafe_copy(char* dest, char* src) { strcpy(dest, src); }",
            "label": 1,
            "project": "test_project"
        },
        {
            "commit_id": "sample_002", 
            "code_change": "void safe_copy(char* dest, char* src, size_t size) { strncpy(dest, src, size-1); dest[size-1] = '\\0'; }",
            "label": 0,
            "project": "test_project"
        }
    ] * 100  # Repeat for testing
    
    # Create train/eval/test splits
    train_data = sample_data[:70]
    eval_data = sample_data[70:85]
    test_data = sample_data[85:]
    
    # Write data files
    data_dir = "/data/scratch/projects/punim2103/data"  # Update with your project ID
    
    with open(f"{data_dir}/train.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    
    with open(f"{data_dir}/eval.jsonl", "w") as f:
        for item in eval_data:
            f.write(json.dumps(item) + "\n")
    
    with open(f"{data_dir}/test.jsonl", "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Sample data files created in {data_dir}")
    print(f"Train: {len(train_data)} samples")
    print(f"Eval: {len(eval_data)} samples") 
    print(f"Test: {len(test_data)} samples")

if __name__ == "__main__":
    prepare_sample_data()
EOF

chmod +x "$PROJECT_ROOT/prepare_data.py"

# Create environment file
cat > "$PROJECT_ROOT/spartan_env.sh" << EOF
#!/bin/bash
# Source this file to set up environment variables
# Usage: source spartan_env.sh

export PROJECT_ROOT="$PROJECT_ROOT"
export REPO_DIR="$REPO_DIR"
export VENV_DIR="$VENV_DIR"

# HuggingFace cache directories
export HF_HOME="$PROJECT_ROOT/huggingface_cache"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/transformers_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/datasets_cache"

# Wandb cache (optional)
export WANDB_CACHE_DIR="$PROJECT_ROOT/wandb_cache"

# CUDA settings
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

echo "Environment variables set for Spartan HPC"
echo "Project root: \$PROJECT_ROOT"
echo "Repository: \$REPO_DIR"
echo "Virtual environment: \$VENV_DIR"
EOF

echo ""
echo "Setup completed! Next steps:"
echo "1. Edit the paths in run_qwen_cot_fewshot_spartan.sh to match your setup"
echo "2. Prepare your data files in JSONL format"
echo "3. Optionally run: python $PROJECT_ROOT/prepare_data.py (for sample data)"
echo "4. Source the environment: source $PROJECT_ROOT/spartan_env.sh"
echo "5. Submit job: sbatch run_qwen_cot_fewshot_spartan.sh"
echo ""
echo "Key files created:"
echo "- $PROJECT_ROOT/spartan_env.sh (environment setup)"
echo "- $PROJECT_ROOT/prepare_data.py (sample data preparation)"
echo "- Virtual environment: $VENV_DIR"
echo ""
echo "Remember to:"
echo "- Update project ID (punim2103) in all scripts to your actual project ID"
echo "- Modify data paths in the training script"
echo "- Set up your Wandb API key if using Wandb logging"
