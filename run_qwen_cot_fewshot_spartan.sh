#!/bin/bash
#SBATCH --job-name=qwen_cot_fewshot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen_cot_%j.out
#SBATCH --error=logs/qwen_cot_%j.err

# Spartan HPC Script for Chain of Thought + Few-Shot Fine-tuning
# Submit with: sbatch run_qwen_cot_fewshot_spartan.sh

echo "Starting job on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"

# Load required modules (adjust based on your Spartan setup)
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load GCC/11.3.0

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/data/scratch/projects/punim2103/huggingface_cache"  # Adjust to your scratch space
export TRANSFORMERS_CACHE="/data/scratch/projects/punim2103/transformers_cache"
export HF_DATASETS_CACHE="/data/scratch/projects/punim2103/datasets_cache"

# Wandb settings (optional - comment out if not using)
export WANDB_API_KEY="fb5a5b79b5aafdb17cb882dd76ac2e0cde9adf8d"
export WANDB_CACHE_DIR="/data/scratch/projects/punim2103/wandb_cache"

# Model and data paths - ADJUST THESE TO YOUR SPARTAN PATHS
MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
PROJECT="primevul_spartan"
MODEL_DIR="qwen_cot_fewshot_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="/data/scratch/projects/punim2103/checkpoints"  # Use scratch space

# Data files - ADJUST THESE TO YOUR DATA LOCATION
TRAIN_DATA="/data/scratch/projects/punim2103/data/train.jsonl"
EVAL_DATA="/data/scratch/projects/punim2103/data/eval.jsonl"
TEST_DATA="/data/scratch/projects/punim2103/data/test.jsonl"

# Training configuration optimized for Spartan
BATCH_SIZE=1
EVAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=32  # Increased for effective batch size
LEARNING_RATE=1e-5  # Slightly lower for stability
EPOCHS=3
BLOCK_SIZE=1024  # Reduced to fit in memory
MAX_PATIENCE=2  # Reduced for faster training

# LoRA configuration
LORA_R=8  # Reduced for memory efficiency
LORA_ALPHA=16
LORA_DROPOUT=0.1

echo "============================================================================="
echo "Chain of Thought + Few-Shot Fine-tuning for Qwen3-Coder-30B-A3B-Instruct"
echo "Running on Spartan HPC"
echo "============================================================================="
echo "Model: $MODEL_NAME"
echo "Project: $PROJECT"
echo "Output Directory: $OUTPUT_DIR"
echo "GPU Memory Available: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
echo "============================================================================="

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/$PROJECT"

# Check if data files exist
for data_file in "$TRAIN_DATA" "$EVAL_DATA" "$TEST_DATA"; do
    if [ ! -f "$data_file" ]; then
        echo "ERROR: Data file not found: $data_file"
        echo "Please update the data paths in this script"
        exit 1
    fi
done

# Activate virtual environment if needed (uncomment and adjust path)
# source /data/scratch/projects/punim2103/venv/bin/activate

# Print system information
echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version | grep release)"
echo "GPU information:"
nvidia-smi

# Change to the correct directory
cd /data/scratch/projects/punim2103/PrimeVul  # Adjust to your repo location

# Run training with optimized settings for Spartan
python os_expr/run_ft_qwen_cot_fewshot.py \
    --project "$PROJECT" \
    --model_dir "$MODEL_DIR" \
    --model_name_or_path "$MODEL_NAME" \
    --tokenizer_name "$MODEL_NAME" \
    --train_data_file "$TRAIN_DATA" \
    --eval_data_file "$EVAL_DATA" \
    --test_data_file "$TEST_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --block_size $BLOCK_SIZE \
    --train_batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --epoch $EPOCHS \
    --max_patience $MAX_PATIENCE \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 50 \
    --save_steps 200 \
    --use_lora \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --use_few_shot \
    --use_chat_template \
    --num_few_shot_examples 3 \
    --do_train \
    --do_test \
    --evaluate_during_training \
    --force_single_gpu \
    --use_wandb \
    --wandb_project "primevul-qwen-cot-spartan" \
    --wandb_run_name "${PROJECT}_${MODEL_DIR}" \
    --wandb_tags "spartan" "qwen" "cot" "few-shot" "lora" \
    --seed 42

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Results saved in: $OUTPUT_DIR/$PROJECT/"
    echo "Logs available in: logs/qwen_cot_$SLURM_JOB_ID.out"
    
    # Optional: Copy results to a more permanent location
    # cp -r "$OUTPUT_DIR/$PROJECT" "/home/punim2103/permanent_storage/"
else
    echo "Training failed with exit code: $?"
    echo "Check logs for details: logs/qwen_cot_$SLURM_JOB_ID.err"
fi

echo "Job completed at $(date)"
