# Spartan HPC Setup Guide for Qwen CoT + Few-Shot Training

This guide provides step-by-step instructions for setting up and running the Chain of Thought + Few-Shot fine-tuning on Spartan HPC.

## Prerequisites

1. Access to Spartan HPC
2. A project allocation with GPU access
3. Sufficient storage quota in `/data/scratch/projects/`

## Setup Instructions

### 1. Initial Setup

First, run the setup script to create directories and environment:

```bash
# SSH to Spartan
ssh your_username@spartan.hpc.unimelb.edu.au

# Navigate to your project directory
cd /data/scratch/projects/punim2103  # Replace with your project ID

# Download the setup script
wget https://raw.githubusercontent.com/TheSyx04/PrimeVul/main/setup_spartan.sh
chmod +x setup_spartan.sh

# Run setup (this will create directories and virtual environment)
./setup_spartan.sh
```

### 2. Update Configuration

Edit the following files to match your project:

**In `run_qwen_cot_fewshot_spartan.sh`:**
- Update `punim2103` to your actual project ID in all paths
- Set correct data file paths
- Adjust memory and time requirements based on your allocation

**Example path updates:**
```bash
# Change these lines:
PROJECT_ROOT="/data/scratch/projects/punim2103"
TRAIN_DATA="/data/scratch/projects/punim2103/data/train.jsonl"

# To your project:
PROJECT_ROOT="/data/scratch/projects/YOUR_PROJECT_ID"
TRAIN_DATA="/data/scratch/projects/YOUR_PROJECT_ID/data/train.jsonl"
```

### 3. Prepare Your Data

Your data should be in JSONL format with this structure:

```json
{
    "commit_id": "unique_identifier",
    "code_change": "source_code_to_analyze",
    "label": 0,  # 0 for safe, 1 for vulnerable
    "project": "project_name"
}
```

Place your data files in:
- `/data/scratch/projects/YOUR_PROJECT_ID/data/train.jsonl`
- `/data/scratch/projects/YOUR_PROJECT_ID/data/eval.jsonl`
- `/data/scratch/projects/YOUR_PROJECT_ID/data/test.jsonl`

### 4. Submit Job

```bash
# Activate environment
source /data/scratch/projects/punim2103/spartan_env.sh  # Use your project ID

# Submit the job
cd /data/scratch/projects/punim2103/PrimeVul  # Use your project ID
sbatch run_qwen_cot_fewshot_spartan.sh
```

### 5. Monitor Job

```bash
# Check job status
squeue -u your_username

# Check job output (replace JOBID with actual job ID)
tail -f logs/qwen_cot_JOBID.out

# Check for errors
tail -f logs/qwen_cot_JOBID.err
```

## Resource Requirements

### Memory and GPU
- **GPU**: 1 x V100 or A100 (32GB recommended)
- **CPU Memory**: 64GB
- **Storage**: ~100GB for model cache and checkpoints

### Time Estimates
- **Model Download**: 10-30 minutes (first time only)
- **Training**: 2-8 hours per epoch (depends on dataset size)
- **Total**: Plan for 12-24 hours for 3 epochs

## SLURM Configuration

The script includes optimized SLURM settings:

```bash
#SBATCH --job-name=qwen_cot_fewshot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
```

Adjust these based on your needs and allocation limits.

## Training Configuration

### Optimized for Spartan
- **Batch Size**: 1 (to fit in GPU memory)
- **Gradient Accumulation**: 32 (effective batch size of 32)
- **Sequence Length**: 1024 (reduced from 2048 for memory)
- **LoRA Rank**: 8 (reduced for efficiency)
- **Learning Rate**: 1e-5 (conservative for stability)

### Memory Optimizations Applied
- Reduced batch size and sequence length
- Lower LoRA rank
- Gradient checkpointing
- Mixed precision (bfloat16)
- Automatic cache cleanup

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM)
```bash
# Reduce batch size in the script
BATCH_SIZE=1  # Already minimal

# Reduce sequence length
BLOCK_SIZE=512  # From 1024

# Reduce LoRA rank
LORA_R=4  # From 8
```

#### 2. Model Download Fails
```bash
# Pre-download model
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-32B-Instruct')"
```

#### 3. Job Killed by Scheduler
- Check if you exceeded time/memory limits
- Adjust SLURM parameters in the script
- Use `sacct -j JOBID` to check resource usage

#### 4. Permission Denied
```bash
# Make scripts executable
chmod +x run_qwen_cot_fewshot_spartan.sh
chmod +x setup_spartan.sh
```

### Checking Resources
```bash
# Check your allocation
mybalance

# Check available partitions
sinfo

# Check GPU availability
sinfo -p gpu
```

## File Structure

After setup, your directory structure should look like:

```
/data/scratch/projects/YOUR_PROJECT_ID/
├── PrimeVul/                          # Repository
│   ├── os_expr/
│   │   └── run_ft_qwen_cot_fewshot.py
│   ├── run_qwen_cot_fewshot_spartan.sh
│   └── setup_spartan.sh
├── data/                              # Your data files
│   ├── train.jsonl
│   ├── eval.jsonl
│   └── test.jsonl
├── checkpoints/                       # Model checkpoints
├── logs/                             # Job logs
├── venv/                             # Python environment
├── huggingface_cache/                # Model cache
├── spartan_env.sh                    # Environment variables
└── prepare_data.py                   # Sample data prep script
```

## Output Files

After successful training:
- **Checkpoints**: `/data/scratch/projects/YOUR_PROJECT_ID/checkpoints/`
- **Predictions**: `checkpoints/YOUR_PROJECT/predictions_qwen_cot.txt`
- **Logs**: `logs/qwen_cot_JOBID.out`

## Advanced Usage

### Custom Few-Shot Examples
Edit the `FEW_SHOT_EXAMPLES` in the Python script to add domain-specific examples.

### Hyperparameter Tuning
Modify these variables in the Spartan script:
- `LEARNING_RATE`
- `LORA_R`, `LORA_ALPHA`
- `GRADIENT_ACCUMULATION_STEPS`
- `EPOCHS`

### Multiple Runs
To run multiple experiments, copy and modify the Spartan script with different `MODEL_DIR` names.

## Support

For Spartan-specific issues:
- Spartan Documentation: https://dashboard.hpc.unimelb.edu.au/
- Help Desk: hpc-support@unimelb.edu.au

For code-related issues:
- Check the GitHub repository issues
- Review the training logs for error messages
