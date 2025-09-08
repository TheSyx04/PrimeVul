#!/bin/bash

# Chain of Thought + Few-Shot Fine-tuning for Qwen3-Coder-30B-A3B-Instruct
# This script demonstrates how to run few-shot learning with chain of thought reasoning

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY="fb5a5b79b5aafdb17cb882dd76ac2e0cde9adf8d"

# Model and data paths
MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
PROJECT="your_project_name"
MODEL_DIR="qwen_cot_fewshot"
OUTPUT_DIR="./checkpoints"

# Data files - update these paths according to your data
TRAIN_DATA="./data/train.jsonl"
EVAL_DATA="./data/eval.jsonl"
TEST_DATA="./data/test.jsonl"

# Training configuration
BATCH_SIZE=1
EVAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=16
LEARNING_RATE=2e-5
EPOCHS=3
BLOCK_SIZE=2048
MAX_PATIENCE=3

# LoRA configuration
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1

echo "Starting Chain of Thought + Few-Shot Fine-tuning for Qwen3-Coder-30B-A3B-Instruct"
echo "============================================================================="
echo "Model: $MODEL_NAME"
echo "Project: $PROJECT"
echo "Using Few-Shot Examples: YES"
echo "Using Chain of Thought: YES"
echo "Using LoRA: YES"
echo "============================================================================="

# Run training with Chain of Thought + Few-Shot + LoRA
python run_ft_qwen_cot_fewshot.py \
    --project $PROJECT \
    --model_dir $MODEL_DIR \
    --model_name_or_path $MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --train_data_file $TRAIN_DATA \
    --eval_data_file $EVAL_DATA \
    --test_data_file $TEST_DATA \
    --output_dir $OUTPUT_DIR \
    --block_size $BLOCK_SIZE \
    --train_batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --epoch $EPOCHS \
    --max_patience $MAX_PATIENCE \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 100 \
    --save_steps 500 \
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
    --wandb_project "primevul-qwen-cot" \
    --wandb_run_name "${PROJECT}_qwen_cot_fewshot_${MODEL_DIR}" \
    --wandb_tags "qwen" "cot" "few-shot" "lora" "vulnerability-detection" \
    --seed 42

echo "Training completed!"
echo "Check the output directory: $OUTPUT_DIR"
echo "Predictions saved in: $OUTPUT_DIR/$PROJECT/predictions_qwen_cot.txt"
