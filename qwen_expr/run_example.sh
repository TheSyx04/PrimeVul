#!/bin/bash

# Example script to run Qwen model with few-shot prompting
# Modify the paths and parameters according to your setup

# Basic usage with few-shot examples
python run_qwen_prompting.py \
    --model "Qwen/Qwen3-Coder-480B-A35B-Instruct" \
    --prompt_strategy "std_cls" \
    --data_path "path/to/your/data.jsonl" \
    --output_folder "results/" \
    --fewshot_eg \
    --temperature 0.0 \
    --max_gen_length 1024 \
    --max_context_length 32768

# Chain of thought prompting
# python run_qwen_prompting.py \
#     --model "Qwen/Qwen3-Coder-480B-A35B-Instruct" \
#     --prompt_strategy "cot" \
#     --data_path "path/to/your/data.jsonl" \
#     --output_folder "results/" \
#     --fewshot_eg \
#     --temperature 0.0

# Zero-shot (no few-shot examples)
# python run_qwen_prompting.py \
#     --model "Qwen/Qwen3-Coder-480B-A35B-Instruct" \
#     --prompt_strategy "std_cls" \
#     --data_path "path/to/your/data.jsonl" \
#     --output_folder "results/" \
#     --temperature 0.0

# For systems with limited memory, you can specify max memory per GPU
# python run_qwen_prompting.py \
#     --model "Qwen/Qwen3-Coder-480B-A35B-Instruct" \
#     --prompt_strategy "std_cls" \
#     --data_path "path/to/your/data.jsonl" \
#     --output_folder "results/" \
#     --fewshot_eg \
#     --max_memory "20GB"