# Qwen Few-Shot Prompting for Vulnerability Detection

This folder contains scripts for running the Qwen/Qwen3-Coder-480B-A35B-Instruct model using few-shot prompting techniques for vulnerability detection.

## Files

- `run_qwen_prompting.py`: Main script for running Qwen model with few-shot prompting
- `utils.py`: Prompt templates and utility functions
- `config.yaml`: Configuration file with default settings
- `requirements.txt`: Python dependencies
- `run_example.sh`: Example shell script with different usage patterns

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have sufficient GPU memory for the Qwen/Qwen3-Coder-480B-A35B-Instruct model, or adjust the `max_memory` parameter accordingly.

## Usage

### Basic Usage with Few-Shot Examples

```bash
python run_qwen_prompting.py \
    --model "Qwen/Qwen3-Coder-480B-A35B-Instruct" \
    --prompt_strategy "std_cls" \
    --data_path "path/to/your/data.jsonl" \
    --output_folder "results/" \
    --fewshot_eg \
    --temperature 0.0
```

### Chain-of-Thought Prompting

```bash
python run_qwen_prompting.py \
    --model "Qwen/Qwen3-Coder-480B-A35B-Instruct" \
    --prompt_strategy "cot" \
    --data_path "path/to/your/data.jsonl" \
    --output_folder "results/" \
    --fewshot_eg \
    --temperature 0.0
```

### Zero-Shot (No Few-Shot Examples)

```bash
python run_qwen_prompting.py \
    --model "Qwen/Qwen3-Coder-480B-A35B-Instruct" \
    --prompt_strategy "std_cls" \
    --data_path "path/to/your/data.jsonl" \
    --output_folder "results/" \
    --temperature 0.0
```

## Arguments

- `--model`: Model name (default: "Qwen/Qwen3-Coder-480B-A35B-Instruct")
- `--prompt_strategy`: Prompting strategy - "std_cls" or "cot" (default: "std_cls")
- `--data_path`: Path to input data file in JSONL format (required)
- `--output_folder`: Output folder for results (required)
- `--fewshot_eg`: Use few-shot examples (flag)
- `--temperature`: Sampling temperature (default: 0.0)
- `--max_gen_length`: Maximum generation length (default: 1024)
- `--max_context_length`: Maximum context length (default: 32768)
- `--device`: Device to load model on (default: "auto")
- `--max_memory`: Maximum memory allocation per GPU (e.g., "20GB")

## Input Data Format

The input data should be in JSONL format with the following structure:
```json
{
    "commit_id": "commit_hash",
    "messages": "commit_message_or_description",
    "code_change": "code_to_analyze_for_vulnerabilities",
    "label": 0
}
```

## Output Format

The output will be in JSONL format with the following structure:
```json
{
    "sample_key": "commit_hash",
    "code_change": "code_to_analyze_for_vulnerabilities",
    "messages": "commit_message_or_description",
    "label": 0,
    "prompt": "formatted_prompt_sent_to_model",
    "response": "model_response"
}
```

## Memory Requirements

The Qwen/Qwen3-Coder-480B-A35B-Instruct model is very large and requires significant GPU memory. For systems with limited memory:

1. Use the `--max_memory` parameter to limit memory usage per GPU
2. Consider using model quantization techniques
3. Use smaller Qwen models if the full 480B model is too large

## Notes

- The script automatically handles tokenization and prompt formatting for the Qwen model
- Few-shot examples are based on the same examples used in the OpenAI experiments
- The script includes proper error handling and token truncation for long inputs
- Results are saved incrementally to avoid data loss during long runs