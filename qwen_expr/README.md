# Qwen Model Vulnerability Detection

This directory contains scripts to run Qwen Code models for vulnerability detection using chain-of-thought (CoT) and few-shot prompting instead of fine-tuning.

## Supported Models

- **Qwen2.5-Coder-32B-Instruct** (default for testing)
- **QwenCoder-480B-A35B-Instruct** (target model - requires significant GPU resources)
- Other Qwen2.5-Coder models (7B, 14B, 72B)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Qwen3-Coder-480B-A35B-Instruct

For the specific model you requested:

```bash
python run_qwen_480b.py \
    --data_path <path_to_test_data.jsonl> \
    --output_folder ./output \
    --strategy cot \
    --temperature 0.0
```

### 3. Run with Different Strategies

#### Chain-of-Thought (Recommended)
```bash
python run_qwen_prompting.py \
    --model_name Qwen/QwenCoder-480B-A35B-Instruct \
    --prompt_strategy cot \
    --data_path <path_to_test_data.jsonl> \
    --output_folder ./output \
    --fewshot_eg \
    --temperature 0.0 \
    --max_gen_length 2048
```

#### Standard Classification
```bash
python run_qwen_prompting.py \
    --model_name Qwen/QwenCoder-480B-A35B-Instruct \
    --prompt_strategy std_cls \
    --data_path <path_to_test_data.jsonl> \
    --output_folder ./output \
    --fewshot_eg \
    --temperature 0.0
```

## Files Description

- **`run_qwen_prompting.py`**: Main script for running Qwen models with various configurations
- **`run_qwen_480b.py`**: Simplified runner specifically for the 480B model
- **`qwen_utils.py`**: Utility functions and enhanced prompts for Qwen models
- **`requirements.txt`**: Required Python packages

## Key Features

### Enhanced Chain-of-Thought Prompting
The CoT strategy guides the model through a systematic security analysis:
1. Code Understanding
2. Input Analysis
3. Data Flow Tracing
4. Security Pattern Detection
5. Bounds Checking
6. Memory Management Analysis
7. Input Validation Assessment

### Few-Shot Learning
Includes carefully selected examples of both vulnerable and safe code to improve model performance.

### Optimized for Large Models
- Supports 8-bit quantization for memory efficiency
- Flash Attention 2 for faster inference
- Proper tokenization and truncation handling

## Model-Specific Configurations

### For 480B Model (Requires ~200GB+ GPU Memory)
```bash
python run_qwen_prompting.py \
    --model_name Qwen/QwenCoder-480B-A35B-Instruct \
    --device auto \
    --max_input_length 8192 \
    --max_gen_length 2048
```

### For 32B Model (Requires ~64GB GPU Memory)
```bash
python run_qwen_prompting.py \
    --model_name Qwen/Qwen2.5-Coder-32B-Instruct \
    --device auto \
    --max_input_length 4096 \
    --max_gen_length 1024
```

## Output Format

Results are saved in JSONL format with the following structure:
```json
{
  "sample_key": "project_commit_id",
  "target": "YES|NO",
  "response": "model_generated_response",
  "prediction": "YES|NO|UNCLEAR",
  "prompt_strategy": "cot|std_cls",
  "fewshot": true,
  "temperature": 0.0
}
```

A `predictions.txt` file is also generated for VD-Score calculation.

## Performance Considerations

### Memory Requirements
- **480B Model**: Requires multiple high-end GPUs (A100 80GB x 4+)
- **72B Model**: Requires 1-2 high-end GPUs
- **32B Model**: Requires 1 high-end GPU
- **14B Model**: Can run on consumer GPUs

### Optimization Tips
- Use `load_in_8bit=True` for memory-constrained setups
- Adjust `max_input_length` based on available memory
- Use `temperature=0.0` for deterministic results
- Enable Flash Attention 2 for faster inference

## Example Usage

```bash
# Check requirements
python run_qwen_480b.py --check_only

# Run with CoT and few-shot (recommended)
python run_qwen_480b.py \
    --data_path ../data/primevul_test.jsonl \
    --output_folder ./results \
    --strategy cot

# Run without few-shot examples
python run_qwen_480b.py \
    --data_path ../data/primevul_test.jsonl \
    --output_folder ./results \
    --strategy cot \
    --no_fewshot

# Calculate VD-Score
python ../calc_vd_score.py \
    --pred_file ./results/predictions.txt \
    --test_file ../data/primevul_test.jsonl
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `max_input_length` and `max_gen_length`
- Enable 8-bit quantization
- Use smaller model variant

### Model Loading Issues
- Ensure you have sufficient disk space for model download
- Check Hugging Face access permissions for gated models
- Verify transformers version compatibility

### Slow Inference
- Enable Flash Attention 2
- Use appropriate batch size (currently supports batch_size=1)
- Consider using multiple GPUs with device_map="auto"

## Results Analysis

The script outputs:
1. **Accuracy**: Basic binary classification accuracy
2. **Prediction Distribution**: Count of YES/NO/UNCLEAR predictions
3. **VD-Score Ready**: Formatted predictions for vulnerability detection scoring

For comprehensive evaluation, use the provided VD-Score calculation script with the generated predictions.
