# Chain of Thought + Few-Shot Fine-tuning for Qwen3-Coder-30B-A3B-Instruct

This implementation provides a comprehensive approach to fine-tuning the Qwen3-Coder-30B-A3B-Instruct model for vulnerability detection using Chain of Thought (CoT) reasoning and Few-Shot learning techniques.

## Features

- **Chain of Thought Prompting**: Step-by-step reasoning for vulnerability analysis
- **Few-Shot Learning**: Uses carefully crafted examples to improve model performance
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning
- **Chat Template Support**: Optimized for instruction-tuned models like Qwen
- **Comprehensive Metrics**: Enhanced evaluation with ROC-AUC, PR-AUC, MCC, etc.
- **Wandb Integration**: Full experiment tracking and visualization

## Chain of Thought Examples

The system includes pre-defined few-shot examples covering common vulnerability patterns:

1. **Buffer Overflow Example**: Classic `strcpy()` vulnerability
2. **Integer Overflow Example**: Usage tracking with overflow potential  
3. **Safe Code Example**: Proper memory allocation and bounds checking

Each example includes:
- Code snippet
- Step-by-step analysis reasoning
- Clear vulnerability conclusion (YES/NO)

## Key Files

- `run_ft_qwen_cot_fewshot.py` - Main training script
- `run_qwen_cot_fewshot.sh` - Linux/macOS execution script
- `run_qwen_cot_fewshot.ps1` - Windows PowerShell execution script

## Usage

### 1. Basic Usage

```bash
python os_expr/run_ft_qwen_cot_fewshot.py \
    --project "your_project" \
    --model_dir "qwen_cot_fewshot" \
    --model_name_or_path "Qwen/Qwen2.5-Coder-32B-Instruct" \
    --train_data_file "./data/train.jsonl" \
    --eval_data_file "./data/eval.jsonl" \
    --test_data_file "./data/test.jsonl" \
    --output_dir "./checkpoints" \
    --use_few_shot \
    --use_chat_template \
    --use_lora \
    --do_train \
    --do_test
```

### 2. Using the Convenience Scripts

**Linux/macOS:**
```bash
chmod +x run_qwen_cot_fewshot.sh
./run_qwen_cot_fewshot.sh
```

**Windows PowerShell:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\run_qwen_cot_fewshot.ps1
```

## Configuration Options

### Chain of Thought & Few-Shot Parameters

- `--use_few_shot`: Enable few-shot examples in prompts
- `--use_chat_template`: Use chat template for instruction-tuned models
- `--num_few_shot_examples`: Number of examples to include (default: 3)

### LoRA Parameters

- `--use_lora`: Enable LoRA fine-tuning
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha parameter (default: 32)
- `--lora_dropout`: LoRA dropout rate (default: 0.1)

### Training Parameters

- `--train_batch_size`: Training batch size (default: 1 for large models)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 16)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--epoch`: Number of training epochs (default: 3)
- `--block_size`: Maximum sequence length (default: 2048)

### Wandb Tracking

- `--use_wandb`: Enable Wandb logging
- `--wandb_project`: Wandb project name
- `--wandb_run_name`: Custom run name
- `--wandb_tags`: Tags for the experiment

## Prompt Structure

The system creates prompts in the following format:

```
You are a security expert specializing in static program analysis...

Example 1:
Code:
```
[example code]
```

Analysis:
[step-by-step reasoning]

Conclusion: YES/NO

---

Now analyze this code:
Code:
```
[target code]
```

Please provide step-by-step analysis and conclude with either YES or NO.
Analysis:
```

## Data Format

Input data should be in JSONL format with the following structure:

```json
{
    "commit_id": "unique_identifier",
    "code_change": "source_code_to_analyze", 
    "label": 0,  // 0 for safe, 1 for vulnerable
    "project": "project_name"
}
```

## Output

The system generates:

1. **Model Checkpoints**: Saved in `output_dir/checkpoint-best-f1/project/model_dir/`
2. **Predictions**: Saved as `predictions_qwen_cot.txt` with format:
   ```
   commit_id\tlabel\tconfidence_score
   ```
3. **Metrics**: Comprehensive evaluation including:
   - Accuracy, Precision, Recall, F1-Score
   - True Negative Rate, False Positive Rate, False Negative Rate  
   - Matthews Correlation Coefficient (MCC)
   - ROC-AUC and PR-AUC scores

## Memory Requirements

For Qwen2.5-Coder-32B-Instruct:
- **Full Fine-tuning**: ~64GB+ VRAM
- **LoRA Fine-tuning**: ~24-32GB VRAM (recommended)
- **CPU Memory**: 64GB+ RAM recommended

## Performance Tips

1. **Use LoRA**: Significantly reduces memory requirements
2. **Gradient Accumulation**: Increase `gradient_accumulation_steps` for larger effective batch sizes
3. **Mixed Precision**: Automatically enabled with bf16
4. **Single GPU**: Use `--force_single_gpu` to avoid distributed training overhead

## Example Results

With proper configuration, you can expect:
- **Training Time**: 2-6 hours per epoch (depending on dataset size)
- **Memory Usage**: 24-32GB VRAM with LoRA
- **Performance**: Competitive with state-of-the-art vulnerability detection models

## Troubleshooting

### Common Issues

1. **CUDA OOM**: Reduce `train_batch_size` or increase `gradient_accumulation_steps`
2. **Flash Attention Errors**: Script automatically falls back to eager attention
3. **Tokenizer Issues**: Ensure model path is correct and accessible
4. **Distributed Training**: Use `--force_single_gpu` to disable

### Model Loading Issues

If the model fails to load:
```bash
# Check if model is accessible
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-32B-Instruct')"
```

## Advanced Usage

### Custom Few-Shot Examples

Modify the `FEW_SHOT_EXAMPLES` list in the script to add domain-specific examples:

```python
FEW_SHOT_EXAMPLES = [
    {
        "code": "your_custom_code",
        "reasoning": "step_by_step_analysis", 
        "label": 0,  # 0 for safe, 1 for vulnerable
        "answer": "NO"  # or "YES"
    }
]
```

### Custom Chain of Thought Prompts

Modify the `COT_SYSTEM_PROMPT` and `create_cot_prompt()` function to customize the reasoning approach.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{qwen_cot_vulnerability_detection,
    title={Chain of Thought and Few-Shot Learning for Code Vulnerability Detection with Qwen},
    author={Your Name},
    year={2024},
    url={https://github.com/your-repo/PrimeVul}
}
```
