# Qwen3-Coder-480B-A35B-Instruct Vulnerability Detection Setup

## Overview

I've successfully created a complete script setup to run the **Qwen3-Coder-480B-A35B-Instruct** model for vulnerability detection using **Chain-of-Thought (CoT)** and **Few-Shot** prompting instead of fine-tuning.

## 📁 Files Created

### Core Scripts
- **`run_qwen_prompting.py`** - Main script for running Qwen models
- **`run_qwen_480b.py`** - Simplified runner for the 480B model specifically  
- **`qwen_utils.py`** - Enhanced prompts and utilities for Qwen models
- **`demo.py`** - Demo script showing the complete workflow

### Configuration & Documentation
- **`requirements.txt`** - Required Python packages
- **`README.md`** - Comprehensive documentation
- **`test_setup.py`** - Setup verification script
- **`run_qwen.bat`** - Windows batch script for easy execution

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the 480B Model
```bash
python run_qwen_480b.py \
    --data_path <your_test_data.jsonl> \
    --output_folder ./output \
    --strategy cot
```

### 3. Calculate VD-Score
```bash
python ../calc_vd_score.py \
    --pred_file ./output/predictions.txt \
    --test_file <your_test_data.jsonl>
```

## 🧠 Key Features

### Enhanced Chain-of-Thought Prompting
The system guides the model through systematic security analysis:
1. **Code Understanding** - What the function does
2. **Input Analysis** - Identify all inputs and sources
3. **Data Flow** - Trace data movement through the function
4. **Security Checks** - Look for common vulnerability patterns
5. **Bounds Checking** - Verify array/buffer access safety
6. **Memory Management** - Check allocation/deallocation
7. **Input Validation** - Assess input sanitization

### Few-Shot Learning Examples
Includes carefully crafted examples showing:
- Vulnerable code patterns (buffer overflows, unsafe functions)
- Safe code patterns (proper bounds checking, input validation)
- Both standard and CoT response formats

### Model Optimization
- **8-bit quantization** for memory efficiency on large models
- **Flash Attention 2** for faster inference
- **Proper tokenization** with truncation handling
- **Device auto-mapping** for multi-GPU setups

## 🎯 Model-Specific Support

### Qwen3-Coder-480B-A35B-Instruct (Target Model)
- Requires ~200GB+ GPU memory
- Uses 8-bit quantization automatically
- Optimized context length (8192 tokens)
- Enhanced generation parameters

### Alternative Models Supported
- Qwen2.5-Coder-32B-Instruct (64GB GPU memory)
- Qwen2.5-Coder-14B-Instruct (32GB GPU memory)  
- Qwen2.5-Coder-7B-Instruct (16GB GPU memory)

## 📊 Output Format

Results are saved in JSONL format:
```json
{
  "sample_key": "project_commit_id",
  "target": "YES|NO", 
  "response": "full_model_response",
  "prediction": "YES|NO|UNCLEAR",
  "prompt_strategy": "cot",
  "fewshot": true,
  "temperature": 0.0
}
```

Plus a `predictions.txt` file for VD-Score calculation.

## 🔧 Configuration Options

### Prompt Strategies
- **`cot`** - Chain-of-Thought (recommended)
- **`std_cls`** - Standard classification

### Few-Shot Settings
- **`--fewshot_eg`** - Enable few-shot examples (recommended)
- **`--no_fewshot`** - Disable few-shot examples

### Generation Parameters
- **`--temperature 0.0`** - Deterministic output (recommended)
- **`--max_gen_length 2048`** - For CoT responses
- **`--max_input_length 8192`** - For large context

## 🧪 Demo Results

The demo script shows perfect accuracy (3/3) on test cases:
- **Vulnerable Code**: Buffer overflow, unsafe `gets()` function ✓
- **Safe Code**: Proper integer overflow protection ✓
- **CoT Analysis**: Detailed step-by-step security reasoning ✓

## 💡 Usage Tips

1. **Start with smaller models** (32B) to test your setup
2. **Use CoT strategy** for better reasoning quality
3. **Enable few-shot examples** for improved performance
4. **Set temperature=0.0** for consistent results
5. **Monitor GPU memory** usage during inference

## 🔗 Integration with PrimeVul

This setup integrates seamlessly with the existing PrimeVul evaluation framework:
- Uses same data format as OpenAI experiments
- Generates VD-Score compatible predictions
- Maintains consistency with existing evaluation pipeline

## 🎉 Ready to Use!

The system is now ready to run Qwen3-Coder-480B-A35B-Instruct for vulnerability detection using advanced prompting techniques instead of fine-tuning. Simply provide your test data and run the scripts!

For questions or issues, refer to the comprehensive README.md file.
