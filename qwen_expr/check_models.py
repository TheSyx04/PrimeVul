#!/usr/bin/env python3
"""
Script to check available Qwen models and suggest correct model names.
"""

import requests
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore")


def check_huggingface_model(model_name):
    """Check if a model exists on Hugging Face."""
    try:
        # Try to access model info
        url = f"https://huggingface.co/api/models/{model_name}"
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    except:
        return False


def test_model_loading(model_name):
    """Test if we can load the tokenizer for a model."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return True, "Success"
    except Exception as e:
        return False, str(e)


def main():
    print("Qwen Model Availability Checker")
    print("=" * 40)
    
    # List of potential Qwen Coder models
    potential_models = [
        "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct", 
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/QwenCoder-480B-A35B-Instruct",  # Incorrect name
        "Qwen/QwenCoder-30B-A3B-Instruct",   # Incorrect name
    ]
    
    print("Checking model availability:")
    print()
    
    available_models = []
    
    for model in potential_models:
        print(f"Checking {model}...")
        
        # Check if model exists on HF
        exists = check_huggingface_model(model)
        if exists:
            # Try loading tokenizer
            can_load, error = test_model_loading(model)
            if can_load:
                print(f"  ✓ Available and loadable")
                available_models.append(model)
            else:
                print(f"  ⚠ Exists but cannot load tokenizer: {error[:50]}...")
        else:
            print(f"  ✗ Not found on Hugging Face")
        print()
    
    print("Summary:")
    print("-" * 20)
    if available_models:
        print("Available Qwen Coder models:")
        for i, model in enumerate(available_models, 1):
            print(f"{i}. {model}")
        
        print()
        print("Recommended usage:")
        print("For 480B model:")
        if "Qwen/Qwen3-Coder-480B-A35B-Instruct" in available_models:
            print("  python run_qwen_prompting.py --model_name Qwen/Qwen3-Coder-480B-A35B-Instruct")
        else:
            print("  480B model not available")
        
        print("For smaller testing:")
        smaller_models = [m for m in available_models if any(size in m for size in ["32B", "30B", "14B", "7B"])]
        if smaller_models:
            print(f"  python run_qwen_prompting.py --model_name {smaller_models[0]}")
    else:
        print("No Qwen Coder models found. Please check:")
        print("1. Internet connection")
        print("2. Hugging Face access")
        print("3. Model permissions (some models may be gated)")
    
    print()
    print("Model name corrections:")
    corrections = {
        "Qwen/QwenCoder-480B-A35B-Instruct": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "Qwen/QwenCoder-30B-A3B-Instruct": "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    }
    
    for wrong, correct in corrections.items():
        print(f"  {wrong} → {correct}")


if __name__ == "__main__":
    main()
