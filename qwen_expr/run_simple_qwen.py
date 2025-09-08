#!/usr/bin/env python3
"""
Simple Qwen runner without complex network configuration.
This version avoids urllib3 compatibility issues.
"""

import os
import json
import argparse
import logging
import time
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from qwen_utils import get_qwen_prompts, format_qwen_messages, extract_qwen_prediction

class SimpleQwenVulnerabilityDetector:
    """Simple vulnerability detector using Qwen models without complex network config."""
    
    def __init__(self, model_name: str, cache_dir: Optional[str] = None, offline: bool = False):
        """Initialize the detector."""
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.offline = offline
        self.tokenizer = None
        self.model = None
        self.prompts = get_qwen_prompts()
        
        # Set basic environment variables for larger timeouts
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '3600'  # 1 hour
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading Qwen model: {self.model_name}")
        
        load_kwargs = {
            "cache_dir": self.cache_dir,
            "local_files_only": self.offline,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        
        # Add quantization for large models
        if "480B" in self.model_name or "405B" in self.model_name:
            print("Using 8-bit quantization for large model")
            load_kwargs["load_in_8bit"] = True
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **load_kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def predict_vulnerability(self, code: str, strategy: str = "cot", fewshot: bool = True) -> Dict:
        """Predict vulnerability for given code."""
        try:
            # Build prompt manually for better control
            prompt_parts = []
            
            # Add system instruction
            prompt_parts.append(f"System: {self.prompts['system']}")
            prompt_parts.append("")
            
            # Add few-shot examples if requested
            if fewshot:
                if strategy == "cot":
                    # CoT few-shot examples
                    prompt_parts.append(f"User: {self.prompts['cot_oneshot_user']}")
                    prompt_parts.append(f"Assistant: {self.prompts['cot_oneshot_assistant']}")
                    prompt_parts.append("")
                    prompt_parts.append(f"User: {self.prompts['cot_twoshot_user']}")
                    prompt_parts.append(f"Assistant: {self.prompts['cot_twoshot_assistant']}")
                    prompt_parts.append("")
                else:
                    # Standard few-shot examples
                    prompt_parts.append(f"User: {self.prompts['oneshot_user']}")
                    prompt_parts.append(f"Assistant: {self.prompts['oneshot_assistant']}")
                    prompt_parts.append("")
                    prompt_parts.append(f"User: {self.prompts['twoshot_user']}")
                    prompt_parts.append(f"Assistant: {self.prompts['twoshot_assistant']}")
                    prompt_parts.append("")
            
            # Add the main prompt
            if strategy == "cot":
                main_prompt = self.prompts['cot'].format(func=code)
            else:
                main_prompt = self.prompts['std_cls'].format(func=code)
            
            prompt_parts.append(f"User: {main_prompt}")
            prompt_parts.append("Assistant: ")
            
            # Combine all parts
            prompt = "\n".join(prompt_parts)
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            
            # Move to device if needed
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Extract prediction
            prediction_str = extract_qwen_prediction(response)
            prediction = 1 if prediction_str == "YES" else 0
            
            return {
                "prediction": prediction,
                "raw_response": response,
                "prediction_str": prediction_str,
                "prompt_strategy": strategy,
                "fewshot": fewshot
            }
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return {
                "prediction": 0,
                "raw_response": f"Error: {str(e)}",
                "prediction_str": "ERROR",
                "prompt_strategy": strategy,
                "fewshot": fewshot
            }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Simple Qwen vulnerability detection")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test data")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder")
    parser.add_argument("--strategy", type=str, default="cot", choices=["cot", "basic"], help="Prompting strategy")
    parser.add_argument("--fewshot", action="store_true", help="Use few-shot prompting")
    parser.add_argument("--cache_dir", type=str, help="Cache directory")
    parser.add_argument("--offline", action="store_true", help="Use offline mode")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Initialize detector
    print("Initializing Simple Qwen detector...")
    detector = SimpleQwenVulnerabilityDetector(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        offline=args.offline
    )
    
    # Load test data
    print(f"Loading test data from: {args.data_path}")
    with open(args.data_path, 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    print(f"Found {len(test_data)} test samples")
    
    # Process samples
    results = []
    correct = 0
    total = 0
    
    for i, sample in enumerate(test_data):
        print(f"\n📝 Processing sample {i+1}/{len(test_data)}")
        
        code = sample.get('func', '')
        true_label = sample.get('target', 0)
        
        # Predict
        result = detector.predict_vulnerability(code, args.strategy, args.fewshot)
        predicted_label = result['prediction']
        
        # Check accuracy
        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1
        total += 1
        
        print(f"True: {true_label}, Predicted: {predicted_label} ({result.get('prediction_str', 'N/A')}), Correct: {is_correct}")
        
        # Store result
        result_entry = {
            "sample_id": i,
            "code": code,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "prediction_str": result.get('prediction_str', 'N/A'),
            "is_correct": is_correct,
            "raw_response": result['raw_response'],
            "prompt_strategy": result['prompt_strategy'],
            "fewshot": result['fewshot']
        }
        results.append(result_entry)
        
        # Save intermediate results every 10 samples
        if (i + 1) % 10 == 0:
            accuracy = correct / total
            print(f"Intermediate accuracy: {accuracy:.3f} ({correct}/{total})")
            
            output_file = os.path.join(args.output_folder, f"simple_qwen_results_intermediate.json")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
    
    # Final results
    accuracy = correct / total if total > 0 else 0
    print(f"\n🎯 Final Results:")
    print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
    
    # Save final results
    output_file = os.path.join(args.output_folder, f"simple_qwen_results_final.json")
    with open(output_file, 'w') as f:
        json.dump({
            "model_name": args.model_name,
            "strategy": args.strategy,
            "fewshot": args.fewshot,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }, f, indent=2)
    
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
