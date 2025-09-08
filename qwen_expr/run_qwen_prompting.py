import argparse
import os
import time
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

# Import Qwen-specific utilities
from qwen_utils import get_qwen_prompts, format_qwen_messages, extract_qwen_prediction


class QwenVulnerabilityDetector:
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-32B-Instruct", device="auto"):
        """
        Initialize the Qwen model for vulnerability detection.
        
        Args:
            model_name: The Qwen model name/path (supports Qwen2.5-Coder series)
            device: Device to load the model on
        """
        self.model_name = model_name
        self.device = device
        
        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Set pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading model {model_name}...")
        
        # Use different settings based on model size
        if "480B" in model_name or "72B" in model_name:
            # For very large models, use 8-bit quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
                load_in_8bit=True,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
            )
        elif "32B" in model_name or "14B" in model_name:
            # For large models, use standard loading
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
            )
        else:
            # For smaller models, use full precision if possible
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
            )
        
        print("Model loaded successfully!")
    
    def format_chat_messages(self, messages):
        """
        Format messages for Qwen chat template.
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def generate_response(self, prompt, args):
        """
        Generate response from Qwen model using enhanced prompting.
        
        Args:
            prompt: Dict containing the prompt information
            args: Arguments containing generation parameters
            
        Returns:
            Generated response text
        """
        # Use Qwen-specific message formatting
        messages = format_qwen_messages(prompt, args.prompt_strategy, args.fewshot_eg)
        
        # Format the conversation
        formatted_prompt = self.format_chat_messages(messages)
        
        # Tokenize with proper truncation for large models
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_length,
            padding=True
        ).to(self.model.device)
        
        # Generate with optimized parameters for Qwen
        generation_config = {
            "max_new_tokens": args.max_gen_length,
            "temperature": args.temperature,
            "do_sample": args.temperature > 0,
            "top_p": 0.9 if args.temperature > 0 else None,
            "top_k": 50 if args.temperature > 0 else None,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.05,
            "length_penalty": 1.0,
        }
        
        # Add specific settings for CoT
        if args.prompt_strategy == "cot":
            generation_config["max_new_tokens"] = min(2048, args.max_gen_length * 2)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **generation_config
            )
        
        # Decode response (excluding the input prompt)
        input_length = inputs.input_ids.shape[1]
        response_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        return response.strip()


def construct_prompts(input_file, inst):
    """
    Construct prompts from input data file.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        samples = f.readlines()
    
    samples = [json.loads(sample) for sample in samples]
    prompts = []
    
    for sample in samples:
        key = sample["project"] + "_" + sample["commit_id"]
        p = {"sample_key": key}
        p["func"] = sample["func"]
        p["target"] = sample["target"]
        # Note: prompt will be constructed in format_qwen_messages
        prompts.append(p)
    
    return prompts


def extract_prediction(response):
    """
    Extract YES/NO prediction from model response using enhanced Qwen extraction.
    """
    return extract_qwen_prediction(response)


def main():
    parser = argparse.ArgumentParser(description="Run Qwen model for vulnerability detection using CoT and few-shot prompting")
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct", 
                       help='Qwen model name or path. For 480B model use: Qwen/QwenCoder-480B-A35B-Instruct')
    parser.add_argument('--device', type=str, default="auto", 
                       help='Device to load model on (auto, cuda, cpu)')
    
    # Prompting arguments
    parser.add_argument('--prompt_strategy', type=str, choices=["std_cls", "cot"], default="cot", 
                       help='Prompt strategy: std_cls for standard classification, cot for chain-of-thought')
    parser.add_argument('--fewshot_eg', action="store_true", 
                       help='Use few-shot examples')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test data file (JSONL format)')
    parser.add_argument('--output_folder', type=str, required=True,
                       help='Output folder for results')
    
    # Generation arguments
    parser.add_argument('--temperature', type=float, default=0.0, 
                       help='Sampling temperature (0.0 for greedy decoding)')
    parser.add_argument('--max_gen_length', type=int, default=1024,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--max_input_length', type=int, default=4096,
                       help='Maximum input sequence length')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing (currently only supports 1)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Initialize model
    print("Initializing Qwen model...")
    detector = QwenVulnerabilityDetector(model_name=args.model_name, device=args.device)
    
    # Prepare output file name
    model_name_clean = args.model_name.replace("/", "_").replace("\\", "_")
    output_file = os.path.join(
        args.output_folder, 
        f"{model_name_clean}_{args.prompt_strategy}_fewshot{args.fewshot_eg}_temp{args.temperature}.jsonl"
    )
    
    # Select instruction template (not needed as prompts are built dynamically)
    # prompts = construct_prompts(args.data_path, None)
    
    # Construct prompts
    print(f"Loading prompts from {args.data_path}...")
    prompts = construct_prompts(args.data_path, None)
    print(f"Loaded {len(prompts)} samples")
    
    # Process prompts
    predictions = []
    with open(output_file, "w", encoding="utf-8") as f:
        print(f"Processing {len(prompts)} prompts with {args.model_name}...")
        print(f"Strategy: {args.prompt_strategy}, Few-shot: {args.fewshot_eg}, Temperature: {args.temperature}")
        
        for i, prompt in enumerate(tqdm(prompts, desc="Processing")):
            try:
                # Generate response
                response = detector.generate_response(prompt, args)
                
                # Extract prediction
                prediction = extract_prediction(response)
                
                # Store results
                result = {
                    "sample_key": prompt["sample_key"],
                    "target": prompt["target"],
                    "response": response,
                    "prediction": prediction,
                    "prompt_strategy": args.prompt_strategy,
                    "fewshot": args.fewshot_eg,
                    "temperature": args.temperature
                }
                
                predictions.append(result)
                
                # Write to file
                f.write(json.dumps(result, ensure_ascii=False))
                f.write("\n")
                f.flush()
                
                # Print progress
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(prompts)} samples")
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # Write error result
                error_result = {
                    "sample_key": prompt["sample_key"],
                    "target": prompt["target"],
                    "response": f"ERROR: {str(e)}",
                    "prediction": "ERROR",
                    "prompt_strategy": args.prompt_strategy,
                    "fewshot": args.fewshot_eg,
                    "temperature": args.temperature
                }
                f.write(json.dumps(error_result, ensure_ascii=False))
                f.write("\n")
                f.flush()
    
    print(f"Processing complete! Results saved to {output_file}")
    
    # Print summary statistics
    correct_predictions = sum(1 for p in predictions if p["prediction"] == p["target"])
    total_predictions = len([p for p in predictions if p["prediction"] != "ERROR"])
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\nSummary:")
        print(f"Total samples: {len(prompts)}")
        print(f"Successful predictions: {total_predictions}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.4f}")
    
    # Create predictions.txt file for VD-Score calculation
    pred_file = os.path.join(args.output_folder, "predictions.txt")
    with open(pred_file, "w") as f:
        for result in predictions:
            if result["prediction"] == "YES":
                f.write("1\n")
            elif result["prediction"] == "NO":
                f.write("0\n")
            else:
                f.write("0\n")  # Default to 0 for unclear/error cases
    
    print(f"Predictions file saved to {pred_file}")
    print("You can now calculate VD-Score using: python calc_vd_score.py --pred_file predictions.txt --test_file <test_file>")


if __name__ == "__main__":
    main()
