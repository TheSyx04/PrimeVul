import argparse
import os
import time
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

# For better network handling
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import huggingface_hub

# Import Qwen-specific utilities
from qwen_utils import get_qwen_prompts, format_qwen_messages, extract_qwen_prediction


class QwenVulnerabilityDetector:
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-32B-Instruct", device="auto", 
                 cache_dir=None, offline=False, max_retries=3, force_download=False):
        """
        Initialize the Qwen model for vulnerability detection.
        
        Args:
            model_name: The Qwen model name/path (supports Qwen2.5-Coder series)
            device: Device to load the model on
            cache_dir: Directory to cache models
            offline: Use only cached models
            max_retries: Maximum retry attempts
            force_download: Force re-download
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.offline = offline
        self.max_retries = max_retries
        self.force_download = force_download
        
        # Configure better network settings for large downloads
        if not offline:
            self._setup_network_config()
        
        # Validate and potentially correct model name
        self.model_name = self._validate_model_name(model_name)
        
        print(f"Loading tokenizer for {self.model_name}...")
        try:
            self.tokenizer = self._load_tokenizer_with_retry()
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            if not offline:
                print("Trying alternative model names...")
                self.model_name = self._try_alternative_models()
                self.tokenizer = self._load_tokenizer_with_retry()
            else:
                raise e
        
        # Set pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading model {self.model_name}...")
        print("Note: Large models may take several minutes to download and load...")
        
        # Load model with retry logic
        self.model = self._load_model_with_retry()
        
        print("Model loaded successfully!")
    
    def _setup_network_config(self):
        """Configure network settings for better handling of large downloads."""
        # Set longer timeouts for huggingface_hub
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '3600'  # 1 hour timeout
        
        # Configure requests session with retry logic
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"],
            backoff_factor=2
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Monkey patch the requests session in huggingface_hub
        huggingface_hub.file_download.requests.Session = lambda: session
    
    def _load_tokenizer_with_retry(self):
        """Load tokenizer with retry logic."""
        for attempt in range(self.max_retries):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    padding_side="left",
                    cache_dir=self.cache_dir,
                    local_files_only=self.offline,
                    resume_download=not self.force_download,
                    force_download=self.force_download
                )
                return tokenizer
            except Exception as e:
                print(f"Tokenizer loading attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise e
    
    def _load_model_with_retry(self):
        """Load model with retry logic and progressive fallback."""
        loading_strategies = [
            # Strategy 1: Standard loading with optimizations
            {
                "torch_dtype": torch.float16,
                "device_map": self.device,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "load_in_8bit": "480B" in self.model_name or "72B" in self.model_name,
                "attn_implementation": "flash_attention_2" if torch.cuda.is_available() else "eager",
                "cache_dir": self.cache_dir,
                "local_files_only": self.offline,
                "resume_download": not self.force_download,
                "force_download": self.force_download
            },
            # Strategy 2: Fallback without flash attention
            {
                "torch_dtype": torch.float16,
                "device_map": self.device,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "load_in_8bit": "480B" in self.model_name or "72B" in self.model_name,
                "cache_dir": self.cache_dir,
                "local_files_only": self.offline,
                "resume_download": not self.force_download
            },
            # Strategy 3: Basic loading
            {
                "torch_dtype": torch.float16,
                "device_map": self.device,
                "trust_remote_code": True,
                "cache_dir": self.cache_dir,
                "local_files_only": self.offline,
                "resume_download": not self.force_download
            }
        ]
        
        for strategy_idx, strategy in enumerate(loading_strategies):
            for attempt in range(self.max_retries):
                try:
                    print(f"Loading strategy {strategy_idx + 1}, attempt {attempt + 1}")
                    if strategy.get("load_in_8bit"):
                        print("Using 8-bit quantization for large model")
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        **strategy
                    )
                    return model
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"Loading attempt {attempt + 1} with strategy {strategy_idx + 1} failed: {error_msg}")
                    
                    # Handle specific error types
                    if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                        if attempt < self.max_retries - 1:
                            wait_time = (2 ** attempt) * 10  # Longer waits for network issues
                            print(f"Network issue detected. Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        continue
                    elif "memory" in error_msg.lower() or "cuda" in error_msg.lower():
                        print("Memory issue detected, trying next strategy...")
                        break  # Try next strategy
                    else:
                        if attempt < self.max_retries - 1:
                            wait_time = 2 ** attempt
                            print(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        continue
        
        # If all strategies fail, try alternative models
        if not self.offline:
            print("All loading strategies failed. Trying alternative models...")
            original_model = self.model_name
            self.model_name = self._try_alternative_models()
            print(f"Switched from {original_model} to {self.model_name}")
            
            # Try loading the alternative model
            return AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                cache_dir=self.cache_dir,
                resume_download=not self.force_download
            )
        else:
            raise ValueError("Failed to load model in offline mode. Check cached files.")
    
    def _validate_model_name(self, model_name):
        """Validate and correct model name if needed."""
        # Common corrections for model names
        corrections = {
            "Qwen/QwenCoder-480B-A35B-Instruct": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
            "Qwen/QwenCoder-30B-A3B-Instruct": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "QwenCoder-480B-A35B-Instruct": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
            "QwenCoder-30B-A3B-Instruct": "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        }
        
        if model_name in corrections:
            corrected = corrections[model_name]
            print(f"Correcting model name: {model_name} -> {corrected}")
            return corrected
        
        return model_name
    
    def _try_alternative_models(self):
        """Try alternative model names if the original fails."""
        alternatives = [
            "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "Qwen/Qwen2.5-Coder-32B-Instruct", 
            "Qwen/Qwen2.5-Coder-14B-Instruct",
            "Qwen/Qwen2.5-Coder-7B-Instruct"
        ]
        
        for alt_model in alternatives:
            try:
                print(f"Trying alternative model: {alt_model}")
                # Just test tokenizer loading
                AutoTokenizer.from_pretrained(alt_model, trust_remote_code=True)
                print(f"Successfully found alternative model: {alt_model}")
                return alt_model
            except:
                continue
        
        raise ValueError("No compatible Qwen models found. Please check model availability and network connection.")
    
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
                       help='Qwen model name or path. For 480B model use: Qwen/Qwen3-Coder-480B-A35B-Instruct')
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
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Directory to cache downloaded models')
    parser.add_argument('--offline', action='store_true',
                       help='Use only locally cached models (no internet download)')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='Maximum number of retries for model loading')
    parser.add_argument('--force_download', action='store_true',
                       help='Force re-download of model files')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Initialize model
    print("Initializing Qwen model...")
    print(f"Using model: {args.model_name}")
    print(f"Cache directory: {args.cache_dir or 'default'}")
    print(f"Offline mode: {args.offline}")
    print(f"Max retries: {args.max_retries}")
    detector = QwenVulnerabilityDetector(
        model_name=args.model_name, 
        device=args.device,
        cache_dir=args.cache_dir,
        offline=args.offline,
        max_retries=args.max_retries,
        force_download=args.force_download
    )
    
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
