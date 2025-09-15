import argparse
import os
import time
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import SYS_INST, PROMPT_INST, PROMPT_INST_COT, ONESHOT_ASSISTANT, ONESHOT_USER, TWOSHOT_USER, TWOSHOT_ASSISTANT


class QwenModel:
    def __init__(self, model_name, device="auto", max_memory=None):
        """
        Initialize Qwen model for few-shot prompting.
        
        Args:
            model_name: Name of the Qwen model (e.g., "Qwen/Qwen3-480B-A35B-Instruct")
            device: Device to load the model on
            max_memory: Maximum memory allocation for model loading
        """
        self.model_name = model_name
        self.device = device
        
        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        print(f"Loading model {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            max_memory=max_memory
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def count_tokens(self, text):
        """Count the number of tokens in the text."""
        return len(self.tokenizer.encode(text))
    
    def truncate_messages(self, messages, max_tokens):
        """
        Truncate messages if they exceed the maximum token limit.
        """
        total_tokens = 0
        for message in messages:
            total_tokens += self.count_tokens(message["content"])
        
        if total_tokens <= max_tokens:
            return messages
        
        # If truncation is needed, prioritize keeping the system message and user query
        # Truncate the few-shot examples first
        truncated_messages = []
        remaining_tokens = max_tokens
        
        # Always keep system message
        if messages and messages[0]["role"] == "system":
            truncated_messages.append(messages[0])
            remaining_tokens -= self.count_tokens(messages[0]["content"])
        
        # Always keep the last user message (the actual query)
        if messages and messages[-1]["role"] == "user":
            user_tokens = self.count_tokens(messages[-1]["content"])
            if user_tokens <= remaining_tokens:
                last_message = messages[-1]
                remaining_tokens -= user_tokens
            else:
                # Truncate the user message if it's too long
                content = messages[-1]["content"]
                encoded = self.tokenizer.encode(content)
                truncated_encoded = encoded[:remaining_tokens]
                truncated_content = self.tokenizer.decode(truncated_encoded)
                last_message = {"role": "user", "content": truncated_content}
                remaining_tokens = 0
        
        # Add few-shot examples if there's space
        if len(messages) > 2:  # If there are few-shot examples
            for message in messages[1:-1]:  # Skip system and last user message
                message_tokens = self.count_tokens(message["content"])
                if message_tokens <= remaining_tokens:
                    truncated_messages.append(message)
                    remaining_tokens -= message_tokens
                else:
                    break
        
        # Add the last user message
        if 'last_message' in locals():
            truncated_messages.append(last_message)
        
        return truncated_messages


def get_qwen_response(prompt, args, model):
    """
    Get response from Qwen model using few-shot prompting.
    """
    if args.fewshot_eg:
        messages = [
            {"role": "system", "content": SYS_INST},
            {"role": "user", "content": ONESHOT_USER},
            {"role": "assistant", "content": ONESHOT_ASSISTANT},
            {"role": "user", "content": TWOSHOT_USER},
            {"role": "assistant", "content": TWOSHOT_ASSISTANT},
            {"role": "user", "content": prompt["prompt"]}
        ]
    else:
        messages = [
            {"role": "system", "content": SYS_INST},
            {"role": "user", "content": prompt["prompt"]}
        ]
    
    # Calculate max tokens for input (leaving space for generation)
    max_input_tokens = args.max_context_length - args.max_gen_length
    messages = model.truncate_messages(messages, max_input_tokens)
    
    try:
        # Convert messages to prompt format for Qwen
        formatted_prompt = model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize the input
        inputs = model.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens
        ).to(model.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.model.generate(
                **inputs,
                max_new_tokens=args.max_gen_length,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
                pad_token_id=model.tokenizer.eos_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode the response
        generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        response = model.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    except Exception as e:
        print(f"Error generating response: {e}")
        return "ERROR"


def construct_prompts(input_file, inst):
    """
    Construct prompts from input data file.
    """
    with open(input_file, "r") as f:
        samples = f.readlines()
    samples = [json.loads(sample) for sample in samples]
    prompts = []
    for sample in samples:
        key = sample["commit_id"]
        p = {"sample_key": key}
        p["code_change"] = sample["code_change"]
        p["messages"] = sample["messages"]
        p["label"] = sample["label"]
        p["prompt"] = inst.format(func=sample["code_change"])
        prompts.append(p)
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-480B-A35B-Instruct", help='Model name')
    parser.add_argument('--prompt_strategy', type=str, choices=["std_cls", "cot"], default="std_cls", help='Prompt strategy')
    parser.add_argument('--data_path', type=str, required=True, help='Data path')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--max_gen_length', type=int, default=1024, help='Maximum generation length')
    parser.add_argument('--max_context_length', type=int, default=32768, help='Maximum context length')
    parser.add_argument('--fewshot_eg', action="store_true", help='Use few-shot examples')
    parser.add_argument('--device', type=str, default="auto", help='Device to load model on')
    parser.add_argument('--max_memory', type=str, default=None, help='Maximum memory allocation (e.g., "20GB")')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Process max_memory argument
    max_memory = None
    if args.max_memory:
        max_memory = {i: args.max_memory for i in range(torch.cuda.device_count())}
    
    # Initialize model
    print("Initializing Qwen model...")
    model = QwenModel(args.model, device=args.device, max_memory=max_memory)
    
    # Set up output file
    model_name_safe = args.model.replace("/", "_")
    output_file = os.path.join(
        args.output_folder, 
        f"{model_name_safe}_{args.prompt_strategy}_fewshoteg{args.fewshot_eg}.jsonl"
    )
    
    # Select prompt instruction
    if args.prompt_strategy == "std_cls":
        inst = PROMPT_INST
    elif args.prompt_strategy == "cot":
        inst = PROMPT_INST_COT
    else:
        raise ValueError("Invalid prompt strategy")
    
    # Construct prompts
    prompts = construct_prompts(args.data_path, inst)
    
    # Process prompts and generate responses
    with open(output_file, "w") as f:
        print(f"Requesting {args.model} to respond to {len(prompts)} prompts...")
        for p in tqdm(prompts):
            response = get_qwen_response(p, args, model)
            p["response"] = response
            f.write(json.dumps(p))
            f.write("\n")
            f.flush()
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()