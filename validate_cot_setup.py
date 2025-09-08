#!/usr/bin/env python3
"""
Validation script for Chain of Thought + Few-Shot prompting
Tests prompt generation and tokenization without full training
"""

import argparse
import json
from transformers import AutoTokenizer

# Import the CoT functions from the main script
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Define the constants here since we can't import from the main script easily
COT_SYSTEM_PROMPT = """You are a security expert specializing in static program analysis and vulnerability detection. 
Your task is to analyze code snippets and determine if they contain security vulnerabilities. 
Always think step-by-step and provide clear reasoning for your analysis."""

FEW_SHOT_EXAMPLES = [
    {
        "code": """static char *clean_path(char *path)
{
        char *ch;
        char *ch2;
        char *str;
        str = xmalloc(strlen(path) + 1);
        ch = path;
        ch2 = str;
        while (true) {
                *ch2 = *ch;
                ch++;
                ch2++;
                if (!*(ch-1))
                        break;
                while (*(ch - 1) == '/' && *ch == '/')
                        ch++;
        }
        /* get rid of trailing / characters */
        while ((ch = strrchr(str, '/'))) {
                if (ch == str)
                        break;
                if (!*(ch+1))
                        *ch = 0;
                else
                        break;
        }
        return str;
}""",
        "reasoning": """Let me analyze this code step by step:
1. The function allocates memory using xmalloc() for a cleaned path
2. It iterates through the input path character by character
3. It removes duplicate forward slashes
4. It removes trailing forward slashes
5. Memory allocation size is strlen(path) + 1, which is appropriate
6. No buffer overflow risks as the destination buffer is allocated correctly
7. No unchecked input validation issues
8. The logic handles edge cases properly""",
        "label": 0,  # No vulnerability
        "answer": "NO"
    },
    {
        "code": """int64 ClientUsageTracker::GetCachedHostUsage(const std::string& host) {
   HostUsageMap::const_iterator found = cached_usage_.find(host);
   if (found == cached_usage_.end())
     return 0;

  int64 usage = 0;
  const UsageMap& map = found->second;
  for (UsageMap::const_iterator iter = map.begin();
       iter != map.end(); ++iter) {
    usage += iter->second;
  }
  return usage;
}""",
        "reasoning": """Let me analyze this code step by step:
1. This function calculates cached host usage by iterating through a usage map
2. In the for loop, it performs usage += iter->second repeatedly
3. The return type is int64, but there's no overflow checking
4. If iter->second contains large values or many iterations occur, integer overflow can happen
5. Integer overflow in usage calculation could lead to incorrect results
6. This could potentially be exploited to bypass usage limits or quotas
7. The lack of overflow protection makes this a security vulnerability""",
        "label": 1,  # Vulnerability detected
        "answer": "YES"
    },
    {
        "code": """void process_user_input(char *input) {
    char buffer[256];
    strcpy(buffer, input);
    printf("Processing: %s\\n", buffer);
}""",
        "reasoning": """Let me analyze this code step by step:
1. A fixed-size buffer of 256 characters is declared
2. strcpy() is used to copy user input into the buffer
3. strcpy() does not check the length of the source string
4. If input is longer than 255 characters (plus null terminator), buffer overflow occurs
5. Buffer overflow can overwrite adjacent memory locations
6. This can lead to code execution, denial of service, or other security issues
7. This is a classic buffer overflow vulnerability""",
        "label": 1,  # Vulnerability detected  
        "answer": "YES"
    }
]

def create_cot_prompt(code, include_examples=True):
    """Create a Chain of Thought prompt with few-shot examples"""
    prompt = COT_SYSTEM_PROMPT + "\n\n"
    
    if include_examples:
        prompt += "Here are some examples of how to analyze code:\n\n"
        
        for i, example in enumerate(FEW_SHOT_EXAMPLES, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Code:\n```\n{example['code']}\n```\n\n"
            prompt += f"Analysis:\n{example['reasoning']}\n\n"
            prompt += f"Conclusion: {example['answer']}\n\n"
            prompt += "---\n\n"
    
    prompt += "Now analyze this code:\n"
    prompt += f"Code:\n```\n{code}\n```\n\n"
    prompt += "Please provide step-by-step analysis and conclude with either YES (vulnerability detected) or NO (no vulnerability).\n"
    prompt += "Analysis:"
    
    return prompt

def test_prompt_generation():
    """Test prompt generation with sample code"""
    sample_code = """
int copy_data(char* dest, char* src, int size) {
    for(int i = 0; i < size; i++) {
        dest[i] = src[i];
    }
    return size;
}
"""
    
    print("Testing Chain of Thought Prompt Generation")
    print("=" * 60)
    
    # Test with few-shot examples
    print("1. WITH Few-Shot Examples:")
    print("-" * 30)
    prompt_with_examples = create_cot_prompt(sample_code, include_examples=True)
    print(f"Prompt length: {len(prompt_with_examples)} characters")
    print(f"First 500 characters:\n{prompt_with_examples[:500]}...")
    print()
    
    # Test without few-shot examples
    print("2. WITHOUT Few-Shot Examples:")
    print("-" * 30)
    prompt_without_examples = create_cot_prompt(sample_code, include_examples=False)
    print(f"Prompt length: {len(prompt_without_examples)} characters")
    print(f"Full prompt:\n{prompt_without_examples}")
    print()

def test_tokenization(model_name="Qwen/Qwen2.5-Coder-32B-Instruct"):
    """Test tokenization with the Qwen model"""
    print("Testing Tokenization")
    print("=" * 60)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"Successfully loaded tokenizer: {model_name}")
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token to eos_token")
        
        sample_code = """
void unsafe_copy(char* dest, char* src) {
    strcpy(dest, src);  // No bounds checking!
}
"""
        
        # Test prompt creation and tokenization
        prompt = create_cot_prompt(sample_code, include_examples=True)
        
        # Test regular encoding
        print("\n1. Regular Encoding:")
        encoded = tokenizer.encode(prompt, max_length=2048, truncation=True)
        print(f"Token count: {len(encoded)}")
        print(f"First 10 tokens: {encoded[:10]}")
        
        # Test chat template if available
        if hasattr(tokenizer, 'apply_chat_template'):
            print("\n2. Chat Template Encoding:")
            messages = [
                {"role": "system", "content": COT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt.replace(COT_SYSTEM_PROMPT + "\n\n", "")}
            ]
            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                chat_encoded = tokenizer.encode(formatted_prompt, max_length=2048, truncation=True)
                print(f"Chat template token count: {len(chat_encoded)}")
                print(f"First 200 chars of formatted prompt:\n{formatted_prompt[:200]}...")
            except Exception as e:
                print(f"Chat template encoding failed: {e}")
        else:
            print("\n2. Chat template not available for this tokenizer")
        
        print(f"\nTokenizer vocab size: {tokenizer.vocab_size}")
        print(f"Model max length: {getattr(tokenizer, 'model_max_length', 'Unknown')}")
        
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        print("Make sure you have the transformers library installed and internet access")

def test_data_format():
    """Test data format parsing"""
    print("Testing Data Format")
    print("=" * 60)
    
    sample_data = [
        {
            "commit_id": "abc123",
            "code_change": "void test() { char buf[10]; strcpy(buf, input); }",
            "label": 1,
            "project": "test_project"
        },
        {
            "commit_id": "def456", 
            "code_change": "void safe_test() { char buf[10]; strncpy(buf, input, 9); buf[9] = '\\0'; }",
            "label": 0,
            "project": "test_project"
        }
    ]
    
    print("Sample data format:")
    for i, item in enumerate(sample_data, 1):
        print(f"\nSample {i}:")
        print(json.dumps(item, indent=2))
        
        # Test prompt generation
        prompt = create_cot_prompt(item["code_change"], include_examples=False)
        print(f"Generated prompt length: {len(prompt)} characters")

def main():
    parser = argparse.ArgumentParser(description="Validate CoT + Few-Shot setup")
    parser.add_argument("--test-prompts", action="store_true", help="Test prompt generation")
    parser.add_argument("--test-tokenization", action="store_true", help="Test tokenization")
    parser.add_argument("--test-data", action="store_true", help="Test data format")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-32B-Instruct", 
                       help="Model name for tokenization test")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if args.all or not any([args.test_prompts, args.test_tokenization, args.test_data]):
        args.test_prompts = True
        args.test_tokenization = True  
        args.test_data = True
    
    if args.test_prompts:
        test_prompt_generation()
        print()
    
    if args.test_tokenization:
        test_tokenization(args.model_name)
        print()
    
    if args.test_data:
        test_data_format()
        print()
    
    print("Validation completed!")

if __name__ == "__main__":
    main()
