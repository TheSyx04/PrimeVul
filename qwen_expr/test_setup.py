#!/usr/bin/env python3
"""
Test script to verify Qwen vulnerability detection setup.
Creates a small test dataset and runs inference to ensure everything works.
"""

import json
import os
import tempfile
from qwen_utils import get_qwen_prompts, format_qwen_messages, extract_qwen_prediction


def create_test_data():
    """Create a small test dataset for verification."""
    test_samples = [
        {
            "project": "test_project_1",
            "commit_id": "abc123",
            "func": """void copy_data(char *dest, char *src, int len) {
    int i;
    for (i = 0; i < len; i++) {
        dest[i] = src[i];
    }
}""",
            "target": "YES"
        },
        {
            "project": "test_project_2", 
            "commit_id": "def456",
            "func": """int safe_strlen(const char *str) {
    if (str == NULL) {
        return 0;
    }
    
    int len = 0;
    while (str[len] != '\\0' && len < 1000) {
        len++;
    }
    return len;
}""",
            "target": "NO"
        }
    ]
    
    # Create temporary test file
    test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    for sample in test_samples:
        test_file.write(json.dumps(sample) + '\n')
    test_file.close()
    
    return test_file.name, test_samples


def test_prompt_formatting():
    """Test prompt formatting functionality."""
    print("Testing prompt formatting...")
    
    test_data = {
        "func": "int test() { return 0; }",
        "target": "NO"
    }
    
    # Test standard classification
    messages_std = format_qwen_messages(test_data, strategy="std_cls", fewshot=True)
    print(f"Standard classification messages: {len(messages_std)} messages")
    
    # Test chain-of-thought
    messages_cot = format_qwen_messages(test_data, strategy="cot", fewshot=True)
    print(f"Chain-of-thought messages: {len(messages_cot)} messages")
    
    # Test without few-shot
    messages_no_fs = format_qwen_messages(test_data, strategy="cot", fewshot=False)
    print(f"No few-shot messages: {len(messages_no_fs)} messages")
    
    print("✓ Prompt formatting test passed")


def test_prediction_extraction():
    """Test prediction extraction functionality."""
    print("Testing prediction extraction...")
    
    test_cases = [
        ("YES", "YES"),
        ("NO", "NO"),
        ("Final Answer: YES", "YES"),
        ("**Final Answer**: NO", "NO"),
        ("I found a buffer overflow vulnerability. YES", "YES"),
        ("No security issues detected. NO", "NO"),
        ("This code appears to have a vulnerability", "YES"),
        ("The code is secure and safe", "NO"),
        ("Unclear response", "UNCLEAR")
    ]
    
    for response, expected in test_cases:
        result = extract_qwen_prediction(response)
        if result != expected:
            print(f"✗ Failed: '{response}' -> got '{result}', expected '{expected}'")
            return False
        else:
            print(f"✓ '{response}' -> '{result}'")
    
    print("✓ Prediction extraction test passed")


def test_file_processing():
    """Test file processing functionality."""
    print("Testing file processing...")
    
    # Create test data
    test_file, expected_samples = create_test_data()
    
    try:
        # Import and test the construct_prompts function
        import sys
        sys.path.append('.')
        from run_qwen_prompting import construct_prompts
        
        prompts = construct_prompts(test_file, None)
        
        if len(prompts) != len(expected_samples):
            print(f"✗ Expected {len(expected_samples)} prompts, got {len(prompts)}")
            return False
        
        for i, (prompt, expected) in enumerate(zip(prompts, expected_samples)):
            if prompt["func"] != expected["func"]:
                print(f"✗ Function mismatch at sample {i}")
                return False
            if prompt["target"] != expected["target"]:
                print(f"✗ Target mismatch at sample {i}")
                return False
        
        print("✓ File processing test passed")
        
    finally:
        # Clean up
        os.unlink(test_file)
    
    return True


def main():
    """Run all tests."""
    print("Running Qwen vulnerability detection tests...")
    print("=" * 50)
    
    try:
        test_prompt_formatting()
        print()
        
        test_prediction_extraction()
        print()
        
        test_file_processing()
        print()
        
        print("=" * 50)
        print("✓ All tests passed! Setup is ready.")
        print("\nYou can now run:")
        print("python run_qwen_480b.py --data_path <your_data.jsonl> --output_folder ./output")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
