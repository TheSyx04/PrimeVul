#!/usr/bin/env python3
"""
Test script to verify Qwen prompts and imports are working correctly.
"""

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    try:
        from qwen_utils import get_qwen_prompts, format_qwen_messages, extract_qwen_prediction
        print("✅ qwen_utils imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_prompts():
    """Test that prompts are correctly formatted."""
    print("\nTesting prompts...")
    try:
        from qwen_utils import get_qwen_prompts
        
        prompts = get_qwen_prompts()
        required_keys = ['system', 'std_cls', 'cot', 'oneshot_user', 'oneshot_assistant']
        
        for key in required_keys:
            if key not in prompts:
                print(f"❌ Missing prompt key: {key}")
                return False
            print(f"✅ Found prompt: {key}")
        
        # Test prompt formatting
        test_code = "void test() { char buf[10]; gets(buf); }"
        cot_prompt = prompts['cot'].format(func=test_code)
        std_prompt = prompts['std_cls'].format(func=test_code)
        
        print(f"✅ CoT prompt length: {len(cot_prompt)} chars")
        print(f"✅ Standard prompt length: {len(std_prompt)} chars")
        return True
        
    except Exception as e:
        print(f"❌ Prompt test failed: {e}")
        return False

def test_prediction_extraction():
    """Test prediction extraction."""
    print("\nTesting prediction extraction...")
    try:
        from qwen_utils import extract_qwen_prediction
        
        test_cases = [
            ("YES", "YES"),
            ("NO", "NO"),
            ("Final Answer: YES", "YES"),
            ("**Final Answer**: NO", "NO"),
            ("Analysis shows vulnerability. YES", "YES"),
            ("No issues found. NO", "NO"),
            ("Unclear response", "UNCLEAR")
        ]
        
        for input_text, expected in test_cases:
            result = extract_qwen_prediction(input_text)
            if result == expected:
                print(f"✅ '{input_text}' → '{result}'")
            else:
                print(f"❌ '{input_text}' → '{result}' (expected '{expected}')")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction extraction test failed: {e}")
        return False

def test_simple_runner_imports():
    """Test that simple runner imports work."""
    print("\nTesting simple runner imports...")
    try:
        import sys
        import os
        
        # Add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Test the imports used in run_simple_qwen.py
        from qwen_utils import get_qwen_prompts, extract_qwen_prediction
        print("✅ Simple runner imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Simple runner import test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Qwen Setup")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_prompts,
        test_prediction_extraction,
        test_simple_runner_imports
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Simple runner should work.")
        print("\n💡 Try running:")
        print("python run_simple_qwen.py --model_name Qwen/Qwen3-Coder-480B-A35B-Instruct --data_path ../data/FFmpeg/Realistic/SETUP2-FFmpeg-deepjit-test.jsonl --output_folder ./output --strategy cot --fewshot")
    else:
        print("❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
