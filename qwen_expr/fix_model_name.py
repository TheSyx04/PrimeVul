#!/usr/bin/env python3
"""
Test script to validate the corrected Qwen model command.
Creates a simple test case and shows the corrected usage.
"""

def show_corrected_usage():
    print("Qwen Model Name Correction")
    print("=" * 30)
    print()
    
    print("❌ INCORRECT (will fail):")
    print("python run_qwen_prompting.py \\")
    print("    --model_name Qwen/QwenCoder-480B-A35B-Instruct \\")
    print("    --prompt_strategy cot \\")
    print("    --data_path ../data/FFmpeg/Realistic/SETUP2-FFmpeg-deepjit-test.jsonl \\")
    print("    --output_folder ./output \\")
    print("    --fewshot_eg \\")
    print("    --temperature 0.0 \\")
    print("    --max_gen_length 2048")
    print()
    
    print("✅ CORRECT (will work):")
    print("python run_qwen_prompting.py \\")
    print("    --model_name Qwen/Qwen3-Coder-480B-A35B-Instruct \\")
    print("    --prompt_strategy cot \\")
    print("    --data_path ../data/FFmpeg/Realistic/SETUP2-FFmpeg-deepjit-test.jsonl \\")
    print("    --output_folder ./output \\")
    print("    --fewshot_eg \\")
    print("    --temperature 0.0 \\")
    print("    --max_gen_length 2048")
    print()
    
    print("🔧 Key Differences:")
    print("- OLD: Qwen/QwenCoder-480B-A35B-Instruct")
    print("- NEW: Qwen/Qwen3-Coder-480B-A35B-Instruct")
    print()
    
    print("📝 Alternative Models (if 480B is not available):")
    print("1. Qwen/Qwen3-Coder-30B-A3B-Instruct  (smaller but still powerful)")
    print("2. Qwen/Qwen2.5-Coder-32B-Instruct    (fallback option)")
    print("3. Qwen/Qwen2.5-Coder-14B-Instruct    (for testing)")
    print()
    
    print("🚀 Quick Test Command:")
    print("python check_models.py  # Check available models")
    print()
    print("python run_qwen_480b.py \\")
    print("    --data_path <your_test_data.jsonl> \\")
    print("    --output_folder ./output")
    print()
    
    print("💡 Tips:")
    print("- The script now has automatic model name correction")
    print("- It will try alternative models if the first one fails")
    print("- Use --check_only flag to verify setup without running")


def create_sample_test_data():
    """Create a small sample test data file for quick testing."""
    import json
    
    sample_data = [
        {
            "project": "test_ffmpeg",
            "commit_id": "sample123", 
            "func": """int parse_packet(uint8_t *data, int size) {
    uint8_t buffer[256];
    if (size > 0) {
        memcpy(buffer, data, size);  // Potential buffer overflow
        return process_buffer(buffer);
    }
    return -1;
}""",
            "target": "YES"
        }
    ]
    
    with open("sample_test.jsonl", "w") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    
    print("📄 Created sample_test.jsonl for quick testing")
    print()
    print("Test with corrected command:")
    print("python run_qwen_prompting.py \\")
    print("    --model_name Qwen/Qwen3-Coder-480B-A35B-Instruct \\")
    print("    --prompt_strategy cot \\")
    print("    --data_path sample_test.jsonl \\")
    print("    --output_folder ./test_output \\")
    print("    --fewshot_eg \\")
    print("    --temperature 0.0")


if __name__ == "__main__":
    show_corrected_usage()
    print("\n" + "=" * 50 + "\n")
    create_sample_test_data()
