#!/usr/bin/env python3
"""
Quick fix for the timeout issue when loading Qwen3-Coder-480B-A35B-Instruct.
This script provides several solutions to handle the network timeout.
"""

def show_timeout_solutions():
    print("🔧 Qwen 480B Model Timeout Issue - Solutions")
    print("=" * 50)
    print()
    
    print("❌ PROBLEM:")
    print("The 480B model download was interrupted due to network timeout.")
    print("Files were partially downloaded but the process failed.")
    print()
    
    print("✅ SOLUTION 1: Use the Simple Runner (No Network Config)")
    print("-" * 50)
    print("python run_simple_qwen.py \\")
    print("    --model_name Qwen/Qwen3-Coder-480B-A35B-Instruct \\")
    print("    --data_path ../data/FFmpeg/Realistic/SETUP2-FFmpeg-deepjit-test.jsonl \\")
    print("    --output_folder ./output \\")
    print("    --strategy cot \\")
    print("    --fewshot")
    print()
    
    print("✅ SOLUTION 2: Use the Network-Robust Runner")
    print("-" * 45)
    print("python run_robust.py \\")
    print("    --model_name Qwen/Qwen3-Coder-480B-A35B-Instruct \\")
    print("    --data_path ../data/FFmpeg/Realistic/SETUP2-FFmpeg-deepjit-test.jsonl \\")
    print("    --output_folder ./output \\")
    print("    --strategy cot \\")
    print("    --fewshot \\")
    print("    --pre_download \\")
    print("    --max_retries 5")
    print()
    
    print("✅ SOLUTION 3: Resume Interrupted Download")
    print("-" * 40)
    print("# First, resume the download")
    print("python download_manager.py --resume Qwen/Qwen3-Coder-480B-A35B-Instruct")
    print()
    print("# Then run normally")
    print("python run_qwen_prompting.py \\")
    print("    --model_name Qwen/Qwen3-Coder-480B-A35B-Instruct \\")
    print("    --data_path ../data/FFmpeg/Realistic/SETUP2-FFmpeg-deepjit-test.jsonl \\")
    print("    --output_folder ./output \\")
    print("    --prompt_strategy cot \\")
    print("    --fewshot_eg \\")
    print("    --max_retries 5")
    print()
    
    print("✅ SOLUTION 4: Use Offline Mode (if model is cached)")
    print("-" * 40)
    print("# Check if model is available")
    print("python download_manager.py --check Qwen/Qwen3-Coder-480B-A35B-Instruct")
    print()
    print("# If available, run in offline mode")
    print("python run_qwen_prompting.py \\")
    print("    --model_name Qwen/Qwen3-Coder-480B-A35B-Instruct \\")
    print("    --data_path ../data/FFmpeg/Realistic/SETUP2-FFmpeg-deepjit-test.jsonl \\")
    print("    --output_folder ./output \\")
    print("    --prompt_strategy cot \\")
    print("    --fewshot_eg \\")
    print("    --offline")
    print()
    
    print("✅ SOLUTION 5: Use Alternative Model (for testing)")
    print("-" * 40)
    print("# Use smaller but still powerful model")
    print("python run_qwen_prompting.py \\")
    print("    --model_name Qwen/Qwen3-Coder-30B-A3B-Instruct \\")
    print("    --data_path ../data/FFmpeg/Realistic/SETUP2-FFmpeg-deepjit-test.jsonl \\")
    print("    --output_folder ./output \\")
    print("    --prompt_strategy cot \\")
    print("    --fewshot_eg")
    print()
    
    print("🔧 ENVIRONMENT SETUP (to prevent future timeouts):")
    print("-" * 50)
    print("export HF_HUB_DOWNLOAD_TIMEOUT=7200  # 2 hours")
    print("export HF_HUB_ENABLE_HF_TRANSFER=1   # Enable faster downloads")
    print()
    
    print("📊 CHECK DISK SPACE:")
    print("-" * 20)
    print("The 480B model requires ~800-900GB of disk space.")
    print("Check available space: df -h ~/.cache/huggingface/")
    print()
    
    print("🚀 RECOMMENDED WORKFLOW:")
    print("-" * 25)
    print("1. Check disk space and network connection")
    print("2. Use run_robust.py with --pre_download")
    print("3. If download fails, use download_manager.py to resume")
    print("4. Once downloaded, run with --offline flag")
    print("5. For testing, start with smaller models first")


def check_current_status():
    """Check the current status of the download."""
    print("\n🔍 CHECKING CURRENT STATUS:")
    print("-" * 30)
    
    try:
        import subprocess
        result = subprocess.run([
            "python", "download_manager.py", "--list"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("Could not check download status")
    except:
        print("Download manager not available")
    
    print("\nNext steps:")
    print("1. python download_manager.py --check Qwen/Qwen3-Coder-480B-A35B-Instruct")
    print("2. python download_manager.py --resume Qwen/Qwen3-Coder-480B-A35B-Instruct")


if __name__ == "__main__":
    show_timeout_solutions()
    check_current_status()
