#!/usr/bin/env python3
"""
Network-robust script for downloading and running large Qwen models.
Handles timeouts, partial downloads, and provides better error recovery.
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path


def setup_environment():
    """Setup environment variables for better downloads."""
    # Set longer timeouts
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '7200'  # 2 hours
    os.environ['TRANSFORMERS_CACHE'] = str(Path.home() / '.cache' / 'huggingface' / 'transformers')
    os.environ['HF_HOME'] = str(Path.home() / '.cache' / 'huggingface')
    
    # Enable resume downloads
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    
    print("Environment configured for large model downloads:")
    print(f"  Download timeout: {os.environ['HF_HUB_DOWNLOAD_TIMEOUT']} seconds")
    print(f"  Cache directory: {os.environ['TRANSFORMERS_CACHE']}")


def check_disk_space(cache_dir=None):
    """Check available disk space."""
    if cache_dir is None:
        cache_dir = Path.home() / '.cache' / 'huggingface'
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Get available space
    statvfs = os.statvfs(cache_dir)
    available_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
    
    print(f"Available disk space: {available_gb:.1f} GB")
    
    # 480B model needs ~800GB+
    if available_gb < 900:
        print("⚠️  WARNING: You may not have enough disk space for the 480B model")
        print("   480B model requires ~800-900GB of space")
        print(f"   Available: {available_gb:.1f} GB")
        
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return False
    
    return True


def pre_download_model(model_name, cache_dir=None):
    """Pre-download model files separately to handle timeouts better."""
    print(f"Pre-downloading model: {model_name}")
    
    try:
        from huggingface_hub import snapshot_download
        
        # Download with longer timeout and better error handling
        snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False,
            repo_type="model"
        )
        print("✅ Model pre-download completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Pre-download failed: {e}")
        print("Will attempt to download during model loading...")
        return False


def run_with_timeout_handling(args):
    """Run the main script with better timeout handling."""
    
    cmd = [
        "python", "run_qwen_prompting.py",
        "--model_name", args.model_name,
        "--prompt_strategy", args.strategy,
        "--data_path", args.data_path,
        "--output_folder", args.output_folder,
        "--temperature", str(args.temperature),
        "--max_gen_length", str(args.max_gen_length),
        "--max_retries", str(args.max_retries)
    ]
    
    if args.fewshot:
        cmd.append("--fewshot_eg")
    
    if args.cache_dir:
        cmd.extend(["--cache_dir", args.cache_dir])
    
    if args.offline:
        cmd.append("--offline")
    
    if args.force_download:
        cmd.append("--force_download")
    
    print("Running command:")
    print(" ".join(cmd))
    print("-" * 60)
    
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            print(f"Attempt {attempt + 1}/{max_attempts}")
            result = subprocess.run(cmd, check=True, timeout=None)
            print("✅ Execution completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Execution failed with exit code {e.returncode}")
            if attempt < max_attempts - 1:
                wait_time = 30 * (attempt + 1)
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print("All attempts failed")
                return False
                
        except KeyboardInterrupt:
            print("❌ Execution interrupted by user")
            return False
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Network-robust runner for large Qwen models"
    )
    
    parser.add_argument("--model_name", type=str, 
                       default="Qwen/Qwen3-Coder-480B-A35B-Instruct",
                       help="Model name to download and run")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to test data")
    parser.add_argument("--output_folder", type=str, required=True,
                       help="Output folder")
    parser.add_argument("--strategy", type=str, choices=["std_cls", "cot"], 
                       default="cot", help="Prompting strategy")
    parser.add_argument("--fewshot", action="store_true",
                       help="Use few-shot examples")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Generation temperature")
    parser.add_argument("--max_gen_length", type=int, default=2048,
                       help="Maximum generation length")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Custom cache directory")
    parser.add_argument("--offline", action="store_true",
                       help="Use offline mode")
    parser.add_argument("--force_download", action="store_true",
                       help="Force re-download")
    parser.add_argument("--max_retries", type=int, default=5,
                       help="Maximum retries")
    parser.add_argument("--pre_download", action="store_true",
                       help="Pre-download model before running")
    parser.add_argument("--skip_space_check", action="store_true",
                       help="Skip disk space check")
    
    args = parser.parse_args()
    
    print("Qwen Large Model Runner with Network Robustness")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Check disk space
    if not args.skip_space_check and not check_disk_space(args.cache_dir):
        sys.exit(1)
    
    # Pre-download if requested
    if args.pre_download and not args.offline:
        print("Pre-downloading model...")
        pre_download_model(args.model_name, args.cache_dir)
        print()
    
    # Run with timeout handling
    success = run_with_timeout_handling(args)
    
    if success:
        print("\n🎉 Execution completed successfully!")
        print(f"Results should be in: {args.output_folder}")
    else:
        print("\n❌ Execution failed after all attempts")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Try using --pre_download to download the model first")
        print("3. Use a smaller model for testing:")
        print("   --model_name Qwen/Qwen2.5-Coder-32B-Instruct")
        print("4. Use --offline if model is already cached")
        sys.exit(1)


if __name__ == "__main__":
    main()
