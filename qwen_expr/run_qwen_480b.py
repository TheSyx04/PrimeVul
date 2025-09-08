#!/usr/bin/env python3
"""
Runner script specifically for Qwen3-Coder-480B-A35B-Instruct model
with optimized settings for vulnerability detection.
"""

import argparse
import os
import subprocess
import sys


def check_requirements():
    """Check if required packages are installed."""
    required_packages = ['torch', 'transformers', 'accelerate', 'bitsandbytes']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:", missing_packages)
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def run_qwen_480b(data_path, output_folder, strategy="cot", fewshot=True, temperature=0.0):
    """
    Run Qwen3-Coder-480B-A35B-Instruct for vulnerability detection.
    
    Args:
        data_path: Path to test data file
        output_folder: Output directory for results
        strategy: Prompt strategy ("std_cls" or "cot")
        fewshot: Whether to use few-shot examples
        temperature: Generation temperature
    """
    
    # The actual 480B model path (correct model name)
    model_name = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
    
    # Prepare command
    cmd = [
        "python", "run_qwen_prompting.py",
        "--model_name", model_name,
        "--prompt_strategy", strategy,
        "--data_path", data_path,
        "--output_folder", output_folder,
        "--temperature", str(temperature),
        "--max_gen_length", "2048" if strategy == "cot" else "512",
        "--max_input_length", "8192",  # Larger context for 480B model
        "--device", "auto",
        "--seed", "42"
    ]
    
    if fewshot:
        cmd.append("--fewshot_eg")
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Model: {model_name}")
    print(f"Strategy: {strategy}")
    print(f"Few-shot: {fewshot}")
    print(f"Temperature: {temperature}")
    print("-" * 50)
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("Execution completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
        return False
    except KeyboardInterrupt:
        print("Execution interrupted by user")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen3-Coder-480B-A35B-Instruct for vulnerability detection"
    )
    
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to test data file (JSONL format)")
    parser.add_argument("--output_folder", type=str, required=True,
                       help="Output folder for results")
    parser.add_argument("--strategy", type=str, choices=["std_cls", "cot"], default="cot",
                       help="Prompt strategy: std_cls or cot (default: cot)")
    parser.add_argument("--no_fewshot", action="store_true",
                       help="Disable few-shot examples")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Generation temperature (default: 0.0 for deterministic)")
    parser.add_argument("--check_only", action="store_true",
                       help="Only check requirements, don't run")
    
    args = parser.parse_args()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    if args.check_only:
        print("Requirements check passed!")
        return
    
    # Validate inputs
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Run the model
    success = run_qwen_480b(
        data_path=args.data_path,
        output_folder=args.output_folder,
        strategy=args.strategy,
        fewshot=not args.no_fewshot,
        temperature=args.temperature
    )
    
    if success:
        print(f"\nResults saved to: {args.output_folder}")
        print("You can calculate VD-Score using:")
        print(f"python ../calc_vd_score.py --pred_file {args.output_folder}/predictions.txt --test_file {args.data_path}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
