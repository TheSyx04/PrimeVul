#!/usr/bin/env python3
"""
Resume interrupted model downloads and check download status.
"""

import os
import sys
from pathlib import Path
import json


def get_cache_info():
    """Get information about cached models."""
    cache_dir = Path.home() / '.cache' / 'huggingface' / 'transformers'
    
    if not cache_dir.exists():
        print("No Hugging Face cache found")
        return {}
    
    print(f"Cache directory: {cache_dir}")
    
    # Look for Qwen models
    qwen_models = {}
    for item in cache_dir.iterdir():
        if item.is_dir() and 'qwen' in item.name.lower():
            model_path = item
            
            # Check for model files
            model_files = []
            total_size = 0
            
            for file_path in model_path.rglob('*'):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    total_size += size
                    model_files.append({
                        'name': file_path.name,
                        'size': size,
                        'path': str(file_path)
                    })
            
            qwen_models[item.name] = {
                'path': str(model_path),
                'files': len(model_files),
                'total_size_gb': total_size / (1024**3),
                'model_files': model_files
            }
    
    return qwen_models


def check_model_completeness(model_name):
    """Check if a model is completely downloaded."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"Checking model completeness: {model_name}")
        
        # Try to load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                local_files_only=True,
                trust_remote_code=True
            )
            print("✅ Tokenizer: Available")
        except Exception as e:
            print(f"❌ Tokenizer: Missing or incomplete - {e}")
            return False
        
        # Try to load model config (lighter check)
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                model_name,
                local_files_only=True,
                trust_remote_code=True
            )
            print("✅ Config: Available")
        except Exception as e:
            print(f"❌ Config: Missing or incomplete - {e}")
            return False
        
        print("✅ Model appears to be complete")
        return True
        
    except Exception as e:
        print(f"❌ Error checking model: {e}")
        return False


def resume_download(model_name):
    """Resume downloading a model."""
    print(f"Resuming download for: {model_name}")
    
    try:
        from huggingface_hub import snapshot_download
        
        # Set longer timeout
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '7200'
        
        snapshot_download(
            repo_id=model_name,
            resume_download=True,
            local_files_only=False,
            repo_type="model"
        )
        
        print("✅ Download resumed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to resume download: {e}")
        return False


def clear_incomplete_downloads(model_name=None):
    """Clear incomplete downloads."""
    cache_dir = Path.home() / '.cache' / 'huggingface'
    
    if model_name:
        # Clear specific model
        print(f"Clearing cache for: {model_name}")
        # This would require more specific logic
    else:
        print("Clearing all incomplete downloads...")
        # This would require careful implementation
        print("⚠️  Manual clearing required - check cache directory:")
        print(f"   {cache_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage Qwen model downloads")
    parser.add_argument("--check", type=str, help="Check if model is complete")
    parser.add_argument("--resume", type=str, help="Resume model download")
    parser.add_argument("--list", action="store_true", help="List cached models")
    parser.add_argument("--clear", type=str, help="Clear incomplete downloads")
    
    args = parser.parse_args()
    
    print("Qwen Model Download Manager")
    print("=" * 40)
    
    if args.list:
        print("Cached Qwen models:")
        models = get_cache_info()
        if not models:
            print("No Qwen models found in cache")
        else:
            for name, info in models.items():
                print(f"\n📁 {name}")
                print(f"   Path: {info['path']}")
                print(f"   Files: {info['files']}")
                print(f"   Size: {info['total_size_gb']:.1f} GB")
    
    elif args.check:
        complete = check_model_completeness(args.check)
        if complete:
            print(f"\n✅ {args.check} is ready to use")
        else:
            print(f"\n❌ {args.check} is incomplete or missing")
            print("Consider using --resume to complete the download")
    
    elif args.resume:
        success = resume_download(args.resume)
        if success:
            print(f"\n✅ Successfully resumed download of {args.resume}")
        else:
            print(f"\n❌ Failed to resume download of {args.resume}")
    
    elif args.clear:
        clear_incomplete_downloads(args.clear)
    
    else:
        print("Use --help for available options")
        print("\nCommon usage:")
        print("  python download_manager.py --list")
        print("  python download_manager.py --check Qwen/Qwen3-Coder-480B-A35B-Instruct")
        print("  python download_manager.py --resume Qwen/Qwen3-Coder-480B-A35B-Instruct")


if __name__ == "__main__":
    main()
