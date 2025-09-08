# coding=utf-8

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os

# Force single GPU mode by clearing distributed environment variables at startup
print("Forcing single GPU mode - clearing distributed environment variables...")
for key in ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
    if key in os.environ:
        print(f"Removing {key}={os.environ[key]}")
        del os.environ[key]

os.environ["HF_ENDPOINT"] = "https://huggingface.co"
# Set environment variable to help with flash attention compatibility
os.environ["FLASH_ATTENTION_SKIP_CUDA_CHECK"] = "1"
# Additional NCCL debugging and configuration
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_TIMEOUT"] = "1800"
# Reduce distributed training timeout issues
os.environ["TORCH_DISTRIBUTED_DETAIL"] = "DEBUG"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
# Set shorter timeout for quicker failure detection
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

# Set wandb token
os.environ["WANDB_API_KEY"] = "fb5a5b79b5aafdb17cb882dd76ac2e0cde9adf8d"

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import json
import torch
import time
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, matthews_corrcoef, auc
from tqdm import tqdm
import multiprocessing
from model import DecoderClassifier

cpu_cont = multiprocessing.cpu_count()
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    get_linear_schedule_with_warmup,
    Qwen2Config, Qwen2Model, Qwen2ForCausalLM
)

# Import AdamW from torch.optim (newer versions) or transformers (older versions)
try:
    from torch.optim import AdamW
except ImportError:
    from transformers import AdamW

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import wandb

# LoRA imports
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
    PeftConfig
)

logger = get_logger(__name__)

def format_time_duration(seconds):
    """Format time duration in a human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m ({seconds:.0f}s)"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{hours:.1f}h ({minutes:.0f}m)"

# Chain of Thought Prompts for Few-Shot Learning
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

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, input_tokens, input_ids, idx, label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label

def convert_examples_to_features_cot(js, tokenizer, args):
    """Convert examples to features with Chain of Thought prompting"""
    code = js['code_change']
    
    # Handle None or non-string code
    if code is None:
        code = ""
    elif not isinstance(code, str):
        code = str(code)
    
    # Ensure code is not empty - if it is, use a placeholder
    if not code.strip():
        code = "// empty code"
    
    # Additional safety check - ensure code is a valid string
    try:
        code = str(code).strip()
        if not code:
            code = "// empty code"
        
        # Clean up problematic Unicode escape sequences
        import re
        code = re.sub(r'\\ud[c-f][0-9a-f]{2}', '?', code, flags=re.IGNORECASE)
        code = code.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        
    except Exception as e:
        print(f"Error converting code to string: {e}, using placeholder")
        code = "// empty code"
    
    # Create Chain of Thought prompt
    if args.use_few_shot:
        prompt = create_cot_prompt(code, include_examples=True)
    else:
        prompt = create_cot_prompt(code, include_examples=False)
    
    # Tokenize the prompt
    try:
        # Use chat template if available (for instruction-tuned models like Qwen)
        if hasattr(tokenizer, 'apply_chat_template') and args.use_chat_template:
            messages = [
                {"role": "system", "content": COT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt.replace(COT_SYSTEM_PROMPT + "\n\n", "")}
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            encoded = tokenizer.encode(formatted_prompt, max_length=args.block_size, truncation=True)
        else:
            # Fallback to regular encoding
            encoded = tokenizer.encode(prompt, max_length=args.block_size, truncation=True)
        
        source_tokens = tokenizer.convert_ids_to_tokens(encoded)
        source_ids = encoded
        
        # Pad to block_size
        padding_length = args.block_size - len(source_ids)
        if padding_length > 0:
            source_ids = source_ids + [tokenizer.pad_token_id] * padding_length
        else:
            source_ids = source_ids[:args.block_size]
            
    except Exception as e:
        print(f"Error tokenizing prompt: {e}")
        # Fallback to simple approach
        source_tokens = [tokenizer.unk_token] * min(10, args.block_size)
        source_ids = [tokenizer.unk_token_id] * args.block_size
    
    return InputFeatures(source_tokens, source_ids, js['commit_id'], js['label'])

class TextDatasetCoT(Dataset):
    def __init__(self, tokenizer, args, file_path=None, verbose=True):
        self.examples = []
        with open(file_path) as f:
            for line_num, line in enumerate(f):
                try:
                    js = json.loads(line.strip())
                    if line_num < 3 and verbose:
                        print(f"Line {line_num}: Processing entry with commit_id={js.get('commit_id', 'N/A')}")
                    self.examples.append(convert_examples_to_features_cot(js, tokenizer, args))
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    print(f"Line content: {line.strip()[:200]}...")
                    raise

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)

def create_qwen_lora_config(args):
    """Create LoRA configuration specifically for Qwen models"""
    # Qwen models typically use these layer names
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj"       # MLP layers
    ]
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # For generative models
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=["lm_head", "embed_tokens"] if hasattr(args, 'modules_to_save') else None,
    )
    
    return lora_config

def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    return trainable_params, all_param

def calculate_metrics(labels, preds, probs=None):
    """Calculate comprehensive metrics"""
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    TN, FP, FN, TP = confusion_matrix(labels, preds).ravel()
    tnr = TN/(TN+FP) if (TN+FP) > 0 else 0
    fpr = FP/(FP+TN) if (FP+TN) > 0 else 0
    fnr = FN/(TP+FN) if (TP+FN) > 0 else 0
    mcc = matthews_corrcoef(labels, preds)
    
    # Calculate AUC metrics if probabilities are provided
    roc_auc = None
    pr_auc = None
    if probs is not None:
        try:
            roc_auc = roc_auc_score(labels, probs)
            precisions, recalls, _ = precision_recall_curve(labels, probs)
            pr_auc = auc(recalls, precisions)
        except ValueError:
            # Handle cases where only one class is present
            roc_auc = 0.0
            pr_auc = 0.0
    
    if roc_auc is not None and pr_auc is not None:
        return round(acc,4)*100, round(prec,4)*100, \
            round(recall,4)*100, round(f1,4)*100, round(tnr,4)*100, \
                round(fpr,4)*100, round(fnr,4)*100, round(mcc,4), \
                    round(roc_auc,4), round(pr_auc,4)
    else:
        return round(acc,4)*100, round(prec,4)*100, \
            round(recall,4)*100, round(f1,4)*100, round(tnr,4)*100, \
                round(fpr,4)*100, round(fnr,4)*100, round(mcc,4)

def train(args, accelerator, train_dataset, eval_dataset, model, tokenizer):
    """Train the model with CoT and few-shot learning"""
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, 
                                num_workers=args.dataloader_num_workers, pin_memory=True)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.eval_batch_size, 
                               num_workers=args.dataloader_num_workers, pin_memory=True)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_steps = args.epoch * num_update_steps_per_epoch
    args.save_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    if args.warmup_steps == 0:
        num_warmup = args.max_steps * args.warmup_ratio
    else:
        num_warmup = args.warmup_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup,
                                                num_training_steps=args.max_steps)
   
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    logger.info("***** Running Chain of Thought + Few-Shot Training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Few-shot examples = %s", args.use_few_shot)
    logger.info("  Chat template = %s", args.use_chat_template)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    # Start timing the entire training
    training_start_time = time.time()
    
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0.0
    best_acc = 0.0
    patience = 0
    epoch_durations = []

    train_dataloader, eval_dataloader, model, optimizer, scheduler = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer, scheduler
    )
    model.zero_grad()

    # Modified checkpoint saving for LoRA
    def save_model_checkpoint(output_dir, epoch, step, is_best=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        if args.use_lora:
            # Save LoRA weights to lora subdirectory
            lora_output_dir = os.path.join(output_dir, "lora")
            if not os.path.exists(lora_output_dir):
                os.makedirs(lora_output_dir, exist_ok=True)
            
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(lora_output_dir)
            
            # Optionally merge and save full model
            if args.merge_lora and is_best:
                logger.info("Merging LoRA weights with base model...")
                merged_output_dir = os.path.join(output_dir, "merged")
                os.makedirs(merged_output_dir, exist_ok=True)
                
                # Merge LoRA weights
                merged_model = unwrapped_model.merge_and_unload()
                merged_model.save_pretrained(merged_output_dir)
                tokenizer.save_pretrained(merged_output_dir)
            
            logger.info(f"Model checkpoint saved to {lora_output_dir}")
        else:
            # Save full model using accelerator
            accelerator.save_state(output_dir)
            logger.info(f"Model checkpoint saved to {output_dir}")
 
    step = 0
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        # Start timing the epoch
        epoch_start_time = time.time()
        
        bar = tqdm(train_dataloader, total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        tr_num = 0
        train_loss = 0
        logits_lst = []
        labels_lst = []
        
        for local_step, batch in enumerate(bar):
            model.train()
            inputs, labels = batch
            
            with accelerator.accumulate(model):
                # Handle both cases: when model returns (loss, logits) or just logits
                model_output = model(inputs, labels)
                if isinstance(model_output, tuple) and len(model_output) == 2:
                    loss, logits = model_output
                else:
                    # If only logits are returned, compute loss manually
                    logits = model_output
                    # Compute loss manually using the same approach as in the model
                    loss_fct = torch.nn.CrossEntropyLoss()
                    if len(logits.shape) == 3:  # if logits are [batch, seq_len, num_classes]
                        # Pool the logits similar to DecoderClassifier
                        batch_size = inputs.size(0)
                        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                            sequence_lengths = torch.eq(inputs, tokenizer.pad_token_id).int().argmax(-1) - 1
                            sequence_lengths = sequence_lengths % inputs.shape[-1]
                            sequence_lengths = sequence_lengths.to(logits.device)
                            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
                        else:
                            pooled_logits = logits[:, -1, :]  # Use last token
                        loss = loss_fct(pooled_logits.view(-1, 2), labels.view(-1))
                        logits = torch.nn.functional.softmax(pooled_logits, dim=-1)
                    else:
                        loss = loss_fct(logits.view(-1, 2), labels.view(-1))

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            logits, labels = accelerator.gather_for_metrics((logits, labels))
            
            # cast to torch.float16
            logits_lst.append(logits.detach().cpu().float().numpy())
            labels_lst.append(labels.detach().cpu().float().numpy())

            step_loss = accelerator.reduce(loss.detach().clone()).item()
            tr_loss += step_loss    
            tr_num += 1
            train_loss += step_loss
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))
                
            # Log after every logging_steps
            if (step + 1) % args.logging_steps == 0:
                avg_loss = round(train_loss/tr_num, 5)
                if args.evaluate_during_training:
                    results = evaluate(args, accelerator, eval_dataloader, eval_dataset, model, tokenizer, eval_when_training=True)
                    for key, value in results.items():
                        logger.info("  %s = %s", key, round(value, 4))
                    
                    # Log to wandb at logging steps
                    if args.use_wandb and accelerator.is_main_process:
                        step_wandb_log = {"step": step, "train/loss": avg_loss}
                        for key, value in results.items():
                            wandb_key = key.replace("eval_", "eval/")
                            step_wandb_log[wandb_key] = value
                        wandb.log(step_wandb_log, step=step)                    
                
                # Save model checkpoint    
                if results['eval_f1'] > best_f1:
                    best_f1 = results['eval_f1']
                    logger.info("  " + "*" * 20)  
                    logger.info("  Best f1:%s", round(best_f1, 4))
                    logger.info("  " + "*" * 20)                          
                    
                    checkpoint_prefix = f'checkpoint-best-f1/{args.project}/{args.model_dir}'
                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                    save_model_checkpoint(output_dir, idx, step, is_best=True)
            
            # increment step within the same epoch
            step += 1
            
            # Memory cleanup for HPC environments
            if args.cleanup_cache and step % 10 == 0:
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        
        # Log after every epoch
        avg_loss = round(train_loss/tr_num, 5)
        logits_lst = np.concatenate(logits_lst, 0)
        labels_lst = np.concatenate(labels_lst, 0)
        
        # Calculate train metrics
        train_preds = logits_lst[:, 1] > 0.5
        train_probs = logits_lst[:, 1]
        
        # Calculate enhanced train metrics
        train_metrics_result = calculate_metrics(labels_lst, train_preds, train_probs)
        if len(train_metrics_result) == 10:  # Enhanced metrics with AUC
            train_acc, train_prec, train_recall, train_f1, train_tnr, train_fpr, train_fnr, train_mcc, train_roc_auc, train_pr_auc = train_metrics_result
        else:  # Fallback to basic metrics
            train_acc, train_prec, train_recall, train_f1, train_tnr, train_fpr, train_fnr, train_mcc = train_metrics_result
            train_roc_auc = 0.0
            train_pr_auc = 0.0
        
        # Calculate epoch duration
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_durations.append(epoch_duration)
        
        # Calculate estimated time remaining
        if len(epoch_durations) > 0:
            avg_epoch_duration = sum(epoch_durations) / len(epoch_durations)
            remaining_epochs = int(args.num_train_epochs) - (idx + 1)
            eta_seconds = avg_epoch_duration * remaining_epochs
        else:
            eta_seconds = 0
        
        if args.local_rank in [-1, 0]:
            # Log epoch timing with ETA
            epoch_time_str = format_time_duration(epoch_duration)
            if remaining_epochs > 0:
                eta_str = format_time_duration(eta_seconds)
                logger.info(f"Epoch {idx} completed in {epoch_time_str}. ETA: {eta_str}")
            else:
                logger.info(f"Epoch {idx} completed in {epoch_time_str}. Final epoch!")
            
            if args.evaluate_during_training:
                results = evaluate(args, accelerator, eval_dataloader, eval_dataset, model, tokenizer, eval_when_training=True)
                for key, value in results.items():
                    logger.info("  %s = %s", key, round(value, 4))
            
            # Save model checkpoint    
            if results['eval_f1'] > best_f1:
                best_f1 = results['eval_f1']
                logger.info("  " + "*" * 20)  
                logger.info("  Best f1:%s", round(best_f1, 4))
                logger.info("  " + "*" * 20)                          
                
                checkpoint_prefix = f'checkpoint-best-f1/{args.project}/{args.model_dir}'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                save_model_checkpoint(output_dir, idx, step, is_best=True)
                patience = 0
            else:
                patience += 1

            if results['eval_acc'] > best_acc:
                best_acc = results['eval_acc']
                logger.info("  " + "*" * 20)
                logger.info("  Best acc:%s", round(best_acc, 4))
                logger.info("  " + "*" * 20)                          
                
                checkpoint_prefix = f'checkpoint-best-acc/{args.project}/{args.model_dir}'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                save_model_checkpoint(output_dir, idx, step, is_best=True)
        
        if patience == args.max_patience:
            logger.info(f"Reached max patience {args.max_patience}. End training now.")
            break
    
    # Calculate total training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    # Log total training time
    if accelerator.is_main_process:
        total_time_str = format_time_duration(total_training_time)
        logger.info(f"***** Training completed in {total_time_str} *****")

def evaluate(args, accelerator, eval_dataloader, eval_dataset, model, tokenizer, eval_when_training=False):
    """Evaluate the model"""
    model.eval()
    losses = []
    logits = [] 
    labels = []
    
    for batch in eval_dataloader:
        inputs, label = batch
        with torch.no_grad():
            # Handle both cases: when model returns (loss, logits) or just logits
            model_output = model(inputs, label)
            if isinstance(model_output, tuple) and len(model_output) == 2:
                lm_loss, logit = model_output
            else:
                # If only logits are returned, compute loss manually
                logit = model_output
                # Compute loss manually using the same approach as in the model
                loss_fct = torch.nn.CrossEntropyLoss()
                if len(logit.shape) == 3:  # if logits are [batch, seq_len, num_classes]
                    # Pool the logits similar to DecoderClassifier
                    batch_size = inputs.size(0)
                    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                        sequence_lengths = torch.eq(inputs, tokenizer.pad_token_id).int().argmax(-1) - 1
                        sequence_lengths = sequence_lengths % inputs.shape[-1]
                        sequence_lengths = sequence_lengths.to(logit.device)
                        pooled_logits = logit[torch.arange(batch_size, device=logit.device), sequence_lengths]
                    else:
                        pooled_logits = logit[:, -1, :]  # Use last token
                    lm_loss = loss_fct(pooled_logits.view(-1, 2), label.view(-1))
                    logit = torch.nn.functional.softmax(pooled_logits, dim=-1)
                else:
                    lm_loss = loss_fct(logit.view(-1, 2), label.view(-1))
        
        losses.append(accelerator.gather_for_metrics(lm_loss.repeat(args.eval_batch_size)))
        logit, label = accelerator.gather_for_metrics((logit, label))
        logits.append(logit.cpu().float().numpy())
        labels.append(label.cpu().float().numpy())
    
    losses = torch.cat(losses)
    eval_loss = torch.mean(losses)
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    
    preds = logits[:, 1] > 0.5
    probs = logits[:, 1]
    
    # Calculate enhanced metrics
    metrics_result = calculate_metrics(labels, preds, probs)
    if len(metrics_result) == 10:  # Enhanced metrics with AUC
        eval_acc, eval_prec, eval_recall, eval_f1, eval_tnr, eval_fpr, eval_fnr, eval_mcc, eval_roc_auc, eval_pr_auc = metrics_result
    else:  # Fallback to basic metrics
        eval_acc, eval_prec, eval_recall, eval_f1, eval_tnr, eval_fpr, eval_fnr, eval_mcc = metrics_result
        eval_roc_auc = 0.0
        eval_pr_auc = 0.0
    
    perplexity = eval_loss.clone().detach()

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": eval_acc,
        "eval_prec": eval_prec,
        "eval_recall": eval_recall,
        "eval_f1": eval_f1,
        "eval_tnr": eval_tnr,
        "eval_fpr": eval_fpr,
        "eval_fnr": eval_fnr,
        "eval_mcc": eval_mcc,
        "eval_roc_auc": eval_roc_auc,
        "eval_pr_auc": eval_pr_auc,
    }
    return result

def test(args, accelerator, model, tokenizer):
    """Test the model"""
    # Load model for testing
    if args.use_lora:
        checkpoint_prefix = f'checkpoint-best-f1/{args.project}/{args.model_dir}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        lora_output_dir = os.path.join(output_dir, "lora")
        
        # Load LoRA weights from lora subdirectory first, then fallback to main directory
        if os.path.exists(lora_output_dir):
            logger.info(f"Loading LoRA weights from {lora_output_dir}")
            model.load_adapter(lora_output_dir, adapter_name="default", is_trainable=False)
        elif os.path.exists(output_dir):
            logger.info(f"Loading LoRA weights from {output_dir}")
            model.load_adapter(output_dir, adapter_name="default", is_trainable=False)
        else:
            logger.warning(f"LoRA checkpoint not found at {output_dir} or {lora_output_dir}")
    else:
        # Load full model checkpoint
        checkpoint_prefix = f'checkpoint-best-f1/{args.project}/{args.model_dir}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        accelerator.load_state(output_dir)
        
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = TextDatasetCoT(tokenizer, args, args.test_data_file, verbose=False)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.eval_batch_size)

    eval_dataloader = accelerator.prepare(eval_dataloader)

    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    logits = []   
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs, label = batch
        with torch.no_grad():
            logit = model(inputs)  
        logit, label = accelerator.gather_for_metrics((logit, label))
        logits.append(logit.cpu().float().numpy())
        labels.append(label.cpu().float().numpy())

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    
    preds = logits[:, 1] > 0.5
    vuln_scores = logits[:, 1].tolist()
    probs = logits[:, 1]
    
    os.makedirs(os.path.join(args.output_dir, args.project), exist_ok=True)

    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, args.project, "predictions_qwen_cot.txt"), 'w') as f:
            for example, pred, vs in zip(eval_dataset.examples, preds, vuln_scores):
                if pred:
                    f.write(example.idx + f'\t1\t{vs}\n')
                else:
                    f.write(example.idx + f'\t0\t{vs}\n')

    # Calculate enhanced metrics
    metrics_result = calculate_metrics(labels, preds, probs)
    if len(metrics_result) == 10:  # Enhanced metrics with AUC
        test_acc, test_prec, test_recall, test_f1, test_tnr, test_fpr, test_fnr, test_mcc, test_roc_auc, test_pr_auc = metrics_result
    else:  # Fallback to basic metrics
        test_acc, test_prec, test_recall, test_f1, test_tnr, test_fpr, test_fnr, test_mcc = metrics_result
        test_roc_auc = 0.0
        test_pr_auc = 0.0

    result = {
        "test_acc": test_acc,
        "test_prec": test_prec,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_tnr": test_tnr,
        "test_fpr": test_fpr,
        "test_fnr": test_fnr,
        "test_mcc": test_mcc,
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
    }
    return result 

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--project', type=str, required=True, help="using dataset from this project.")
    parser.add_argument('--model_dir', type=str, required=True, help="directory to store the model weights.")
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    ## Chain of Thought and Few-Shot parameters
    parser.add_argument('--use_few_shot', action='store_true',
                        help="Use few-shot examples in Chain of Thought prompting")
    parser.add_argument('--use_chat_template', action='store_true',
                        help="Use chat template for instruction-tuned models")
    parser.add_argument('--num_few_shot_examples', type=int, default=3,
                        help="Number of few-shot examples to use")
    
    ## LoRA parameters
    parser.add_argument('--use_lora', action='store_true',
                        help="Whether to use LoRA for fine-tuning")
    parser.add_argument('--lora_r', type=int, default=16,
                        help="LoRA rank")
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help="LoRA dropout")
    parser.add_argument('--load_lora_path', type=str, default=None,
                        help="Path to load pre-trained LoRA weights")
    parser.add_argument('--merge_lora', action='store_true',
                        help="Merge LoRA weights with base model before saving")

    ## Model parameters
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-Coder-32B-Instruct", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3")
    parser.add_argument("--block_size", default=1024, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--local_cache_dir", default=None, type=str,
                        help="Local cache directory for models on HPC systems")

    ## Training parameters
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0.05, type=float,
                        help="Linear warmup ratio over all steps.")
    parser.add_argument("--dataloader_num_workers", default=2, type=int,
                        help="Number of workers for dataloader (reduce for HPC)")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=2,
                        help='Limit the total amount of checkpoints')
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=3,
                        help="number of training epochs")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--max_patience', type=int, default=2, help="Max iterations for model with no improvement.")
    parser.add_argument('--force_single_gpu', action='store_true',
                        help="Force single GPU mode, disable distributed training")
    parser.add_argument('--cleanup_cache', action='store_true',
                        help="Clean up cache and unused variables for memory efficiency")
    
    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help="Use Weights & Biases for logging")
    parser.add_argument('--wandb_project', type=str, default="primevul-qwen-cot",
                        help="Wandb project name")
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help="Wandb entity/team name")
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help="Wandb run name (if not provided, will use model_dir)")
    parser.add_argument('--wandb_tags', nargs='+', default=None,
                        help="Wandb tags for the run")

    args = parser.parse_args()

    # Set CUDA device visibility to help with distributed training
    if 'LOCAL_RANK' in os.environ and not args.force_single_gpu:
        local_rank = int(os.environ['LOCAL_RANK'])
        os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

    if args.force_single_gpu:
        print("Forcing single GPU mode...")
        # Clear distributed environment variables to ensure single process
        for key in ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
            if key in os.environ:
                del os.environ[key]
        
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                                cpu=args.no_cuda,
                                project_dir=".")
        device = accelerator.device
        args.n_gpu = 1
        args.device = device
        args.per_gpu_train_batch_size = args.train_batch_size 
        args.per_gpu_eval_batch_size = args.eval_batch_size
    else:
        try:
            print("Attempting to initialize Accelerator in single GPU mode...")
            accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                                    mixed_precision='bf16',
                                    cpu=False) 
            device = accelerator.device
            print(f"Accelerator initialized with {accelerator.num_processes} processes")
            args.n_gpu = 1
            args.device = device
            args.per_gpu_train_batch_size = args.train_batch_size 
            args.per_gpu_eval_batch_size = args.eval_batch_size
            
            if accelerator.num_processes > 1:
                print(f"Warning: Accelerator detected {accelerator.num_processes} processes, but forcing single GPU mode")
        except Exception as e:
            print(f"Distributed training setup failed: {e}")
            print("Falling back to single GPU/CPU mode...")
            
            # Clear distributed environment variables
            for key in ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
                if key in os.environ:
                    del os.environ[key]
            
            accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                                    cpu=args.no_cuda,
                                    project_dir=".")
            device = accelerator.device
            args.n_gpu = 1
            args.device = device
            args.per_gpu_train_batch_size = args.train_batch_size 
            args.per_gpu_eval_batch_size = args.eval_batch_size

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    
    # Only log from main process to avoid duplicate messages
    if accelerator.is_main_process:
        logger.warning("device: %s, n_gpu: %s, distributed training: %s",
                       device, args.n_gpu, bool(args.n_gpu > 1))
        logger.info(accelerator.state, main_process_only=False)

    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0

    # Load Qwen model and tokenizer
    # Setup cache directory for HPC environments
    cache_dir = args.cache_dir if args.cache_dir else None
    if args.local_cache_dir:
        cache_dir = args.local_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Using local cache directory: {cache_dir}")
    
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=cache_dir
    )
    config.num_labels = 2
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.eos_token_id

    # Load the model
    if args.model_name_or_path:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir=cache_dir,
                attn_implementation="flash_attention_2",
                device_map="auto" if torch.cuda.is_available() else None
            )
            logger.info("Successfully loaded model with flash_attention_2")
        except Exception as e:
            logger.warning(f"Flash attention failed ({e}), falling back to eager attention")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    config=config,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    attn_implementation="eager",
                    device_map="auto" if torch.cuda.is_available() else None
                )
                logger.info("Successfully loaded model with eager attention")
            except Exception as e2:
                logger.warning(f"Eager attention also failed ({e2}), loading without device_map")
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    config=config,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )
                logger.info("Successfully loaded model without device_map")
    else:
        model = AutoModelForCausalLM.from_config(config)

    # Wrap model with classifier
    model = DecoderClassifier(model, config, tokenizer, args)

    # Apply LoRA if requested
    if args.use_lora:
        logger.info("Applying LoRA to the model...")
        
        # Print original model parameters
        logger.info("Original model parameters:")
        print_trainable_parameters(model)
        
        # Create LoRA config
        lora_config = create_qwen_lora_config(args)
        
        # Apply LoRA
        if args.load_lora_path:
            logger.info(f"Loading pre-trained LoRA weights from {args.load_lora_path}")
            model = PeftModel.from_pretrained(model, args.load_lora_path)
        else:
            model = get_peft_model(model, lora_config)
        
        # Print LoRA model parameters
        logger.info("LoRA model parameters:")
        trainable_params, all_params = print_trainable_parameters(model)
        
        # Store LoRA info for later wandb logging
        lora_info = {
            "use_lora": True,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "trainable_params": trainable_params,
            "all_params": all_params,
            "trainable_percentage": 100 * trainable_params / all_params,
        }
    else:
        logger.info("Using full fine-tuning (no LoRA)")
        print_trainable_parameters(model)
        lora_info = {"use_lora": False}

    # Initialize Wandb if requested
    if args.use_wandb and accelerator.is_main_process:
        # Set run name if not provided
        if args.wandb_run_name is None:
            cot_str = "cot" if args.use_few_shot else "no_cot"
            fewshot_str = "fewshot" if args.use_few_shot else "zeroshot"
            args.wandb_run_name = f"{args.project}_qwen_{cot_str}_{fewshot_str}_{args.model_dir.replace('/', '_')}"
        
        # Prepare wandb config
        wandb_config = {
            "model_name_or_path": args.model_name_or_path,
            "project": args.project,
            "use_few_shot": args.use_few_shot,
            "use_chat_template": args.use_chat_template,
            "num_few_shot_examples": args.num_few_shot_examples,
            "learning_rate": args.learning_rate,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_train_epochs": args.epoch,
            "block_size": args.block_size,
            "warmup_steps": args.warmup_steps,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "seed": args.seed,
        }
        
        # Add LoRA config if available
        wandb_config.update(lora_info)
        
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=args.wandb_tags,
            config=wandb_config
        )

    # Only log from main process to avoid duplicate messages
    if accelerator.is_main_process:
        logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        # For single GPU mode, we don't need main_process_first wrapper
        is_main = accelerator.is_main_process
        if args.n_gpu == 1:
            train_dataset = TextDatasetCoT(tokenizer, args, args.train_data_file, verbose=is_main)
            eval_dataset = TextDatasetCoT(tokenizer, args, args.eval_data_file, verbose=is_main)
        else:
            with accelerator.main_process_first():
                train_dataset = TextDatasetCoT(tokenizer, args, args.train_data_file, verbose=is_main)
                eval_dataset = TextDatasetCoT(tokenizer, args, args.eval_data_file, verbose=is_main)

        train(args, accelerator, train_dataset, eval_dataset, model, tokenizer)

    # Testing
    if args.do_test:
        result = test(args, accelerator, model, tokenizer) 
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))
        
        # Log test results to wandb
        if args.use_wandb and accelerator.is_main_process:
            test_wandb_log = {}
            for key, value in result.items():
                wandb_key = key.replace("test_", "test/")
                test_wandb_log[wandb_key] = value
            wandb.log(test_wandb_log)              
    
    # Finish wandb run
    if args.use_wandb and accelerator.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main()
