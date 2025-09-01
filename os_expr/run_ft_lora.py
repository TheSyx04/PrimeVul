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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, matthews_corrcoef, auc
import numpy as np
import pandas as pd
import torch
import time
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import json


from tqdm import tqdm, trange
import multiprocessing
from model import Model, DecoderClassifier

cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          LlamaConfig, LlamaModel, LlamaTokenizer,
                          Starcoder2Config, Starcoder2Model, AutoTokenizer,
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

import os

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

MODEL_CLASSES = {
    'codegen': (LlamaConfig, LlamaModel, LlamaTokenizer),
    'starcoder': (Starcoder2Config, Starcoder2Model, AutoTokenizer)
}


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx=str(idx)
        self.label=label

        
def convert_examples_to_features(js,tokenizer,args):
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
        
        # Clean up problematic Unicode escape sequences that can cause tokenizer issues
        import re
        # Replace problematic Unicode escape sequences with placeholders
        code = re.sub(r'\\ud[c-f][0-9a-f]{2}', '?', code, flags=re.IGNORECASE)
        # Also handle any other escape sequences that might cause issues
        code = code.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        
    except Exception as e:
        print(f"Error converting code to string: {e}, using placeholder")
        code = "// empty code"
    
    if args.model_type in ["codegen"]:
        code_tokens = tokenizer.tokenize(code)
        if '</s>' in code_tokens:
            code_tokens = code_tokens[:code_tokens.index('</s>')]
        source_tokens = code_tokens[:args.block_size]
    elif args.model_type in ["starcoder"]:
        try:
            # Use encode instead of tokenize for StarCoder models to avoid issues
            encoded = tokenizer.encode(code, add_special_tokens=False, max_length=args.block_size, truncation=True)
            source_tokens = tokenizer.convert_ids_to_tokens(encoded)
        except Exception as e:
            print(f"Error tokenizing code at line: {e}")
            # Fallback to a simple approach
            source_tokens = [tokenizer.unk_token] * min(10, args.block_size)
        source_tokens = source_tokens[:args.block_size]
    else:
        code_tokens=tokenizer.tokenize(code)
        code_tokens = code_tokens[:args.block_size-2]
        source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    if args.model_type in ["codegen"]:
        source_ids = tokenizer.encode(code.split("</s>")[0], max_length=args.block_size, padding='max_length', truncation=True)
    elif args.model_type in ["starcoder"]:
        # For starcoder, we already have the encoded tokens from above
        source_ids = tokenizer.encode(code, add_special_tokens=False, max_length=args.block_size, padding='max_length', truncation=True)
    else:
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,js['commit_id'],js['label'])

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, verbose=True):
        self.examples = []
        with open(file_path) as f:
            for line_num, line in enumerate(f):
                try:
                    js=json.loads(line.strip())
                    # Debug print only for the first few entries and only if verbose
                    if line_num < 3 and verbose:
                        print(f"Line {line_num}: Processing entry with commit_id={js.get('commit_id', 'N/A')}")
                    self.examples.append(convert_examples_to_features(js,tokenizer,args))
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    print(f"Line content: {line.strip()[:200]}...")  # Truncate long lines
                    raise


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)


def train(args, accelerator, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=4)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.eval_batch_size, num_workers=4)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_steps = args.epoch*num_update_steps_per_epoch
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
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                total_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    # Start timing the entire training
    training_start_time = time.time()
    
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1=0.0
    best_acc=0.0
    patience = 0
    epoch_durations = []  # Store epoch durations for ETA calculation

    train_dataloader, eval_dataloader, model, optimizer, scheduler = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer, scheduler
    )
    model.zero_grad()

    # Modified checkpoint saving for LoRA
    def save_model_checkpoint(output_dir, epoch, step, is_best=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        if args.use_lora:
            # Save LoRA weights only
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir)
            
            # Optionally merge and save full model
            if args.merge_lora and is_best:
                logger.info("Merging LoRA weights with base model...")
                merged_output_dir = os.path.join(output_dir, "merged")
                os.makedirs(merged_output_dir, exist_ok=True)
                
                # Merge LoRA weights
                merged_model = unwrapped_model.merge_and_unload()
                merged_model.save_pretrained(merged_output_dir)
                tokenizer.save_pretrained(merged_output_dir)
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
                
            ###
            # log the first model
            ###
            if step == 0:
                avg_loss = round(train_loss/tr_num,5)
                # train_acc, train_prec, train_recall, train_f1, train_tnr, train_fpr, train_fnr = calculate_metrics(step_labels_lst, step_preds_lst)
                if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(args, accelerator, eval_dataloader, eval_dataset, model, tokenizer, eval_when_training=True)
                    for key, value in results.items():
                        logger.info("  %s = %s", key, round(value, 4))
                    
                    # Log to wandb at step 0
                    if args.use_wandb and accelerator.is_main_process:
                        step_wandb_log = {"step": step, "train/loss": avg_loss}
                        for key, value in results.items():
                            wandb_key = key.replace("eval_", "eval/")
                            step_wandb_log[wandb_key] = value
                        wandb.log(step_wandb_log, step=step)                    
            
            ###
            # log after every logging_steps (e.g., 1000)
            ###
            if (step + 1) % args.logging_steps == 0:
                avg_loss=round(train_loss/tr_num,5)
                # train_acc, train_prec, train_recall, train_f1, train_tnr, train_fpr, train_fnr = calculate_metrics(step_labels_lst, step_preds_lst)
                if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(args, accelerator, eval_dataloader, eval_dataset, model, tokenizer,eval_when_training=True)
                    for key, value in results.items():
                        logger.info("  %s = %s", key, round(value,4))
                    
                    # Log to wandb at logging steps
                    if args.use_wandb and accelerator.is_main_process:
                        step_wandb_log = {"step": step, "train/loss": avg_loss}
                        for key, value in results.items():
                            wandb_key = key.replace("eval_", "eval/")
                            step_wandb_log[wandb_key] = value
                        wandb.log(step_wandb_log, step=step)                    
                
                # Save model checkpoint    
                if results['eval_f1']>best_f1:
                    best_f1=results['eval_f1']
                    logger.info("  "+"*"*20)  
                    logger.info("  Best f1:%s",round(best_f1,4))
                    logger.info("  "+"*"*20)                          
                    
                    checkpoint_prefix = f'checkpoint-best-f1/{args.project}/{args.model_dir}'
                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                    save_model_checkpoint(output_dir, idx, step, is_best=True)
            
            # increment step within the same epoch
            step += 1
            torch.cuda.empty_cache()
        ###
        # log after every epoch
        ###
        avg_loss=round(train_loss/tr_num,5)
        logits_lst=np.concatenate(logits_lst,0)
        labels_lst=np.concatenate(labels_lst,0)
        
        # Calculate train metrics
        if args.model_type in set(['codegen', 'starcoder']):
            train_preds = logits_lst[:,1] > 0.5
            train_probs = logits_lst[:,1]
        else:
            train_preds = logits_lst[:,0] > 0.5
            train_probs = logits_lst[:,0]
        
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
        epoch_duration_minutes = epoch_duration / 60.0
        epoch_durations.append(epoch_duration)
        
        # Calculate estimated time remaining
        if len(epoch_durations) > 0:
            avg_epoch_duration = sum(epoch_durations) / len(epoch_durations)
            remaining_epochs = int(args.num_train_epochs) - (idx + 1)
            eta_seconds = avg_epoch_duration * remaining_epochs
            eta_minutes = eta_seconds / 60.0
            eta_hours = eta_minutes / 60.0
        else:
            eta_seconds = eta_minutes = eta_hours = 0
        
        if args.local_rank in [-1, 0]:
            # Log epoch timing with ETA
            epoch_time_str = format_time_duration(epoch_duration)
            if eta_hours >= 1:
                eta_str = format_time_duration(eta_seconds)
                logger.info(f"Epoch {idx} completed in {epoch_time_str}. ETA: {eta_str}")
            elif remaining_epochs > 0:
                eta_str = format_time_duration(eta_seconds)
                logger.info(f"Epoch {idx} completed in {epoch_time_str}. ETA: {eta_str}")
            else:
                logger.info(f"Epoch {idx} completed in {epoch_time_str}. Final epoch!")
            
            # Log to wandb at epoch level
            if args.use_wandb and accelerator.is_main_process:
                wandb_log = {
                    "epoch": idx,
                    "train/loss": avg_loss,
                    "train/accuracy": train_acc,
                    "train/precision": train_prec,
                    "train/recall": train_recall,
                    "train/f1": train_f1,
                    "train/tnr": train_tnr,
                    "train/fpr": train_fpr,
                    "train/fnr": train_fnr,
                    "train/mcc": train_mcc,
                    "train/roc_auc": train_roc_auc,
                    "train/pr_auc": train_pr_auc,
                    "epoch_duration_seconds": epoch_duration,
                    "epoch_duration_minutes": epoch_duration_minutes,
                    "avg_epoch_duration_seconds": avg_epoch_duration if len(epoch_durations) > 0 else epoch_duration,
                    "eta_seconds": eta_seconds,
                    "eta_minutes": eta_minutes,
                    "eta_hours": eta_hours,
                    "remaining_epochs": remaining_epochs,
                }
                
            if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                results = evaluate(args, accelerator, eval_dataloader, eval_dataset, model, tokenizer,eval_when_training=True)
                for key, value in results.items():
                    logger.info("  %s = %s", key, round(value,4))
                
                # Add evaluation metrics to wandb log
                if args.use_wandb and accelerator.is_main_process:
                    for key, value in results.items():
                        wandb_key = key.replace("eval_", "eval/")
                        wandb_log[wandb_key] = value                    
            
            # save model checkpoint at ep10
            if idx == 9:
                checkpoint_prefix = f'checkpoint-acsac/{args.project}/{args.model_dir}'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                save_model_checkpoint(output_dir, idx, step, is_best=False)
            
            # Save model checkpoint    
            if results['eval_f1']>best_f1:
                best_f1=results['eval_f1']
                logger.info("  "+"*"*20)  
                logger.info("  Best f1:%s",round(best_f1,4))
                logger.info("  "+"*"*20)                          
                
                checkpoint_prefix = f'checkpoint-best-f1/{args.project}/{args.model_dir}'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                save_model_checkpoint(output_dir, idx, step, is_best=True)
                patience = 0
            else:
                patience += 1

            if results['eval_acc']>best_acc:
                best_acc=results['eval_acc']
                logger.info("  "+"*"*20)
                logger.info("  Best acc:%s",round(best_acc,4))
                logger.info("  "+"*"*20)                          
                
                checkpoint_prefix = f'checkpoint-best-acc/{args.project}/{args.model_dir}/lora'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                save_model_checkpoint(output_dir, idx, step, is_best=True)
                patience = 0
            else:
                patience += 1
            
            # Log to wandb at the end of epoch
            if args.use_wandb and accelerator.is_main_process:
                wandb_log.update({
                    "best_f1": best_f1,
                    "best_acc": best_acc,
                    "patience": patience,
                })
                wandb.log(wandb_log, step=step)
        
        if patience == args.max_patience:
            logger.info(f"Reached max patience {args.max_patience}. End training now.")
            if best_f1 == 0.0:
                checkpoint_prefix = f'checkpoint-best-f1/{args.project}/{args.model_dir}'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                save_model_checkpoint(output_dir, idx, step, is_best=False)
            break
    
    # Calculate total training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    total_training_minutes = total_training_time / 60.0
    total_training_hours = total_training_minutes / 60.0
    
    # Log total training time
    if accelerator.is_main_process:
        total_time_str = format_time_duration(total_training_time)
        logger.info(f"***** Training completed in {total_time_str} *****")
        
        # Log total training time to wandb
        if args.use_wandb:
            wandb.log({
                "total_training_time_seconds": total_training_time,
                "total_training_time_minutes": total_training_minutes,
                "total_training_time_hours": total_training_hours,
                "completed_epochs": idx + 1,
                "avg_epoch_duration_final": sum(epoch_durations) / len(epoch_durations) if epoch_durations else 0,
            })
    
    # writer.close()
    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-f1/{args.project}/{args.model_dir}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        
        # Load checkpoint appropriately for LoRA or full model
        if args.use_lora and os.path.exists(os.path.join(output_dir, "adapter_config.json")):
            # Load LoRA weights
            logger.info(f"Loading LoRA checkpoint from {output_dir}")
            model.load_adapter(output_dir, adapter_name="default", is_trainable=False)
        else:
            # Load full model checkpoint
            logger.info(f"Loading full model checkpoint from {output_dir}")
            accelerator.load_state(output_dir)
            
        result = test(args, accelerator, model, tokenizer) 
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
        
        # Log test results to wandb
        if args.use_wandb and accelerator.is_main_process:
            test_wandb_log = {}
            for key, value in result.items():
                wandb_key = key.replace("test_", "test/")
                test_wandb_log[wandb_key] = value
            wandb.log(test_wandb_log)              

def calculate_metrics(labels, preds, probs=None):
    acc=accuracy_score(labels, preds)
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

def get_recall_at_k_percent_effort(percent_effort, result_df_arg, real_buggy_commits):
    """Calculate recall at k percent effort for effort-aware metrics"""
    cum_LOC_k_percent = (percent_effort / 100) * result_df_arg.iloc[-1]['cum_LOC']
    buggy_line_k_percent = result_df_arg[result_df_arg['cum_LOC'] <= cum_LOC_k_percent]
    buggy_commit = buggy_line_k_percent[buggy_line_k_percent['label'] == 1]
    recall_k_percent_effort = len(buggy_commit) / float(len(real_buggy_commits))
    return recall_k_percent_effort

def eval_metrics_comprehensive(result_df, model, columns):
    """Comprehensive metrics evaluation including effort-aware metrics if LOC data is available"""
    pred = result_df['prediction']
    y_test = result_df['label']
    y_proba = result_df["probability"]
    
    # find AUC
    roc_auc = roc_auc_score(y_true=y_test, y_score=y_proba)
    precisions, recalls, _ = precision_recall_curve(y_true=y_test, probas_pred=y_proba)
    pr_auc = auc(recalls, precisions)

    # find metrics
    acc = accuracy_score(y_true=y_test, y_pred=pred)
    f1 = f1_score(y_true=y_test, y_pred=pred)
    prc = precision_score(y_true=y_test, y_pred=pred)
    rc = recall_score(y_true=y_test, y_pred=pred)
    mcc = matthews_corrcoef(y_true=y_test, y_pred=pred)

    if "LOC" not in result_df.columns:
        import pandas as pd
        metric_df = pd.DataFrame([[roc_auc, pr_auc, acc, f1, prc, rc, mcc]], 
                             columns=columns, index=[model])
        return metric_df

    # find Effort metrics
    import math
    result_df['defect_density'] = result_df['probability'] / result_df['LOC']  # predicted defect density
    result_df['actual_defect_density'] = result_df['label'] / result_df['LOC']  # defect density

    result_df = result_df.sort_values(by='defect_density', ascending=False)
    actual_result_df = result_df.sort_values(by='actual_defect_density', ascending=False)
    actual_worst_result_df = result_df.sort_values(by='actual_defect_density', ascending=True)

    result_df['cum_LOC'] = result_df['LOC'].cumsum()
    actual_result_df['cum_LOC'] = actual_result_df['LOC'].cumsum()
    actual_worst_result_df['cum_LOC'] = actual_worst_result_df['LOC'].cumsum()
    real_buggy_commits = result_df[result_df['label'] == 1]
    
    # find Recall@20%Effort
    cum_LOC_20_percent = 0.2 * result_df.iloc[-1]['cum_LOC']
    buggy_line_20_percent = result_df[result_df['cum_LOC'] <= cum_LOC_20_percent]
    buggy_commit = buggy_line_20_percent[buggy_line_20_percent['label'] == 1]
    recall_20_percent_effort = len(buggy_commit) / float(len(real_buggy_commits))

    # find Effort@20%Recall
    buggy_20_percent = real_buggy_commits.head(math.ceil(0.2 * len(real_buggy_commits)))
    buggy_20_percent_LOC = buggy_20_percent.iloc[-1]['cum_LOC']
    effort_at_20_percent_LOC_recall = int(buggy_20_percent_LOC) / float(result_df.iloc[-1]['cum_LOC'])

    # find P_opt
    percent_effort_list = []
    predicted_recall_at_percent_effort_list = []
    actual_recall_at_percent_effort_list = []
    actual_worst_recall_at_percent_effort_list = []

    for percent_effort in np.arange(10, 101, 10):
        predicted_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, result_df, real_buggy_commits)
        actual_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_result_df, real_buggy_commits)
        actual_worst_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_worst_result_df, real_buggy_commits)

        percent_effort_list.append(percent_effort / 100)

        predicted_recall_at_percent_effort_list.append(predicted_recall_k_percent_effort)
        actual_recall_at_percent_effort_list.append(actual_recall_k_percent_effort)
        actual_worst_recall_at_percent_effort_list.append(actual_worst_recall_k_percent_effort)

    p_opt = 1 - ((auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                  auc(percent_effort_list, predicted_recall_at_percent_effort_list)) /
                 (auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                  auc(percent_effort_list, actual_worst_recall_at_percent_effort_list)))
    
    import pandas as pd
    metric_df = pd.DataFrame([[roc_auc, pr_auc, acc, f1, prc, rc, mcc, effort_at_20_percent_LOC_recall, recall_20_percent_effort, p_opt]], 
                             columns=columns, index=[model])
    return metric_df

def get_comprehensive_metrics(predict_df, model, features_file=None):
    """Get comprehensive metrics including effort-aware metrics if LOC data is available"""
    if features_file is not None:
        columns = ["roc_auc", "pr_auc", "accuracy", "f1_score", "precision", "recall", "mcc", "Effort@20", "Recall@20", "Popt"]
    else:
        columns = ["roc_auc", "pr_auc", "accuracy", "f1_score", "precision", "recall", "mcc"]
              
    predict_df.columns = ["commit_id", "probability", "prediction", "label"]
    predict_df['prediction'] = predict_df['prediction'].apply(lambda x: float(bool(x)))
    
    if features_file is not None:
        import pandas as pd
        features_df = pd.read_json(features_file, lines=True)
        assert all(col in features_df.columns for col in ["la", "ld"]), "Provide add lines (la), and delete lines (ld) in size set"
        
        LOC_df = features_df[["commit_id", "la", "ld"]].copy()
        LOC_df["LOC"] = LOC_df["la"] + LOC_df["ld"]
        LOC_df = LOC_df[["commit_id", "LOC"]]

        predict_df = pd.merge(predict_df, LOC_df, how="inner", on="commit_id")
        
    return eval_metrics_comprehensive(predict_df, model, columns)

def evaluate(args, accelerator, eval_dataloader, eval_dataset, model, tokenizer, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
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
    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    if args.model_type in set(['codegen', 'starcoder']):
        preds=logits[:,1]>0.5
        probs=logits[:,1]
    else:
        preds=logits[:,0]>0.5
        probs=logits[:,0]
    
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

def load_data(jsonl_path):
    '''
    Load data from jsonl file
    '''
    with open(jsonl_path, 'r') as f:
        data = [json.loads(line.rstrip()) for line in f]
    return data

def analyze_predictions(predictions_path, data_path):
    data = load_data(data_path)
    predictions = []
    with open(predictions_path, 'r') as f:
        for line in f:
            id, pred = line.strip().split('\t')
            id = int(id)
            pred = int(pred)
            predictions.append({"id": id, "pred": pred})
    
    assert len(predictions) == len(data)
    assert len(predictions) % 2 == 0
    num_acc = 0
    num_all_vul = 0
    num_all_sec = 0
    num_reversed = 0
    for i in range(len(predictions) // 2):
        assert predictions[2 * i]['id'] == data[2 * i]['idx']
        assert predictions[2 * i + 1]['id'] == data[2 * i + 1]['idx']
        assert data[2 * i]["commit_id"] == data[2 * i + 1]["commit_id"]
        if predictions[2 * i]['pred'] == data[2 * i]['target'] and predictions[2 * i + 1]['pred'] == data[2 * i + 1]['target']:
            num_acc += 1
        elif predictions[2 * i]['pred'] == predictions[2 * i + 1]['pred'] and predictions[2 * i]['pred'] == 0:
            num_all_sec += 1
        elif predictions[2 * i]['pred'] == predictions[2 * i + 1]['pred'] and predictions[2 * i]['pred'] == 1:
            num_all_vul += 1
        else:
            num_reversed += 1

    print("num_acc: {}".format(num_acc))
    print("num_all_sec: {}".format(num_all_sec))
    print("num_all_vul: {}".format(num_all_vul))
    print("num_reversed: {}".format(num_reversed))

def test(args, accelerator, model, tokenizer):
    # Load model for testing
    if args.use_lora:
        checkpoint_prefix = f'checkpoint-best-f1/{args.project}/{args.model_dir}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        
        # Load LoRA weights
        if os.path.exists(output_dir):
            logger.info(f"Loading LoRA weights from {output_dir}")
            # For LoRA models, we need to get the base model first
            if hasattr(model, 'peft_config'):
                # Model is already a PEFT model, load from checkpoint
                model.load_adapter(output_dir, adapter_name="default", is_trainable=False)
            else:
                # Model is base model, apply PEFT
                model = PeftModel.from_pretrained(model, output_dir)
        else:
            logger.warning(f"LoRA checkpoint not found at {output_dir}")
    else:
        # Load full model checkpoint
        checkpoint_prefix = f'checkpoint-best-f1/{args.project}/{args.model_dir}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        accelerator.load_state(output_dir)
        
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file, verbose=False)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.eval_batch_size)

    eval_dataloader = accelerator.prepare(eval_dataloader)

    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    logits=[]   
    labels=[]
    for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
        inputs, label = batch
        with torch.no_grad():
            logit = model(inputs)  
        logit, label = accelerator.gather_for_metrics((logit, label))
        logits.append(logit.cpu().float().numpy())
        labels.append(label.cpu().float().numpy())

    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    if args.model_type in set(['codegen', 'starcoder']):
        preds=logits[:,1]>0.5
        vuln_scores = logits[:,1].tolist()
        probs = logits[:,1]
    else:
        preds=logits[:,0]>0.5
        vuln_scores = logits[:,0].tolist()
        probs = logits[:,0]
    os.makedirs(os.path.join(args.output_dir, args.project), exist_ok=True)

    if accelerator.is_main_process:
        if args.test_cwe == None:
            with open(os.path.join(args.output_dir, args.project, "predictions.txt"),'w') as f:
                for example,pred,vs in zip(eval_dataset.examples,preds,vuln_scores):
                    if pred:
                        f.write(example.idx+f'\t1\t{vs}\n')
                    else:
                        f.write(example.idx+f'\t0\t{vs}\n')
        else:
            with open(os.path.join(args.output_dir, args.project, f"predictions_{args.test_cwe}.txt"),'w') as f:
                for example,pred in zip(eval_dataset.examples,preds):
                    if pred:
                        f.write(example.idx+'\t1\n')
                    else:
                        f.write(example.idx+'\t0\n')   

    # Calculate enhanced metrics
    metrics_result = calculate_metrics(labels, preds, probs)
    if len(metrics_result) == 10:  # Enhanced metrics with AUC
        test_acc, test_prec, test_recall, test_f1, test_tnr, test_fpr, test_fnr, test_mcc, test_roc_auc, test_pr_auc = metrics_result
    else:  # Fallback to basic metrics
        test_acc, test_prec, test_recall, test_f1, test_tnr, test_fpr, test_fnr, test_mcc = metrics_result
        test_roc_auc = 0.0
        test_pr_auc = 0.0

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        try:
            analyze_predictions(os.path.join(args.output_dir, args.project, "predictions.txt"), args.test_data_file)
        except Exception as e:
            print(e)
            pass
            

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
    
def test_prob(args, model, tokenizer):
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file, verbose=False)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]   
    labels=[]
    for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
        inputs = batch[0].to(args.device)        
        label=batch[1].to(args.device) 
        with torch.no_grad():
            logit = model(inputs)
            logits.append(logit.cpu().float().numpy())
            labels.append(label.cpu().float().numpy())

    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    soft_preds=logits[:,0]
    os.makedirs(os.path.join(args.output_dir, args.project), exist_ok=True)
    if args.test_cwe == None:
        with open(os.path.join(args.output_dir, args.project, "predictions.txt"),'w') as f:
            for example,pred in zip(eval_dataset.examples,soft_preds):
                f.write(example.idx+'\t%f\n' % pred)
    else:
        with open(os.path.join(args.output_dir, args.project, f"predictions_{args.test_cwe}.txt"),'w') as f:
            for example,pred in zip(eval_dataset.examples,soft_preds):
                f.write(example.idx+'\t%f\n' % pred)

def create_lora_config(args):
    """Create LoRA configuration based on model type and arguments"""
    if args.model_type == "codegen":
        # LLaMA/CodeGen models - target attention and MLP layers
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif args.model_type == "starcoder":
        # StarCoder models - target attention layers
        target_modules = ["q_proj", "k_proj", "v_proj", "dense"]
    else:
        # Default for other models
        target_modules = ["query", "key", "value", "dense"]
    
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,  # For classification tasks
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",  # or "all" or "lora_only"
        modules_to_save=["classifier", "score"] if hasattr(args, 'modules_to_save') else ["classifier", "score"],
    )
    
    return lora_config

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
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
    
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--project', type=str, required=True, help="using dataset from this project.")
    parser.add_argument('--train_project', type=str, required=False, help="using training dataset from this project.")
    parser.add_argument('--model_dir', type=str, required=True, help="directory to store the model weights.")
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--test_cwe', type=str, default=None, required=False, help="using dataset from this CWE for testing.")
    
    # run dir
    parser.add_argument('--run_dir', type=str, default="runs", help="parent directory to store run stats.")

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

    ## Other parameters
    parser.add_argument("--max_source_length", default=400, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default="codegen", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test_prob", action='store_true',
                        help="Whether to run eval and save the prediciton probabilities.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--weighted_sampler", action='store_true',
                        help="Whether to do project balanced sampler using WeightedRandomSampler.")
    # Soft F1 loss function
    parser.add_argument("--soft_f1", action='store_true',
                        help="Use soft f1 loss instead of regular cross entropy loss.")
    parser.add_argument("--class_weight", action='store_true',
                        help="Use class weight in the regular cross entropy loss.")
    parser.add_argument("--vul_weight", default=1.0, type=float,
                        help="Weight for the vulnerable class in the regular cross entropy loss.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Linear warmup ratio over all steps.")

    parser.add_argument('--logging_steps', type=int, default=1000,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--max_patience', type=int, default=-1, help="Max iterations for model with no improvement.")
    parser.add_argument('--force_single_gpu', action='store_true',
                        help="Force single GPU mode, disable distributed training")
    
    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help="Use Weights & Biases for logging")
    parser.add_argument('--wandb_project', type=str, default="primevul-training",
                        help="Wandb project name")
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help="Wandb entity/team name")
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help="Wandb run name (if not provided, will use model_dir)")
    parser.add_argument('--wandb_tags', nargs='+', default=None,
                        help="Wandb tags for the run")

    args = parser.parse_args()

    # Set CUDA device visibility to help with distributed training
    import os
    if 'LOCAL_RANK' in os.environ and not args.force_single_gpu:
        local_rank = int(os.environ['LOCAL_RANK'])
        os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

    if args.force_single_gpu:
        print("Forcing single GPU mode...")
        # Clear distributed environment variables to ensure single process
        import os
        for key in ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
            if key in os.environ:
                del os.environ[key]
        
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                                cpu=args.no_cuda,
                                project_dir=".")
        device = accelerator.device
        args.n_gpu = 1
        args.device = device
        args.per_gpu_train_batch_size=args.train_batch_size 
        args.per_gpu_eval_batch_size=args.eval_batch_size
    else:
        try:
            # Force single GPU by explicitly setting CPU mode for multi-process detection
            print("Attempting to initialize Accelerator in single GPU mode...")
            accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                                    mixed_precision='bf16',
                                    cpu=False) 
            device = accelerator.device
            print(f"Accelerator initialized with {accelerator.num_processes} processes")
            # Force single GPU regardless of accelerator.num_processes
            args.n_gpu = 1
            args.device = device
            args.per_gpu_train_batch_size=args.train_batch_size 
            args.per_gpu_eval_batch_size=args.eval_batch_size
            
            # If we still have multiple processes, force single mode
            if accelerator.num_processes > 1:
                print(f"Warning: Accelerator detected {accelerator.num_processes} processes, but forcing single GPU mode")
                print("If you see CUDA OOM errors, this means distributed training is still active")
                print("Consider using --force_single_gpu flag or running without accelerate launch")
        except Exception as e:
            # Fallback to single GPU/CPU mode if distributed setup fails
            print(f"Distributed training setup failed: {e}")
            print("Falling back to single GPU/CPU mode...")
            
            # Create a basic accelerator without distributed training
            # Force single process by clearing distributed environment variables
            import os
            for key in ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
                if key in os.environ:
                    del os.environ[key]
            
            accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                                    cpu=args.no_cuda,
                                    project_dir=".")
            device = accelerator.device
            args.n_gpu = 1
            args.device = device
            args.per_gpu_train_batch_size=args.train_batch_size 
            args.per_gpu_eval_batch_size=args.eval_batch_size
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
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    
    # Explicitly disable auto flash attention detection to prevent compatibility issues
    if hasattr(config, '_attn_implementation_autoset') and hasattr(config, '_attn_implementation'):
        config._attn_implementation_autoset = False
    
    # Disable use_cache for StarCoder2 models to prevent initialization errors
    if args.model_type in ["starcoder"]:
        config.use_cache = False
    
    config.num_labels = 2
    if args.model_type not in ["codegen"]:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                    do_lower_case=args.do_lower_case,
                                                    cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                    trust_remote_code=True)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    if args.model_name_or_path:
        if args.model_type in ["starcoder"]:
            try:
                # Explicitly import flash_attn to check availability in current environment
                import flash_attn
                import sys
                logger.info(f"flash_attn version {flash_attn.__version__} detected on Python {sys.version_info}, using flash_attention_2")
                
                # Try setting attention implementation in config first
                if hasattr(config, '_attn_implementation'):
                    config._attn_implementation = "flash_attention_2"
                
                model = model_class.from_pretrained(args.model_name_or_path,
                                                    config=config,
                                                    torch_dtype = torch.bfloat16,
                                                    attn_implementation = "flash_attention_2")
            except (ImportError, ModuleNotFoundError, RuntimeError, Exception) as e:
                # More comprehensive exception handling for flash attention issues
                if accelerator.is_main_process:
                    logger.warning(f"flash_attn import/initialization failed ({type(e).__name__}: {e}), falling back to eager attention")
                    logger.info("This might be due to Python 3.13 compatibility issues with flash-attn")
                
                try:
                    # Explicitly set eager attention in config
                    if hasattr(config, '_attn_implementation'):
                        config._attn_implementation = "eager"
                    
                    model = model_class.from_pretrained(args.model_name_or_path,
                                                        config=config,
                                                        torch_dtype = torch.bfloat16,
                                                        attn_implementation = "eager")
                except Exception as e2:
                    # Final fallback without specifying attention implementation
                    if accelerator.is_main_process:
                        logger.warning(f"Eager attention also failed ({type(e2).__name__}: {e2}), using default attention")
                    model = model_class.from_pretrained(args.model_name_or_path,
                                                        config=config,
                                                        torch_dtype = torch.bfloat16)
        else:
            model = model_class.from_pretrained(args.model_name_or_path,
                                                torch_dtype = torch.bfloat16,)
    else:
        model = model_class(config)
    

    # Set up pad token properly for different model types
    if args.model_type in ['codegen', 'starcoder']:
        if tokenizer.pad_token is None:
            # Use eos_token as pad_token if pad_token is not available
            tokenizer.pad_token = tokenizer.eos_token
            config.pad_token_id = tokenizer.eos_token_id
        else:
            config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    else:
        config.pad_token_id = tokenizer(tokenizer.pad_token, truncation=True)['input_ids'][0]
    
    if args.model_type in ['codegen', 'starcoder']:
        model = DecoderClassifier(model,config,tokenizer,args)
    else:
        model = Model(model,config,tokenizer,args)

    # Apply LoRA if requested
    if args.use_lora:
        logger.info("Applying LoRA to the model...")
        
        # Print original model parameters
        logger.info("Original model parameters:")
        print_trainable_parameters(model)
        
        # Create LoRA config
        lora_config = create_lora_config(args)
        
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
        lora_config = {
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
        lora_config = {"use_lora": False}

    # Initialize Wandb if requested
    if args.use_wandb and accelerator.is_main_process:
        # Set run name if not provided
        if args.wandb_run_name is None:
            args.wandb_run_name = f"{args.project}_{args.model_dir.replace('/', '_')}_epoch{args.epoch}"
        
        # Prepare wandb config
        wandb_config = {
            "model_type": args.model_type,
            "model_name_or_path": args.model_name_or_path,
            "project": args.project,
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
        if 'lora_config' in locals():
            wandb_config.update(lora_config)
        
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
            train_dataset = TextDataset(tokenizer, args, args.train_data_file, verbose=is_main)
            eval_dataset = TextDataset(tokenizer, args, args.eval_data_file, verbose=is_main)
        else:
            with accelerator.main_process_first():
                train_dataset = TextDataset(tokenizer, args, args.train_data_file, verbose=is_main)
                eval_dataset = TextDataset(tokenizer, args, args.eval_data_file, verbose=is_main)

        train(args, accelerator, train_dataset, eval_dataset, model, tokenizer)


    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoint_prefix = f'checkpoint-best-f1/{args.project}/{args.model_dir}/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))      
        model.to(args.device)
    
    if args.do_test_prob and args.local_rank in [-1, 0]:
        checkpoint_prefix = f'checkpoint-best-f1/{args.project}/{args.model_dir}/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))                  
        model.to(args.device)
        test_prob(args, model, tokenizer)
    
    # Finish wandb run
    if args.use_wandb and accelerator.is_main_process:
        wandb.finish()
    
    return results


if __name__ == "__main__":
    main()