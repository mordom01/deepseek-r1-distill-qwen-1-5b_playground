"""
DeepSeek R1 Model Benchmarking System

This module provides comprehensive benchmarking capabilities for the DeepSeek R1 model
using various datasets from Hugging Face. It supports multiple evaluation tasks and
saves detailed results for analysis.
"""

import sys
import json
import time
import logging
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor, as_completed

# Simple logging setup
import os
from datetime import datetime

home_dir_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0,home_dir_path)

def setup_simple_logging():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/benchmark_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_simple_logging()

# Your existing modules
from model_loader import ModelLoader


def log_gpu_memory(stage=""):
    """Log essential GPU information using nvidia-smi and PyTorch memory stats"""
    try:
        # Get essential GPU info in a clean format
        gpu_query = [
            'nvidia-smi', 
            '--query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw,utilization.gpu',
            '--format=csv,noheader,nounits'
        ]
        
        process_query = [
            'nvidia-smi',
            '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader,nounits'
        ]
        
        # Get GPU stats
        gpu_result = subprocess.run(gpu_query, capture_output=True, text=True)
        if gpu_result.returncode == 0:
            lines = gpu_result.stdout.strip().split('\n')
            logger.info(f"=== GPU Status {stage} ===")
            for i, line in enumerate(lines):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        name, mem_used, mem_total, temp, power, util = parts[:6]
                        logger.info(f"GPU {i}: {name}")
                        logger.info(f"  Memory: {mem_used}MB / {mem_total}MB ({float(mem_used)/float(mem_total)*100:.1f}%)")
                        logger.info(f"  Temp: {temp}°C | Power: {power}W | Util: {util}%")
        
        # Get process info
        proc_result = subprocess.run(process_query, capture_output=True, text=True)
        if proc_result.returncode == 0 and proc_result.stdout.strip():
            logger.info("=== GPU Processes ===")
            proc_lines = proc_result.stdout.strip().split('\n')
            for line in proc_lines:
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        pid, proc_name, mem_used = parts[:3]
                        logger.info(f"  PID {pid}: {proc_name} ({mem_used}MB)")
        
        # Add PyTorch memory stats if CUDA is available
        if torch.cuda.is_available():
            logger.info("=== PyTorch CUDA Memory ===")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
                reserved = torch.cuda.memory_reserved(i) / 1024**2   # MB
                max_allocated = torch.cuda.max_memory_allocated(i) / 1024**2  # MB
                logger.info(f"  GPU {i}: Allocated: {allocated:.1f}MB | Reserved: {reserved:.1f}MB | Peak: {max_allocated:.1f}MB")
        
        logger.info("==================")
        
    except FileNotFoundError:
        logger.warning("nvidia-smi not found - GPU monitoring unavailable")
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e}")


class DeepSeekBenchmark:
    """
    Comprehensive benchmarking system for DeepSeek R1 model
    """
    
    def __init__(self, output_dir: str = "benchmark_results", global_seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set global seed for reproducibility
        self.global_seed = global_seed
        torch.manual_seed(global_seed)
        logger.info(f"Set global seed to {global_seed} for reproducible results")
        
        # Log initial GPU memory
        log_gpu_memory("(Initial)")
        
        # Load model once
        self.tokenizer, self.model = ModelLoader.load_model()
        
        # Log GPU memory after model loading
        logger.info("GPU memory after model loading:")
        log_gpu_memory("(Post-Model Load)")
        
        # Benchmark configurations - increased token limits for complete answers
        self.configs = {
            "math_reasoning": {
                "temperature": 0.1,  # Very low for deterministic math
                "top_k": 20,
                "top_p": 0.8,
                "max_new_tokens": 600,
                "min_new_tokens": 100,
                "do_sample": True,
                "stop_sequences": ["</answer>", "\n\nProblem:", "\n\nQuestion:", "Let me solve another", "Now let's solve"]
            },
            "general_reasoning": {
                "temperature": 0.1,  # Low for consistency
                "top_k": 20,
                "top_p": 0.8,
                "max_new_tokens": 400,
                "min_new_tokens": 80,
                "do_sample": True,
                "stop_sequences": ["</answer>", "\n\nAnother", "\n\nNext", "Let me try again"]
            },
            "knowledge_qa": {
                "temperature": 0.05,  # Very deterministic for facts
                "top_k": 15,
                "top_p": 0.7,
                "max_new_tokens": 300,
                "min_new_tokens": 60,
                "do_sample": True,
                "stop_sequences": ["</answer>", "\n\nQuestion:", "\n\nAnother question"]
            }
        }
    
    def load_benchmark_dataset(self, dataset_name: str, subset: Optional[str] = None, 
                             split: str = "test", limit: Optional[int] = None) -> List[Dict]:
        """
        Load and prepare benchmark dataset from Hugging Face
        
        Args:
            dataset_name: HF dataset identifier
            subset: Subset/configuration name if applicable
            split: Dataset split to use
            limit: Maximum number of examples to load
            
        Returns:
            List of formatted examples
        """
        logger.info(f"Loading dataset: {dataset_name}")
        
        try:
            # Some datasets need trust_remote_code=True
            trust_remote_code = dataset_name in ["hellaswag", "truthful_qa"]
            
            if subset:
                dataset = load_dataset(dataset_name, subset, split=split, trust_remote_code=trust_remote_code)
            else:
                dataset = load_dataset(dataset_name, split=split, trust_remote_code=trust_remote_code)
            
            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))
            
            examples = []
            for i, example in enumerate(dataset):
                formatted = self._format_example(dataset_name, example, i)
                if formatted:
                    examples.append(formatted)
            
            logger.info(f"Loaded {len(examples)} examples from {dataset_name}")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            return []
    
    def _format_example(self, dataset_name: str, example: Dict, index: int) -> Optional[Dict]:
        """Format examples based on dataset type with proper end markers"""
        
        if dataset_name == "gsm8k":
            return {
                "id": f"gsm8k_{index}",
                "prompt": f"Solve this math problem step by step:\n\n{example['question']}\n\nAfter your reasoning, provide the final numerical answer and end with </answer>",
                "expected_answer": example.get("answer", ""),
                "dataset": "gsm8k",
                "task_type": "math_reasoning"
            }
        
        elif dataset_name == "cais/mmlu":
            choices = example.get("choices", [])
            choice_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            return {
                "id": f"mmlu_{index}",
                "prompt": f"Question: {example['question']}\n\nChoices:\n{choice_text}\n\nProvide your reasoning, then give your final answer as the letter (A, B, C, or D). End with </answer>",
                "expected_answer": chr(65 + example.get("answer", 0)),
                "dataset": "mmlu",
                "task_type": "knowledge_qa",
                "subject": example.get("subject", "unknown")
            }
        
        elif dataset_name == "hellaswag":
            # Handle potential string/int conversion issues
            ctx = str(example.get('ctx', ''))
            endings = example.get('endings', [])
            label = example.get('label', 0)
            
            # Ensure all endings are strings
            endings = [str(ending) for ending in endings]
            
            return {
                "id": f"hellaswag_{index}",
                "prompt": f"Complete this scenario with the most logical continuation:\n\n{ctx}\n\nOptions:\n" + 
                         "\n".join([f"{i+1}. {ending}" for i, ending in enumerate(endings)]) +
                         "\n\nExplain your reasoning, then choose the number (1, 2, 3, or 4) that best completes the scenario. End with </answer>",
                "expected_answer": str(int(label) + 1),
                "dataset": "hellaswag", 
                "task_type": "general_reasoning"
            }
        
        elif dataset_name == "ai2_arc":
            choices = example.get("choices", {}).get("text", [])
            choice_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            return {
                "id": f"arc_{index}",
                "prompt": f"Science Question: {example['question']}\n\nChoices:\n{choice_text}\n\nProvide your reasoning, then give the letter of your answer. End with </answer>",
                "expected_answer": example.get("answerKey", ""),
                "dataset": "ai2_arc",
                "task_type": "knowledge_qa"
            }
        
        elif dataset_name == "truthful_qa":
            return {
                "id": f"truthfulqa_{index}",
                "prompt": f"Question: {example['question']}\n\nProvide a truthful and accurate answer. End with </answer>",
                "expected_answer": example.get("best_answer", ""),
                "dataset": "truthful_qa",
                "task_type": "knowledge_qa"
            }
        
        return None
    
    def generate_response(self, prompt: str, config_name: str = "general_reasoning") -> Dict[str, Any]:
        """
        Generate response using specified configuration with deterministic settings
        """
        config = self.configs.get(config_name, self.configs["general_reasoning"])
        
        # Add thinking prompt for DeepSeek R1 with clearer instructions
        thinking_prompt = f"""
        {prompt}
        
        <think>
        Let me think through this step by step:
        """
        
        start_time = time.time()
        
        try:
            # Set seed for deterministic generation (based on prompt hash for consistency)
            seed = hash(prompt) % 2**32
            torch.manual_seed(seed)
            
            inputs = self.tokenizer(thinking_prompt, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Clear PyTorch memory cache and reset peak memory tracking
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # Log memory just before inference (debug level to avoid spam)
            if logger.isEnabledFor(logging.DEBUG):
                log_gpu_memory("(Pre-Inference)")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config["max_new_tokens"],
                    min_new_tokens=config.get("min_new_tokens", 20),
                    temperature=config["temperature"],
                    top_k=config["top_k"],
                    top_p=config["top_p"],
                    do_sample=config["do_sample"],
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1  # Slight penalty to reduce repetition
                )
            
            # Log memory immediately after inference (before cleanup) - debug level
            if logger.isEnabledFor(logging.DEBUG):
                log_gpu_memory("(Post-Inference)")
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract thinking and final answer properly
            thinking_part = ""
            final_answer = full_response
            
            if "<think>" in full_response:
                if "</think>" in full_response:
                    # Extract thinking part (between <think> and </think>)
                    thinking_start = full_response.find("<think>") + len("<think>")
                    thinking_end = full_response.find("</think>")
                    thinking_part = full_response[thinking_start:thinking_end].strip()
                    
                    # Remove any </answer> markers that leaked into thinking
                    thinking_part = thinking_part.replace("</answer>", "").strip()
                    
                    # Extract answer part (everything after </think>)
                    final_answer = full_response[thinking_end + len("</think>"):].strip()
                else:
                    # If no closing </think>, treat everything after <think> as thinking
                    thinking_part = full_response.split("<think>")[1].strip()
                    final_answer = ""
            
            # Apply stop sequences and end marker truncation to the answer only
            final_answer = self._apply_stop_sequences(final_answer, config.get("stop_sequences", []))
            
            # Log memory after processing (cleanup phase) - debug level
            if logger.isEnabledFor(logging.DEBUG):
                log_gpu_memory("(Post-Processing)")
            
            processing_time = time.time() - start_time
            
            # Get peak memory usage for this generation
            peak_memory_mb = 0
            if torch.cuda.is_available():
                peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            
            return {
                "thinking": thinking_part,
                "answer": final_answer,
                "full_response": full_response,
                "processing_time": processing_time,
                "config_used": config_name,
                "success": True,
                "seed_used": seed,
                "peak_memory_mb": peak_memory_mb
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            
            # Get peak memory even for failed cases
            peak_memory_mb = 0
            if torch.cuda.is_available():
                peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
                
            return {
                "thinking": "",
                "answer": f"ERROR: {str(e)}",
                "full_response": "",
                "processing_time": time.time() - start_time,
                "config_used": config_name,
                "success": False,
                "seed_used": None,
                "peak_memory_mb": peak_memory_mb
            }
    
    def _apply_stop_sequences(self, text: str, stop_sequences: List[str]) -> str:
        """
        Apply stop sequences and end marker truncation
        """
        if not text:
            return text
        
        # Handle </answer> marker - look for it at a reasonable ending point
        if "</answer>" in text:
            # Find all occurrences of </answer>
            answer_positions = []
            start = 0
            while True:
                pos = text.find("</answer>", start)
                if pos == -1:
                    break
                answer_positions.append(pos)
                start = pos + 1
            
            # Use the last occurrence that comes after substantial content
            for pos in reversed(answer_positions):
                if pos > 50:  # Ensure there's substantial content before the marker
                    text = text[:pos + len("</answer>")]
                    break
        
        # Apply other stop sequences
        for stop_seq in stop_sequences:
            if stop_seq in text and stop_seq != "</answer>":
                # Find the position and truncate there
                text = text.split(stop_seq)[0].strip()
                break
        
        # Remove common rambling patterns that indicate restarting
        rambling_patterns = [
            "\n\nLet me solve this step by step again:",
            "\n\nTo solve this problem again:",
            "\n\nNow let me work through this again:",
            "\n\nI'll solve this again:",
            "\n\nLet me try a different approach:",
            "\n\nAlternatively, let me solve this:",
            "\n\nLet's break down the problem again:",
            "\n\n**Given:** (again)",  # If it restarts with formatting
        ]
        
        for pattern in rambling_patterns:
            if pattern in text:
                # Only truncate if this appears after substantial content
                parts = text.split(pattern)
                if len(parts[0]) > 100:  # Keep if there's substantial content before
                    text = parts[0].strip()
                    break
        
        # Clean up any incomplete sentences at the end if they seem truncated
        text = text.strip()
        
        # If text ends abruptly mid-sentence, try to end at last complete sentence
        # But only if it looks like an incomplete mathematical expression or sentence
        incomplete_patterns = [
            r'= $',  # Ends with equals sign
            r'\\text\{[^}]*$',  # Incomplete LaTeX
            r'\\\[$',  # Incomplete LaTeX bracket
            r':\s*$',  # Ends with colon and whitespace
            r',\s*$',  # Ends with comma and whitespace
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, text):
                # Find the last complete sentence
                last_period = text.rfind('.')
                last_exclaim = text.rfind('!')
                last_question = text.rfind('?')
                last_complete = max(last_period, last_exclaim, last_question)
                
                if last_complete > len(text) * 0.6:  # Only if we're not cutting too much
                    text = text[:last_complete + 1]
                    break
        
        return text.strip()
    
    def run_batch_evaluation(self, examples: List[Dict], max_workers: int = 4) -> List[Dict]:
        """
        Run evaluation on a batch of examples with parallel processing
        """
        results = []
        
        logger.info(f"Starting batch evaluation of {len(examples)} examples with {max_workers} workers")
        
        def process_example(example):
            logger.info(f"Processing {example['id']}")
            
            config_name = example.get("task_type", "general_reasoning")
            response = self.generate_response(example["prompt"], config_name)
            
            result = {
                **example,
                **response,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_example = {executor.submit(process_example, ex): ex for ex in examples}
            
            completed_count = 0
            for future in as_completed(future_to_example):
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    logger.info(f"Completed {result['id']} in {result['processing_time']:.2f}s ({completed_count}/{len(examples)})")
                    
                    # Log GPU memory every 5 completions (reduced frequency to avoid spam)
                    if completed_count % 5 == 0:
                        logger.info(f"GPU memory after {completed_count} completions:")
                        log_gpu_memory(f"(After {completed_count} completions)")
                        
                except Exception as e:
                    logger.error(f"Example failed: {e}")
        
        # Final GPU memory check
        logger.info("Final GPU memory status:")
        log_gpu_memory("(Final)")
        
        return results
    
    def save_results(self, results: List[Dict], dataset_name: str) -> Tuple[str, str]:
        """
        Save results to both JSON and text files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON file for structured data
        json_path = self.output_dir / f"{dataset_name}_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Text file for human reading
        txt_path = self.output_dir / f"{dataset_name}_readable_{timestamp}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"DeepSeek R1 Benchmark Results - {dataset_name}\n")
            f.write("=" * 60 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"Example {i}: {result['id']}\n")
                f.write("-" * 40 + "\n")
                f.write(f"PROMPT:\n{result['prompt']}\n\n")
                
                if result['thinking']:
                    f.write(f"THINKING:\n{result['thinking']}\n\n")
                
                f.write(f"MODEL ANSWER:\n{result['answer']}\n\n")
                f.write(f"EXPECTED:\n{result['expected_answer']}\n\n")
                f.write(f"Processing Time: {result['processing_time']:.2f}s\n")
                f.write(f"Peak Memory: {result.get('peak_memory_mb', 0):.1f}MB\n")
                f.write(f"Success: {result['success']}\n")
                f.write("\n" + "="*60 + "\n\n")
        
        logger.info(f"Results saved to {json_path} and {txt_path}")
        return str(json_path), str(txt_path)
    
    def generate_summary_report(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Generate summary statistics from results including token consistency
        """
        total_examples = len(results)
        successful = sum(1 for r in results if r['success'])
        failed = total_examples - successful
        
        avg_time = sum(r['processing_time'] for r in results) / total_examples if results else 0
        total_time = sum(r['processing_time'] for r in results)
        
        # Calculate token statistics for consistency analysis
        answer_lengths = []
        thinking_lengths = []
        peak_memories = []
        
        for result in results:
            if result['success']:
                answer_tokens = len(self.tokenizer.encode(result.get('answer', '')))
                thinking_tokens = len(self.tokenizer.encode(result.get('thinking', '')))
                answer_lengths.append(answer_tokens)
                thinking_lengths.append(thinking_tokens)
                
                # Collect peak memory usage
                peak_mem = result.get('peak_memory_mb', 0)
                if peak_mem > 0:
                    peak_memories.append(peak_mem)
        
        # Group by task type
        by_task = {}
        for result in results:
            task = result.get('task_type', 'unknown')
            if task not in by_task:
                by_task[task] = []
            by_task[task].append(result)
        
        task_stats = {}
        for task, task_results in by_task.items():
            successful_tasks = [r for r in task_results if r['success']]
            task_answer_lengths = [len(self.tokenizer.encode(r.get('answer', ''))) for r in successful_tasks]
            task_peak_memories = [r.get('peak_memory_mb', 0) for r in successful_tasks if r.get('peak_memory_mb', 0) > 0]
            
            task_stats[task] = {
                "count": len(task_results),
                "successful": len(successful_tasks),
                "avg_time": sum(r['processing_time'] for r in task_results) / len(task_results),
                "avg_answer_tokens": sum(task_answer_lengths) / len(task_answer_lengths) if task_answer_lengths else 0,
                "min_answer_tokens": min(task_answer_lengths) if task_answer_lengths else 0,
                "max_answer_tokens": max(task_answer_lengths) if task_answer_lengths else 0,
                "avg_peak_memory_mb": sum(task_peak_memories) / len(task_peak_memories) if task_peak_memories else 0,
                "max_peak_memory_mb": max(task_peak_memories) if task_peak_memories else 0
            }
        
        summary = {
            "total_examples": total_examples,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_examples if total_examples > 0 else 0,
            "average_processing_time": avg_time,
            "total_processing_time": total_time,
            "token_stats": {
                "avg_answer_tokens": sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
                "min_answer_tokens": min(answer_lengths) if answer_lengths else 0,
                "max_answer_tokens": max(answer_lengths) if answer_lengths else 0,
                "avg_thinking_tokens": sum(thinking_lengths) / len(thinking_lengths) if thinking_lengths else 0
            },
            "memory_stats": {
                "avg_peak_memory_mb": sum(peak_memories) / len(peak_memories) if peak_memories else 0,
                "max_peak_memory_mb": max(peak_memories) if peak_memories else 0,
                "min_peak_memory_mb": min(peak_memories) if peak_memories else 0,
                "examples_with_memory_data": len(peak_memories)
            },
            "by_task_type": task_stats,
            "global_seed": self.global_seed,
            "timestamp": datetime.now().isoformat()
        }
        
        return summary


def main():
    """
    Main benchmarking function - customize datasets and parameters here
    """
    # Set global seed for fully reproducible results
    benchmark = DeepSeekBenchmark(global_seed=42)
    
    # Define datasets to benchmark
    benchmark_configs = [
        {
            "name": "gsm8k",
            "dataset": "gsm8k",
            "subset": "main",
            "split": "test",
            "limit": 2  # Start small for testing
        },
        {
            "name": "mmlu_elementary_math", 
            "dataset": "cais/mmlu",
            "subset": "elementary_mathematics",
            "split": "test",
            "limit": 2
        },
        {
            "name": "hellaswag",
            "dataset": "hellaswag", 
            "subset": None,
            "split": "validation",
            "limit": 2
        }
    ]
    
    all_results = {}
    
    for config in benchmark_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running benchmark: {config['name']}")
        logger.info(f"{'='*60}")
        
        # Load dataset
        examples = benchmark.load_benchmark_dataset(
            dataset_name=config["dataset"],
            subset=config.get("subset"),
            split=config["split"],
            limit=config.get("limit")
        )
        
        if not examples:
            logger.warning(f"No examples loaded for {config['name']}, skipping...")
            continue
        
        # Run evaluation
        results = benchmark.run_batch_evaluation(examples, max_workers=3)
        
        # Save results
        json_path, txt_path = benchmark.save_results(results, config["name"])
        
        # Generate summary
        summary = benchmark.generate_summary_report(results)
        
        logger.info(f"\nSUMMARY for {config['name']}:")
        logger.info(f"Total examples: {summary['total_examples']}")
        logger.info(f"Success rate: {summary['success_rate']:.2%}")
        logger.info(f"Average time per example: {summary['average_processing_time']:.2f}s")
        logger.info(f"Total processing time: {summary['total_processing_time']:.2f}s")
        logger.info(f"Average answer tokens: {summary['token_stats']['avg_answer_tokens']:.1f}")
        logger.info(f"Token range: {summary['token_stats']['min_answer_tokens']}-{summary['token_stats']['max_answer_tokens']}")
        logger.info(f"Peak memory usage: {summary['memory_stats']['avg_peak_memory_mb']:.1f}MB avg, {summary['memory_stats']['max_peak_memory_mb']:.1f}MB max")
        
        all_results[config["name"]] = {
            "results": results,
            "summary": summary,
            "files": {"json": json_path, "txt": txt_path}
        }
    
    # Save overall summary
    overall_summary_path = benchmark.output_dir / f"overall_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(overall_summary_path, 'w') as f:
        json.dump({k: v["summary"] for k, v in all_results.items()}, f, indent=2)
    
    logger.info(f"\nBenchmarking complete! Check {benchmark.output_dir} for all results.")
    logger.info(f"Overall summary saved to: {overall_summary_path}")
    logger.info(f"All results generated with global seed: {benchmark.global_seed}")


if __name__ == "__main__":
    main()