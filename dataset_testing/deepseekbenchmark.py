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
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor, as_completed

# Simple logging setup
import logging
from datetime import datetime
import os

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


def log_gpu_memory():
    """Log essential GPU information using nvidia-smi"""
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
            logger.info("=== GPU Status ===")
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
        
        logger.info("==================")
        
    except FileNotFoundError:
        logger.warning("nvidia-smi not found - GPU monitoring unavailable")
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e}")


class DeepSeekBenchmark:
    """
    Comprehensive benchmarking system for DeepSeek R1 model
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Log initial GPU memory
        log_gpu_memory()
        
        # Load model once
        self.tokenizer, self.model = ModelLoader.load_model()
        
        # Log GPU memory after model loading
        logger.info("GPU memory after model loading:")
        log_gpu_memory()
        
        # Benchmark configurations - reduced max_new_tokens to prevent rambling
        self.configs = {
            "math_reasoning": {
                "temperature": 0.3,  # More deterministic for math
                "top_k": 40,
                "top_p": 0.9,
                "max_new_tokens": 300  # Reduced from 512
            },
            "general_reasoning": {
                "temperature": 0.5,
                "top_k": 50,
                "top_p": 0.9,
                "max_new_tokens": 200  # Reduced from 256
            },
            "knowledge_qa": {
                "temperature": 0.4,
                "top_k": 45,
                "top_p": 0.85,
                "max_new_tokens": 150  # Reduced from 200
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
        """Format examples based on dataset type"""
        
        if dataset_name == "gsm8k":
            return {
                "id": f"gsm8k_{index}",
                "prompt": f"Solve this math problem step by step:\n\n{example['question']}\n\nPlease show your reasoning and provide the final answer.",
                "expected_answer": example.get("answer", ""),
                "dataset": "gsm8k",
                "task_type": "math_reasoning"
            }
        
        elif dataset_name == "cais/mmlu":
            choices = example.get("choices", [])
            choice_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            return {
                "id": f"mmlu_{index}",
                "prompt": f"Question: {example['question']}\n\nChoices:\n{choice_text}\n\nAnswer with the letter (A, B, C, or D) and explain your reasoning:",
                "expected_answer": chr(65 + example.get("answer", 0)),
                "dataset": "mmlu",
                "task_type": "knowledge_qa",
                "subject": example.get("subject", "unknown")
            }
        
        elif dataset_name == "hellaswag":
            return {
                "id": f"hellaswag_{index}",
                "prompt": f"Complete this scenario with the most logical continuation:\n\n{example['ctx']}\n\nOptions:\n" + 
                         "\n".join([f"{i+1}. {ending}" for i, ending in enumerate(example['endings'])]) +
                         "\n\nChoose the number (1, 2, 3, or 4) that best completes the scenario:",
                "expected_answer": str(example.get("label", 0) + 1),
                "dataset": "hellaswag", 
                "task_type": "general_reasoning"
            }
        
        elif dataset_name == "ai2_arc":
            choices = example.get("choices", {}).get("text", [])
            choice_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            return {
                "id": f"arc_{index}",
                "prompt": f"Science Question: {example['question']}\n\nChoices:\n{choice_text}\n\nAnswer with the letter and explain your reasoning:",
                "expected_answer": example.get("answerKey", ""),
                "dataset": "ai2_arc",
                "task_type": "knowledge_qa"
            }
        
        elif dataset_name == "truthful_qa":
            return {
                "id": f"truthfulqa_{index}",
                "prompt": f"Question: {example['question']}\n\nPlease provide a truthful and accurate answer:",
                "expected_answer": example.get("best_answer", ""),
                "dataset": "truthful_qa",
                "task_type": "knowledge_qa"
            }
        
        return None
    
    def generate_response(self, prompt: str, config_name: str = "general_reasoning") -> Dict[str, Any]:
        """
        Generate response using specified configuration
        """
        config = self.configs.get(config_name, self.configs["general_reasoning"])
        
        # Add thinking prompt for DeepSeek R1
        thinking_prompt = f"""
        {prompt}
        
        <think>
        Let me think through this step by step:
        """
        
        start_time = time.time()
        
        try:
            inputs = self.tokenizer(thinking_prompt, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config["max_new_tokens"],
                    temperature=config["temperature"],
                    top_k=config["top_k"],
                    top_p=config["top_p"],
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract thinking and final answer
            thinking_part = ""
            final_answer = full_response
            
            if "<think>" in full_response and "</think>" in full_response:
                thinking_part = full_response.split("<think>")[1].split("</think>")[0].strip()
                final_answer = full_response.split("</think>")[1].strip()
            
            # Simple rambling prevention - truncate at reasonable stopping points
            final_answer = self._truncate_rambling(final_answer)
            
            processing_time = time.time() - start_time
            
            return {
                "thinking": thinking_part,
                "answer": final_answer,
                "full_response": full_response,
                "processing_time": processing_time,
                "config_used": config_name,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "thinking": "",
                "answer": f"ERROR: {str(e)}",
                "full_response": "",
                "processing_time": time.time() - start_time,
                "config_used": config_name,
                "success": False
            }
    
    def _truncate_rambling(self, text: str) -> str:
        """
        Simple rambling prevention - stop at natural ending points
        """
        if not text:
            return text
            
        # Look for natural stopping points (but don't overdo it)
        stop_phrases = [
            "In conclusion",
            "To summarize", 
            "The answer is",
            "Therefore, the answer",
            "\n\nLet me solve this again",
            "\n\nNow let me",
            "\n\nTo solve this problem"
        ]
        
        # Find the earliest reasonable stopping point
        for phrase in stop_phrases:
            if phrase in text:
                # Keep the sentence with the stop phrase, then truncate
                parts = text.split(phrase)
                if len(parts) > 1:
                    # Keep the part with the phrase + a reasonable amount after
                    kept_part = parts[0] + phrase + parts[1].split('.')[0] + '.'
                    if len(kept_part) < len(text) * 0.8:  # Only truncate if we're removing substantial text
                        return kept_part
        
        return text
    
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
                    
                    # Log GPU memory every 10 completions
                    if completed_count % 10 == 0:
                        logger.info(f"GPU memory after {completed_count} completions:")
                        log_gpu_memory()
                        
                except Exception as e:
                    logger.error(f"Example failed: {e}")
        
        # Final GPU memory check
        logger.info("Final GPU memory status:")
        log_gpu_memory()
        
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
                f.write(f"Success: {result['success']}\n")
                f.write("\n" + "="*60 + "\n\n")
        
        logger.info(f"Results saved to {json_path} and {txt_path}")
        return str(json_path), str(txt_path)
    
    def generate_summary_report(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Generate summary statistics from results
        """
        total_examples = len(results)
        successful = sum(1 for r in results if r['success'])
        failed = total_examples - successful
        
        avg_time = sum(r['processing_time'] for r in results) / total_examples if results else 0
        total_time = sum(r['processing_time'] for r in results)
        
        # Group by task type
        by_task = {}
        for result in results:
            task = result.get('task_type', 'unknown')
            if task not in by_task:
                by_task[task] = []
            by_task[task].append(result)
        
        task_stats = {}
        for task, task_results in by_task.items():
            task_stats[task] = {
                "count": len(task_results),
                "successful": sum(1 for r in task_results if r['success']),
                "avg_time": sum(r['processing_time'] for r in task_results) / len(task_results)
            }
        
        summary = {
            "total_examples": total_examples,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_examples if total_examples > 0 else 0,
            "average_processing_time": avg_time,
            "total_processing_time": total_time,
            "by_task_type": task_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        return summary


def main():
    """
    Main benchmarking function - customize datasets and parameters here
    """
    benchmark = DeepSeekBenchmark()
    
    # Define datasets to benchmark
    benchmark_configs = [
        {
            "name": "gsm8k",
            "dataset": "gsm8k",
            "subset": "main",
            "split": "test",
            "limit": 4  # Start small for testing
        },
        {
            "name": "mmlu_elementary_math", 
            "dataset": "cais/mmlu",
            "subset": "elementary_mathematics",
            "split": "test",
            "limit": 4
        },
        {
            "name": "hellaswag",
            "dataset": "hellaswag", 
            "subset": None,
            "split": "validation",
            "limit": 4
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
        results = benchmark.run_batch_evaluation(examples, max_workers=4)
        
        # Save results
        json_path, txt_path = benchmark.save_results(results, config["name"])
        
        # Generate summary
        summary = benchmark.generate_summary_report(results)
        
        logger.info(f"\nSUMMARY for {config['name']}:")
        logger.info(f"Total examples: {summary['total_examples']}")
        logger.info(f"Success rate: {summary['success_rate']:.2%}")
        logger.info(f"Average time per example: {summary['average_processing_time']:.2f}s")
        logger.info(f"Total processing time: {summary['total_processing_time']:.2f}s")
        
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


if __name__ == "__main__":
    main()