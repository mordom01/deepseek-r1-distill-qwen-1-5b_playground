"""
Enhanced DeepSeek R1 Model Benchmarking System with True Batch Processing
Place this file as: dataset_testing/enhanced_deepseek_benchmark.py
"""

import sys
import json
import time
import logging
import subprocess
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

# Add parent directory to path to import your existing modules
import os
home_dir_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, home_dir_path)

def setup_simple_logging():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/enhanced_benchmark_{timestamp}.log"
    
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

# Import your existing modules
from model_loader import ModelLoader

class ComputeMemoryBoundAnalyzer:
    """Analyzer to determine if operations are compute-bound or memory-bound"""
    
    def __init__(self):
        self.gpu_specs = self._detect_gpu_specs()
    
    def _detect_gpu_specs(self) -> Dict[str, float]:
        """Detect GPU specifications for theoretical performance calculation"""
        if not torch.cuda.is_available():
            return {}
        
        gpu_name = torch.cuda.get_device_name()
        logger.info(f"Detected GPU: {gpu_name}")
        
        # Common GPU specifications (GB/s for memory bandwidth, TFLOPS for compute)
        gpu_specs_db = {
            'RTX 4090': {'memory_bandwidth_gb_s': 1008, 'fp32_tflops': 83, 'fp16_tflops': 166},
            'RTX 3090': {'memory_bandwidth_gb_s': 936, 'fp32_tflops': 36, 'fp16_tflops': 71},
            'A100': {'memory_bandwidth_gb_s': 1935, 'fp32_tflops': 19.5, 'fp16_tflops': 78},
            'H100': {'memory_bandwidth_gb_s': 3350, 'fp32_tflops': 67, 'fp16_tflops': 134},
            'V100': {'memory_bandwidth_gb_s': 900, 'fp32_tflops': 15.7, 'fp16_tflops': 31.4},
        }
        
        # Try to match GPU name
        for gpu_pattern, specs in gpu_specs_db.items():
            if gpu_pattern in gpu_name:
                logger.info(f"Using GPU specs for {gpu_pattern}")
                return specs
        
        # Default conservative estimates
        logger.warning(f"GPU {gpu_name} not in database, using conservative estimates")
        return {'memory_bandwidth_gb_s': 500, 'fp32_tflops': 20, 'fp16_tflops': 40}
    
    def analyze_operation_bottleneck(self, 
                                   operation_name: str,
                                   execution_time_ms: float, 
                                   memory_delta_mb: float,
                                   tensor_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze whether an operation is compute-bound or memory-bound"""
        
        if not self.gpu_specs or execution_time_ms <= 0:
            return {'analysis': 'insufficient_data', 'bottleneck_type': 'unknown'}
        
        # Calculate theoretical memory-bound time
        memory_bandwidth_gb_s = self.gpu_specs['memory_bandwidth_gb_s']
        memory_gb = max(memory_delta_mb / 1024, 0.001)  # Prevent division by zero
        theoretical_memory_time_ms = memory_gb / memory_bandwidth_gb_s * 1000
        
        # Estimate FLOPs for common operations
        estimated_flops = self._estimate_flops(operation_name, tensor_info)
        
        # Calculate theoretical compute-bound time
        dtype = tensor_info.get('dtype', 'fp16') if tensor_info else 'fp16'
        tflops_key = 'fp16_tflops' if 'fp16' in str(dtype) else 'fp32_tflops'
        peak_tflops = self.gpu_specs[tflops_key]
        theoretical_compute_time_ms = (estimated_flops / 1e12) / peak_tflops * 1000
        
        # Prevent division by zero
        theoretical_memory_time_ms = max(theoretical_memory_time_ms, 0.001)
        theoretical_compute_time_ms = max(theoretical_compute_time_ms, 0.001)
        
        # Determine bottleneck type
        memory_bound_ratio = execution_time_ms / theoretical_memory_time_ms
        compute_bound_ratio = execution_time_ms / theoretical_compute_time_ms
        
        # Classification logic
        if memory_bound_ratio > 0.5 and memory_bound_ratio > compute_bound_ratio:
            bottleneck_type = "memory_bound"
            efficiency = min(theoretical_memory_time_ms / execution_time_ms, 1.0)
        elif compute_bound_ratio > 0.5 and compute_bound_ratio > memory_bound_ratio:
            bottleneck_type = "compute_bound" 
            efficiency = min(theoretical_compute_time_ms / execution_time_ms, 1.0)
        elif execution_time_ms < max(theoretical_memory_time_ms, theoretical_compute_time_ms) * 0.5:
            bottleneck_type = "highly_efficient"
            efficiency = 1.0
        else:
            bottleneck_type = "balanced_or_overhead"
            efficiency = 0.5
        
        return {
            'operation': operation_name,
            'bottleneck_type': bottleneck_type,
            'efficiency_score': efficiency,
            'analysis': {
                'actual_time_ms': execution_time_ms,
                'theoretical_memory_time_ms': theoretical_memory_time_ms,
                'theoretical_compute_time_ms': theoretical_compute_time_ms,
                'memory_bound_ratio': memory_bound_ratio,
                'compute_bound_ratio': compute_bound_ratio,
                'memory_transferred_mb': memory_delta_mb,
                'estimated_flops': estimated_flops
            },
            'optimization_hint': self._get_optimization_hint(bottleneck_type, operation_name)
        }
    
    def _estimate_flops(self, operation_name: str, tensor_info: Dict[str, Any] = None) -> float:
        """Estimate FLOPs for common transformer operations"""
        
        if not tensor_info:
            return 1e9  # Default estimate
        
        # Extract common dimensions
        batch_size = tensor_info.get('batch_size', 1)
        seq_len = tensor_info.get('seq_len', 1024)
        hidden_size = tensor_info.get('hidden_size', 4096)
        intermediate_size = tensor_info.get('intermediate_size', 11008)
        num_heads = tensor_info.get('num_heads', 32)
        head_dim = hidden_size // num_heads if num_heads > 0 else 128
        
        op_lower = operation_name.lower()
        
        # Attention operations
        if 'q_projection' in op_lower or 'k_projection' in op_lower or 'v_projection' in op_lower:
            return batch_size * seq_len * hidden_size * hidden_size * 2
        elif 'output_projection' in op_lower:
            return batch_size * seq_len * hidden_size * hidden_size * 2
        elif 'attention_matmul_qk' in op_lower:
            return batch_size * num_heads * seq_len * seq_len * head_dim * 2
        elif 'attention_matmul_v' in op_lower:
            return batch_size * num_heads * seq_len * seq_len * head_dim * 2
        elif 'attention_softmax' in op_lower:
            return batch_size * num_heads * seq_len * seq_len * 5
        
        # MLP operations
        elif 'gate_projection' in op_lower or 'up_projection' in op_lower:
            return batch_size * seq_len * hidden_size * intermediate_size * 2
        elif 'down_projection' in op_lower:
            return batch_size * seq_len * intermediate_size * hidden_size * 2
        elif 'activation_function' in op_lower:
            return batch_size * seq_len * intermediate_size * 4
        elif 'element_wise_multiply' in op_lower:
            return batch_size * seq_len * intermediate_size
        
        # Normalization
        elif 'rms_norm' in op_lower or 'layernorm' in op_lower:
            return batch_size * seq_len * hidden_size * 8
        
        # RoPE
        elif 'rotary_pos_emb' in op_lower:
            return batch_size * seq_len * hidden_size * 6
        
        else:
            # Default estimate based on typical operations
            total_elements = batch_size * seq_len * hidden_size
            return total_elements * 4  # Assume 4 operations per element
    
    def _get_optimization_hint(self, bottleneck_type: str, operation_name: str) -> str:
        """Provide optimization hints based on bottleneck analysis"""
        
        op_lower = operation_name.lower()
        
        if bottleneck_type == "memory_bound":
            if 'attention' in op_lower:
                return "Consider Flash Attention to reduce memory transfers"
            elif 'norm' in op_lower:
                return "Consider fused normalization kernels"
            elif 'projection' in op_lower:
                return "Consider kernel fusion with adjacent operations"
            else:
                return "Optimize memory access patterns or use kernel fusion"
        
        elif bottleneck_type == "compute_bound":
            if any(x in op_lower for x in ['projection', 'matmul']):
                return "Consider mixed precision (FP16) or optimized GEMM libraries"
            elif 'softmax' in op_lower:
                return "Consider fused attention kernels"
            else:
                return "Optimize algorithmic complexity or use tensor cores"
        
        elif bottleneck_type == "balanced_or_overhead":
            return "Investigate kernel launch overhead or synchronization issues"
        
        else:  # highly_efficient
            return "Operation is already well-optimized"


def log_gpu_memory(stage=""):
    """Log essential GPU information using nvidia-smi and PyTorch memory stats"""
    try:
        gpu_query = [
            'nvidia-smi', 
            '--query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw,utilization.gpu',
            '--format=csv,noheader,nounits'
        ]
        
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
        
        if torch.cuda.is_available():
            logger.info("=== PyTorch CUDA Memory ===")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**2
                reserved = torch.cuda.memory_reserved(i) / 1024**2
                max_allocated = torch.cuda.max_memory_allocated(i) / 1024**2
                logger.info(f"  GPU {i}: Allocated: {allocated:.1f}MB | Reserved: {reserved:.1f}MB | Peak: {max_allocated:.1f}MB")
        
        logger.info("==================")
        
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e}")


class EnhancedDeepSeekBenchmark:
    """
    Enhanced benchmarking system with true batch processing and compute/memory bound analysis
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
        
        # Load model with enhanced profiling
        try:
            self.tokenizer, self.model = ModelLoader.load_model_with_profiling({
                'sample_rate': 1.0,  # Full profiling as requested
                'adaptive_sampling': False,  # Disable adaptive sampling
                'max_operations_per_layer': float('inf'),  # No operation limits
                'use_cuda_events': True
            })
            self.profiling_enabled = True
            logger.info("✅ Enhanced profiling enabled")
        except Exception as e:
            logger.warning(f"Enhanced profiling failed: {e}")
            logger.info("Falling back to basic model loading...")
            self.tokenizer, self.model = ModelLoader.load_model(enable_profiling=False)
            self.profiling_enabled = False
            logger.info("⚠️ Running without enhanced profiling")
        
        # Initialize bottleneck analyzer
        self.bottleneck_analyzer = ComputeMemoryBoundAnalyzer()
        
        # Batch processing configurations
        self.batch_sizes = [1, 2, 4, 8]  # Start smaller for testing
        self.max_seq_length = 2048
        
        # Log GPU memory after model loading
        log_gpu_memory("(Post-Model Load)")
        
        # Benchmark configurations
        self.configs = {
            "math_reasoning": {
                "temperature": 0.1,
                "top_k": 20,
                "top_p": 0.8,
                "max_new_tokens": 600,
                "min_new_tokens": 100,
                "do_sample": True,
            },
            "general_reasoning": {
                "temperature": 0.1,
                "top_k": 20,
                "top_p": 0.8,
                "max_new_tokens": 400,
                "min_new_tokens": 80,
                "do_sample": True,
            },
            "knowledge_qa": {
                "temperature": 0.05,
                "top_k": 15,
                "top_p": 0.7,
                "max_new_tokens": 300,
                "min_new_tokens": 60,
                "do_sample": True,
            }
        }

    def load_benchmark_dataset(self, dataset_name: str, subset: Optional[str] = None, 
                             split: str = "test", limit: Optional[int] = None) -> List[Dict]:
        """Load benchmark dataset from Hugging Face"""
        logger.info(f"Loading dataset: {dataset_name}")
        
        try:
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
            ctx = str(example.get('ctx', ''))
            endings = example.get('endings', [])
            label = example.get('label', 0)
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
        
        return None

    def prepare_batch(self, prompts: List[str], batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Prepare batched inputs with proper padding and attention masks"""
        batches = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Tokenize all prompts in the batch
            tokenized = self.tokenizer(
                batch_prompts,
                padding='longest',
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            
            # Move to GPU
            device = next(self.model.parameters()).device
            batch_data = {k: v.to(device) for k, v in tokenized.items()}
            
            # Add metadata for analysis
            batch_data['metadata'] = {
                'batch_size': len(batch_prompts),
                'actual_batch_size': len(batch_prompts),
                'sequence_lengths': [len(self.tokenizer.encode(p)) for p in batch_prompts],
                'max_seq_length': batch_data['input_ids'].shape[1],
                'total_tokens': batch_data['input_ids'].numel()
            }
            
            batches.append(batch_data)
        
        return batches

    def generate_batch_response(self, 
                              batch_data: Dict[str, torch.Tensor],
                              config_name: str = "general_reasoning") -> Dict[str, Any]:
        """Generate responses for a batch with enhanced profiling"""
        config = self.configs.get(config_name, self.configs["general_reasoning"])
        metadata = batch_data['metadata']
        actual_batch_size = metadata['actual_batch_size']
        
        # Reset profiling stats for this batch
        if self.profiling_enabled and hasattr(self.model.model, 'profiler'):
            self.model.model.profiler.reset_stats()
        
        # Clear GPU memory and reset peak tracking
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        start_time = time.time()
        
        try:
            with torch.no_grad():
                # Single batched forward pass
                outputs = self.model.generate(
                    input_ids=batch_data['input_ids'],
                    attention_mask=batch_data['attention_mask'],
                    max_new_tokens=config["max_new_tokens"],
                    min_new_tokens=config.get("min_new_tokens", 20),
                    temperature=config["temperature"],
                    top_k=config["top_k"],
                    top_p=config["top_p"],
                    do_sample=config["do_sample"],
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            
            # Decode responses
            responses = []
            for i in range(actual_batch_size):
                input_length = batch_data['input_ids'][i].numel()
                generated_tokens = outputs[i][input_length:]
                response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                responses.append(response_text)
            
            # Collect profiling results with enhanced analysis
            profiling_results = None
            if self.profiling_enabled and hasattr(self.model.model, 'profiler'):
                try:
                    raw_profiling = self.model.model.profiler.get_aggregated_stats()
                    
                    # Add compute/memory bound analysis
                    enhanced_profiling = {'operations': {}}
                    for op_name, stats in raw_profiling.get('operations', {}).items():
                        bottleneck_analysis = self.bottleneck_analyzer.analyze_operation_bottleneck(
                            operation_name=op_name,
                            execution_time_ms=stats.get('avg_time_ms', 0),
                            memory_delta_mb=stats.get('avg_memory_delta_mb', 0),
                            tensor_info={
                                'batch_size': actual_batch_size,
                                'seq_len': metadata['max_seq_length'],
                                'hidden_size': 4096,  # Default, update if you know your model config
                                'intermediate_size': 11008,
                                'num_heads': 32,
                            }
                        )
                        
                        enhanced_profiling['operations'][op_name] = {
                            **stats,
                            'bottleneck_analysis': bottleneck_analysis
                        }
                    
                    enhanced_profiling['summary'] = raw_profiling.get('summary', {})
                    profiling_results = enhanced_profiling
                except Exception as e:
                    logger.warning(f"Failed to analyze profiling results: {e}")
                    profiling_results = {"error": str(e)}
            
            return {
                'responses': responses,
                'batch_metadata': metadata,
                'performance': {
                    'batch_size': actual_batch_size,
                    'total_time_ms': (end_time - start_time) * 1000,
                    'time_per_sample_ms': (end_time - start_time) * 1000 / actual_batch_size,
                    'tokens_per_second': metadata['total_tokens'] / (end_time - start_time),
                    'memory_usage': {
                        'start_memory_mb': start_memory / 1024**2,
                        'end_memory_mb': end_memory / 1024**2,
                        'peak_memory_mb': peak_memory / 1024**2,
                        'memory_delta_mb': (end_memory - start_memory) / 1024**2,
                        'memory_per_sample_mb': (peak_memory - start_memory) / 1024**2 / actual_batch_size
                    }
                },
                'profiling_results': profiling_results,
                'config_used': config_name,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            return {
                'responses': ['ERROR'] * actual_batch_size,
                'batch_metadata': metadata,
                'performance': {'error': str(e)},
                'profiling_results': None
            }

    def run_batch_size_analysis(self, 
                               examples: List[Dict], 
                               max_examples_per_batch_size: int = 8) -> Dict[str, Any]:
        """Run analysis across different batch sizes - THIS IS WHAT YOU REQUESTED"""
        results = {}
        
        # Limit examples for faster iteration across batch sizes
        test_examples = examples[:max_examples_per_batch_size * max(self.batch_sizes)]
        prompts = [ex['prompt'] for ex in test_examples]
        
        logger.info(f"Running batch size analysis with {len(prompts)} prompts")
        logger.info(f"Testing batch sizes: {self.batch_sizes}")
        
        for batch_size in self.batch_sizes:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing batch size: {batch_size}")
            logger.info(f"{'='*50}")
            
            # Prepare batches for this batch size
            batches = self.prepare_batch(prompts, batch_size)
            batch_results = []
            
            for batch_idx, batch_data in enumerate(batches):
                logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} "
                           f"(actual size: {batch_data['metadata']['actual_batch_size']})")
                
                # Generate responses for this batch
                batch_result = self.generate_batch_response(batch_data, "general_reasoning")
                batch_results.append(batch_result)
                
                # Log immediate results
                perf = batch_result['performance']
                if 'error' not in perf:
                    logger.info(f"  Time per sample: {perf['time_per_sample_ms']:.2f}ms")
                    logger.info(f"  Memory per sample: {perf['memory_usage']['memory_per_sample_mb']:.2f}MB")
                    logger.info(f"  Tokens/sec: {perf['tokens_per_second']:.1f}")
                
                # Brief pause to allow GPU memory to stabilize
                time.sleep(1)
            
            # Aggregate results for this batch size
            results[batch_size] = self._analyze_batch_size_results(batch_results, batch_size)
        
        return results

    def _analyze_batch_size_results(self, batch_results: List[Dict], batch_size: int) -> Dict[str, Any]:
        """Analyze results for a specific batch size"""
        valid_results = [r for r in batch_results if 'error' not in r['performance']]
        
        if not valid_results:
            return {'error': 'No valid results', 'batch_size': batch_size}
        
        # Extract performance metrics
        times_per_sample = [r['performance']['time_per_sample_ms'] for r in valid_results]
        memory_per_sample = [r['performance']['memory_usage']['memory_per_sample_mb'] for r in valid_results]
        tokens_per_second = [r['performance']['tokens_per_second'] for r in valid_results]
        peak_memories = [r['performance']['memory_usage']['peak_memory_mb'] for r in valid_results]
        
        # Aggregate profiling results with bottleneck analysis
        aggregated_profiling = self._aggregate_profiling_across_batches(valid_results)
        
        return {
            'batch_size': batch_size,
            'num_batches': len(valid_results),
            'performance_summary': {
                'avg_time_per_sample_ms': np.mean(times_per_sample),
                'std_time_per_sample_ms': np.std(times_per_sample),
                'avg_memory_per_sample_mb': np.mean(memory_per_sample),
                'avg_tokens_per_second': np.mean(tokens_per_second),
                'avg_peak_memory_mb': np.mean(peak_memories),
                'max_peak_memory_mb': np.max(peak_memories),
                'memory_efficiency': np.mean(memory_per_sample) / np.max(peak_memories) if peak_memories else 0,
            },
            'profiling_summary': aggregated_profiling,
        }

    def _aggregate_profiling_across_batches(self, batch_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate profiling results with bottleneck analysis across multiple batches"""
        if not batch_results or not batch_results[0].get('profiling_results'):
            return {}
        
        # Combine operation stats across batches
        combined_operations = {}
        
        for result in batch_results:
            profiling = result.get('profiling_results', {})
            if 'error' in profiling:
                continue
                
            operations = profiling.get('operations', {})
            
            for op_name, stats in operations.items():
                if op_name not in combined_operations:
                    combined_operations[op_name] = []
                combined_operations[op_name].append(stats)
        
        # Calculate averages across batches with bottleneck analysis
        aggregated = {}
        for op_name, stats_list in combined_operations.items():
            if stats_list:
                avg_stats = {
                    'avg_time_ms': np.mean([s.get('avg_time_ms', 0) for s in stats_list]),
                    'total_calls': np.sum([s.get('total_calls', 0) for s in stats_list]),
                    'avg_memory_delta_mb': np.mean([s.get('avg_memory_delta_mb', 0) for s in stats_list]),
                }
                
                # Include bottleneck analysis from first result (should be consistent)
                if 'bottleneck_analysis' in stats_list[0]:
                    avg_stats['bottleneck_analysis'] = stats_list[0]['bottleneck_analysis']
                
                aggregated[op_name] = avg_stats
        
        return aggregated

    def save_batch_analysis_results(self, batch_results: Dict[str, Any], dataset_name: str) -> str:
        """Save batch analysis results with detailed profiling"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON results
        json_path = self.output_dir / f"{dataset_name}_batch_analysis_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Create human-readable summary
        summary_path = self.output_dir / f"{dataset_name}_batch_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Batch Size Analysis Summary - {dataset_name}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("PERFORMANCE BY BATCH SIZE:\n")
            f.write("-" * 30 + "\n")
            
            for batch_size in sorted(batch_results.keys()):
                result = batch_results[batch_size]
                if 'error' in result:
                    f.write(f"Batch Size {batch_size:3d}: ERROR - {result['error']}\n")
                    continue
                    
                perf = result['performance_summary']
                f.write(f"Batch Size {batch_size:3d}:\n")
                f.write(f"  Avg time per sample: {perf['avg_time_per_sample_ms']:.2f}ms\n")
                f.write(f"  Tokens per second: {perf['avg_tokens_per_second']:.1f}\n")
                f.write(f"  Memory per sample: {perf['avg_memory_per_sample_mb']:.2f}MB\n")
                f.write(f"  Peak memory: {perf['avg_peak_memory_mb']:.0f}MB\n")
                f.write(f"  Memory efficiency: {perf['memory_efficiency']:.3f}\n\n")
            
            # Add profiling insights
            f.write("TOP BOTTLENECKS ACROSS ALL BATCH SIZES:\n")
            f.write("-" * 40 + "\n")
            
            # Collect all operations across batch sizes
            all_operations = {}
            for batch_size, result in batch_results.items():
                if 'profiling_summary' in result:
                    for op_name, stats in result['profiling_summary'].items():
                        if op_name not in all_operations:
                            all_operations[op_name] = []
                        all_operations[op_name].append(stats)
            
            # Find top operations by average time
            op_avg_times = {}
            for op_name, stats_list in all_operations.items():
                if stats_list:
                    avg_time = np.mean([s.get('avg_time_ms', 0) for s in stats_list])
                    op_avg_times[op_name] = avg_time
            
            # Sort and display top operations
            sorted_ops = sorted(op_avg_times.items(), key=lambda x: x[1], reverse=True)
            for i, (op_name, avg_time) in enumerate(sorted_ops[:10], 1):
                f.write(f"{i:2d}. {op_name}: {avg_time:.3f}ms average\n")
                
                # Try to get bottleneck analysis from any batch
                for batch_size, result in batch_results.items():
                    profiling = result.get('profiling_summary', {})
                    if op_name in profiling and 'bottleneck_analysis' in profiling[op_name]:
                        bottleneck = profiling[op_name]['bottleneck_analysis']
                        f.write(f"     Type: {bottleneck['bottleneck_type']}\n")
                        f.write(f"     Hint: {bottleneck['optimization_hint']}\n")
                        break
                f.write("\n")
        
        logger.info(f"Batch analysis results saved to {json_path} and {summary_path}")
        return str(json_path)


def main():
    """Main function to run enhanced benchmark"""
    
    # Create enhanced benchmark
    benchmark = EnhancedDeepSeekBenchmark(global_seed=42)
    
    # Test with small dataset first
    logger.info("Loading test dataset...")
    examples = benchmark.load_benchmark_dataset("gsm8k", "main", "test", limit=16)
    
    if not examples:
        logger.error("No examples loaded! Check your dataset configuration.")
        return
    
    logger.info(f"Loaded {len(examples)} examples")
    
    # Run batch size analysis (this is what you specifically requested)
    logger.info("Starting batch size analysis...")
    batch_results = benchmark.run_batch_size_analysis(examples, max_examples_per_batch_size=4)
    
    # Save results
    results_path = benchmark.save_batch_analysis_results(batch_results, "gsm8k_test")
    
    # Print quick summary
    logger.info("\n" + "="*60)
    logger.info("BATCH SIZE ANALYSIS COMPLETE")
    logger.info("="*60)
    
    for batch_size in sorted(batch_results.keys()):
        result = batch_results[batch_size]
        if 'performance_summary' in result:
            perf = result['performance_summary']
            logger.info(f"Batch size {batch_size:2d}: "
                       f"{perf['avg_time_per_sample_ms']:6.2f}ms/sample, "
                       f"{perf['avg_tokens_per_second']:6.1f} tok/s, "
                       f"{perf['avg_peak_memory_mb']:6.0f}MB peak")
    
    logger.info(f"\nDetailed results saved to: {results_path}")
    logger.info("Check the _batch_summary.txt file for human-readable analysis!")


if __name__ == "__main__":
    main()