#!/usr/bin/env python3
"""
bnb_enhanced_profiler_counters.py

Enhanced profiler using PyTorch profiler for timing and instrumented counters for operation validation.
Provides fast end-to-end timing with granular operation counting.

Usage:
    python bnb_enhanced_profiler_counters.py --model qwen2.5-7b --counters
    python bnb_enhanced_profiler_counters.py --model deepseek-r1-14b --tag baseline
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import torch
import torch.profiler
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import json
import numpy as np

# Import our counter bindings
try:
    import bnb_counter_bindings as bnb_counters
    COUNTERS_AVAILABLE = True
    print("✅ Kernel counters available")
except ImportError:
    COUNTERS_AVAILABLE = False
    print("⚠️ Kernel counters not available - using PyTorch profiling only")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class BnBCounterProfiler:
    """Profiler using PyTorch timing + operation counters for validation"""
    
    def __init__(self, output_dir: str = "bnb_counter_profiling", use_counters: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_counters = use_counters and COUNTERS_AVAILABLE
        
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        self.model = None
        self.tokenizer = None
        
        if self.use_counters:
            try:
                self.counter_extractor = bnb_counters.get_counter_extractor()
                print("🔢 Operation counter tracking enabled")
            except Exception as e:
                print(f"⚠️ Failed to initialize counter extractor: {e}")
                self.use_counters = False
        
        if not self.use_counters:
            print("📊 Using PyTorch profiling only")
        
        self.bnb_operation_patterns = self._define_bnb_patterns()
        
    def _define_bnb_patterns(self) -> Dict[str, List[str]]:
        """Define regex patterns to identify BitsAndBytes operations"""
        return {
            'dequantization': [
                r'.*dequant.*',
                r'.*bnb.*dequant.*',
                r'.*4bit.*dequant.*',
                r'.*nf4.*dequant.*',
                r'.*unpack.*4bit.*',
                r'.*kDequantizeBlockwise.*',
                r'.*dequantize_blockwise.*',
                r'.*blockwise.*dequant.*'
            ],
            'quantization': [
                r'.*bnb.*quant.*',
                r'.*4bit.*quant.*',
                r'.*nf4.*quant.*',
                r'.*pack.*4bit.*'
            ],
            'bnb_linear': [
                r'.*bnb.*linear.*',
                r'.*Linear4bit.*',
                r'.*bnb.*matmul.*',
                r'.*4bit.*linear.*',
                r'.*bnb.*mm.*'
            ],
            'memory_ops': [
                r'.*fp4.*',
                r'.*nf4.*', 
                r'.*int4.*',
                r'.*4bit.*format.*'
            ]
        }
    
    def load_model_with_bnb(self, model_name: str) -> bool:
        """Load model with BitsAndBytes quantization"""
        
        print(f"🔄 Loading {model_name} with BitsAndBytes...")
        
        try:
            from transformers import BitsAndBytesConfig
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"   🧹 GPU memory before loading: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
            
            # Quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_storage=torch.uint8,
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with BitsAndBytes
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            
            # Verify quantization
            quantized_layers = 0
            total_layers = 0
            
            for name, module in self.model.named_modules():
                if hasattr(module, 'weight'):
                    total_layers += 1
                    if hasattr(module.weight, 'quant_type'):
                        quantized_layers += 1
            
            print(f"   📊 Quantization: {quantized_layers}/{total_layers} layers")
            
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.memory_allocated() / 1024**3
                print(f"   📊 GPU memory after loading: {gpu_memory_gb:.1f} GB")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def profile_with_counters(self, batch_size: int, max_tokens: int = 100) -> Dict[str, Any]:
        """Profile using PyTorch profiler and operation counters"""
        
        print(f"📊 Profiling - Batch size {batch_size}")
        
        # Load test prompts
        prompts = self.load_test_prompts(batch_size)
        
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            padding='longest',
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        input_tokens = inputs['input_ids'].numel()
        print(f"   Input tokens: {input_tokens}")
        
        # Warmup
        print("  🔥 Warmup...")
        with torch.no_grad():
            _ = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Reset counters with enhanced method for large models
        # Reset counters with enhanced method for large models
        if self.use_counters:
            try:
                # Diagnose first (optional, for debugging)
                if hasattr(self.counter_extractor, 'diagnose_counter_state'):
                    self.counter_extractor.diagnose_counter_state()
                
                # Force memory clear for large models
                if hasattr(self.counter_extractor, 'force_gpu_memory_clear'):
                    self.counter_extractor.force_gpu_memory_clear()
                
                # Enhanced reset
                reset_success = self.counter_extractor.reset_counters()
                
                if not reset_success:
                    print("   🚨 Counter reset failed - results will be cumulative")
                    # Continue anyway - PyTorch profiler still works
                    
            except Exception as e:
                print(f"   ❌ Counter reset failed: {e}")
        
        profiler_kwargs = {
            "activities": [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            "record_shapes": True,
            "profile_memory": True,
            "with_flops": True,
            "with_modules": True,
        }
        
        # Run profiling
        with torch.profiler.profile(**profiler_kwargs) as prof:
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.0
                )
            
            torch.cuda.synchronize()
            end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        output_tokens = outputs.numel() - input_tokens
        total_tokens = outputs.numel()
        
        tokens_per_sec = total_tokens / total_time
        new_tokens_per_sec = output_tokens / total_time
        
        # Calculate end-to-end latency metrics
        time_to_first_token = total_time / output_tokens if output_tokens > 0 else 0  # Rough approximation
        tokens_per_batch_item = total_tokens / batch_size
        time_per_batch_item = total_time / batch_size
        
        print(f"   ⚡ Generated {output_tokens} new tokens in {total_time:.2f}s")
        print(f"   📈 Throughput: {tokens_per_sec:.1f} total tokens/sec, {new_tokens_per_sec:.1f} new tokens/sec")
        print(f"   ⏱️ Latency: {total_time:.2f}s total, {time_per_batch_item:.2f}s per batch item")
        print(f"   🎯 Efficiency: {tokens_per_batch_item:.1f} tokens per batch item")
        
        # Memory stats
        memory_stats = {
            'peak_memory_mb': torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            'current_memory_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
        }
        
        # PyTorch profiler analysis
        pytorch_analysis = self._analyze_pytorch_profiling(prof)
        
        # Counter analysis
        counter_analysis = {}
        if self.use_counters:
            try:
                counter_analysis = self.counter_extractor.get_comprehensive_counter_report()
                total_ops = counter_analysis.get('total_operations', 0)
                print(f"   🎯 Operations counted: {total_ops:,}")
                
                # Validate counter data makes sense for this batch size
                if total_ops > 0:
                    nf4_count = counter_analysis.get('nf4_lookup_count', 0)
                    kernel_count = counter_analysis.get('kernel_call_count', 0)
                    
                    if kernel_count > 0:
                        ops_per_kernel = total_ops / kernel_count
                        nf4_per_kernel = nf4_count / kernel_count
                        
                        print(f"   📈 {ops_per_kernel:,.0f} ops/kernel, {nf4_per_kernel:,.0f} NF4/kernel")
                        
                        # Sanity checks
                        if ops_per_kernel < 1000:
                            print(f"   ⚠️ Very low ops/kernel - may indicate counter reset issues")
                        if nf4_per_kernel < 100:
                            print(f"   ⚠️ Very low NF4/kernel - may not be hitting blockwise kernels")
                        
                        # For large models, expect high operation density
                        # Note: We don't have model_name in this scope, so skip this check
                        if ops_per_kernel < 10000:
                            print(f"   🚨 Low ops/kernel detected: {ops_per_kernel:.0f}")
                            print(f"       This could indicate counter issues or different kernel patterns")
                    
            except Exception as e:
                print(f"   ⚠️ Failed to extract counters: {e}")
                counter_analysis = {'error': str(e)}
        
        return {
            'batch_size': batch_size,
            'total_time_s': total_time,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'tokens_per_sec': tokens_per_sec,
            'new_tokens_per_sec': new_tokens_per_sec,
            'memory_stats': memory_stats,
            'pytorch_profiling': pytorch_analysis,
            'operation_counters': counter_analysis,
            # New latency metrics
            'end_to_end_latency_s': total_time,
            'latency_per_batch_item_s': time_per_batch_item,
            'tokens_per_batch_item': tokens_per_batch_item,
            'time_to_first_token_approx_s': time_to_first_token,
        }
    
    def _analyze_pytorch_profiling(self, prof: torch.profiler.profile) -> Dict[str, Any]:
        """Analyze PyTorch profiler results for BitsAndBytes operations"""
        
        key_averages = prof.key_averages()
        
        # Find BitsAndBytes operations
        bnb_operations = []
        total_cuda_time = 0
        bnb_total_time = 0
        
        for event in key_averages:
            device_time_total = getattr(event, 'device_time_total', 0) or getattr(event, 'cuda_time_total', 0)
            total_cuda_time += device_time_total
            
            op_classification = self.classify_bnb_operation(event.key)
            
            event_data = {
                'name': event.key,
                'classification': op_classification,
                'device_time_us': getattr(event, 'device_time', 0) or getattr(event, 'cuda_time', 0),
                'device_time_total_us': device_time_total,
                'cpu_time_total_us': getattr(event, 'cpu_time_total', 0),
                'count': event.count,
                'avg_time_us': device_time_total / event.count if event.count > 0 else 0,
            }
            
            if 'bnb' in op_classification or 'dequant' in op_classification:
                bnb_operations.append(event_data)
                bnb_total_time += device_time_total
        
        # Sort by time
        bnb_operations.sort(key=lambda x: x['device_time_total_us'], reverse=True)
        
        return {
            'bnb_operations': bnb_operations[:10],  # Top 10 operations
            'bnb_time_percentage': (bnb_total_time / total_cuda_time * 100) if total_cuda_time > 0 else 0,
            'bnb_total_time_us': bnb_total_time,
            'total_cuda_time_us': total_cuda_time,
        }
    
    def classify_bnb_operation(self, op_name: str) -> str:
        """Classify operation as BitsAndBytes-specific or general"""
        op_lower = op_name.lower()
        
        for category, patterns in self.bnb_operation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, op_lower):
                    return f"bnb_{category}"
        
        # Fallback to general classification
        if any(x in op_lower for x in ['gemm', 'sgemm', 'hgemm', 'matmul', 'mm', 'bmm']):
            return 'matrix_multiply'
        elif any(x in op_lower for x in ['attention', 'softmax']):
            return 'attention'
        elif any(x in op_lower for x in ['copy', 'clone', 'cat', 'view']):
            return 'memory_ops'
        else:
            return 'other'
    
    def load_test_prompts(self, batch_size: int) -> List[str]:
        """Load test prompts for profiling"""
        
        try:
            dataset = load_dataset("gsm8k", "main", split="test")
            samples = dataset.select(range(min(batch_size, len(dataset))))
            
            prompts = []
            for sample in samples:
                prompt = f"Solve this math problem step by step:\n\n{sample['question']}\n\nAnswer:"
                prompts.append(prompt)
            
            return prompts
            
        except Exception as e:
            print(f"⚠️ Using fallback prompts: {e}")
            base_prompts = [
                "What is 25 + 37?",
                "Calculate 12 × 8 + 15",
                "If 5 apples cost $2, how much do 20 apples cost?",
                "Find the area of a rectangle with length 8 and width 6.",
                "Solve: 2x + 5 = 15",
                "What is 15% of 240?",
                "Convert 75°F to Celsius",
                "Calculate the square root of 144"
            ]
            return (base_prompts * ((batch_size // len(base_prompts)) + 1))[:batch_size]
    
    def generate_profiling_report(self, results: Dict[int, Dict[str, Any]], tag: str = "") -> str:
        """Generate comprehensive profiling report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag_suffix = f"_{tag}" if tag else ""
        report_path = self.output_dir / f"bnb_profiling_report{tag_suffix}_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("BITSANDBYTES PROFILING REPORT\n")
            f.write("=" * 40 + "\n\n")
            f.write("PyTorch Profiler + Operation Counter Analysis\n")
            if tag:
                f.write(f"Configuration: {tag}\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Counter tracking: {'Enabled' if self.use_counters else 'Disabled'}\n\n")
            
            # Performance Summary
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 20 + "\n")
            
            # Check for identical counter values (indicates reset bug)
            counter_values_check = {}
            
            for batch_size, result in results.items():
                f.write(f"Batch {batch_size}:\n")
                f.write(f"  Tokens/sec: {result.get('tokens_per_sec', 0):.1f}\n")
                f.write(f"  New tokens/sec: {result.get('new_tokens_per_sec', 0):.1f}\n")
                f.write(f"  End-to-end latency: {result.get('end_to_end_latency_s', 0):.2f}s\n")
                f.write(f"  Latency per batch item: {result.get('latency_per_batch_item_s', 0):.2f}s\n")
                f.write(f"  Tokens per batch item: {result.get('tokens_per_batch_item', 0):.1f}\n")
                f.write(f"  Memory: {result.get('memory_stats', {}).get('peak_memory_mb', 0):.0f} MB\n")
                
                # BitsAndBytes overhead
                pytorch_data = result.get('pytorch_profiling', {})
                bnb_pct = pytorch_data.get('bnb_time_percentage', 0)
                f.write(f"  BnB overhead: {bnb_pct:.1f}%\n")
                
                # Operation counts with validation
                if self.use_counters and 'operation_counters' in result:
                    counters = result['operation_counters']
                    if 'error' not in counters:
                        nf4_count = counters.get('nf4_lookup_count', 0)
                        scaling_count = counters.get('scaling_factor_count', 0)
                        memory_count = counters.get('memory_access_count', 0)
                        kernel_count = counters.get('kernel_call_count', 0)
                        
                        f.write(f"  NF4 lookups: {nf4_count:,}\n")
                        f.write(f"  Scaling accesses: {scaling_count:,}\n")
                        f.write(f"  Memory accesses: {memory_count:,}\n")
                        f.write(f"  Kernel calls: {kernel_count:,}\n")
                        
                        # Track for duplicate detection
                        counter_signature = (nf4_count, scaling_count, memory_count, kernel_count)
                        if counter_signature in counter_values_check:
                            counter_values_check[counter_signature].append(batch_size)
                        else:
                            counter_values_check[counter_signature] = [batch_size]
                f.write("\n")
            
            # Check for duplicate counter values
            duplicates_found = False
            for signature, batch_sizes in counter_values_check.items():
                if len(batch_sizes) > 1:
                    duplicates_found = True
                    f.write(f"🚨 CRITICAL ISSUE: Identical counter values for batches {batch_sizes}\n")
                    f.write(f"   Values: NF4={signature[0]:,}, Scaling={signature[1]:,}, Memory={signature[2]:,}, Kernels={signature[3]:,}\n")
                    f.write(f"   This indicates counter reset failure between batch runs!\n\n")
            
            if duplicates_found:
                f.write("⚠️ COUNTER RESET BUG DETECTED!\n")
                f.write("   The profiler is not properly resetting counters between batch sizes.\n")
                f.write("   All counter data in this report may be cumulative rather than per-batch.\n")
                f.write("   Recommendation: Fix counter reset logic and re-run profiling.\n\n")
            
            # Add latency scaling analysis
            f.write("LATENCY SCALING ANALYSIS:\n")
            f.write("-" * 25 + "\n")
            
            batch_sizes = sorted(results.keys())
            if len(batch_sizes) >= 2:
                for i, batch_size in enumerate(batch_sizes[1:], 1):
                    prev_batch = batch_sizes[i-1]
                    
                    current_result = results[batch_size]
                    prev_result = results[prev_batch]
                    
                    # Calculate scaling factors
                    batch_scaling = batch_size / prev_batch
                    throughput_scaling = current_result.get('tokens_per_sec', 0) / max(prev_result.get('tokens_per_sec', 0), 0.001)
                    latency_scaling = current_result.get('end_to_end_latency_s', 0) / max(prev_result.get('end_to_end_latency_s', 0), 0.001)
                    
                    # Counter scaling (if available)
                    counter_scaling = "N/A"
                    if 'operation_counters' in current_result and 'operation_counters' in prev_result:
                        current_ops = current_result['operation_counters'].get('total_operations', 0)
                        prev_ops = prev_result['operation_counters'].get('total_operations', 0)
                        if prev_ops > 0:
                            counter_scaling = f"{current_ops / prev_ops:.2f}x"
                    
                    f.write(f"Batch {prev_batch} → {batch_size} (expected {batch_scaling:.1f}x scaling):\n")
                    f.write(f"  Throughput scaling: {throughput_scaling:.2f}x\n")
                    f.write(f"  Latency scaling: {latency_scaling:.2f}x\n")
                    f.write(f"  Counter scaling: {counter_scaling}\n")
                    
                    # Diagnosis
                    if isinstance(counter_scaling, str) and counter_scaling != "N/A":
                        counter_scale_val = float(counter_scaling.replace('x', ''))
                        if abs(counter_scale_val - 1.0) < 0.01:  # Essentially no scaling
                            f.write(f"  🚨 DIAGNOSIS: Identical counter values suggest reset bug!\n")
                        elif abs(counter_scale_val - batch_scaling) < 0.2:  # Close to expected
                            f.write(f"  ✅ DIAGNOSIS: Counter scaling looks normal\n")
                        else:
                            f.write(f"  ⚠️ DIAGNOSIS: Unexpected counter scaling pattern\n")
                    
                    if abs(throughput_scaling - batch_scaling) < 0.3:  # Within 30% of expected
                        f.write(f"  ✅ Throughput scaling is reasonable\n")
                    elif throughput_scaling > batch_scaling * 1.3:
                        f.write(f"  🚀 Better than expected throughput scaling (efficient batching)\n")
                    else:
                        f.write(f"  ⚠️ Lower than expected throughput scaling (bottlenecks)\n")
                    
                    f.write("\n")
            f.write("\n")
            
            # Detailed Operation Analysis
            f.write("DETAILED OPERATION ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            
            for batch_size, result in results.items():
                f.write(f"\nBatch {batch_size} Details:\n")
                f.write("-" * 20 + "\n")
                
                # Top BitsAndBytes operations from PyTorch profiler
                pytorch_data = result.get('pytorch_profiling', {})
                bnb_ops = pytorch_data.get('bnb_operations', [])
                
                if bnb_ops:
                    f.write("Top BitsAndBytes Operations (PyTorch Profiler):\n")
                    for i, op in enumerate(bnb_ops[:5], 1):
                        f.write(f"  {i}. {op['name'][:60]}...\n")
                        f.write(f"     Classification: {op['classification']}\n")
                        f.write(f"     Total time: {op['device_time_total_us']/1000:.1f} ms\n")
                        f.write(f"     Count: {op['count']}\n")
                        f.write(f"     Avg time: {op['avg_time_us']:.1f} μs\n\n")
                
                # Operation counter breakdown
                if self.use_counters and 'operation_counters' in result:
                    counters = result['operation_counters']
                    if 'error' not in counters:
                        f.write("Operation Counter Breakdown:\n")
                        f.write(f"  NF4 lookups: {counters.get('nf4_lookup_count', 0):,}\n")
                        f.write(f"  Scaling factor accesses: {counters.get('scaling_factor_count', 0):,}\n")
                        f.write(f"  Memory accesses: {counters.get('memory_access_count', 0):,}\n")
                        f.write(f"  Kernel calls: {counters.get('kernel_call_count', 0):,}\n")
                        f.write(f"  Total operations: {counters.get('total_operations', 0):,}\n")
                        
                        # Calculate ratios
                        kernel_calls = counters.get('kernel_call_count', 0)
                        if kernel_calls > 0:
                            f.write(f"\nAverages per kernel call:\n")
                            f.write(f"  NF4 lookups/kernel: {counters.get('nf4_lookup_count', 0) / kernel_calls:.1f}\n")
                            f.write(f"  Scaling accesses/kernel: {counters.get('scaling_factor_count', 0) / kernel_calls:.1f}\n")
                            f.write(f"  Memory accesses/kernel: {counters.get('memory_access_count', 0) / kernel_calls:.1f}\n")
                        
                        f.write("\n")
            
            # Optimization Recommendations
            f.write("OPTIMIZATION ANALYSIS:\n")
            f.write("-" * 25 + "\n")
            
            if self.use_counters:
                f.write("Based on operation counters:\n\n")
                
                f.write("1. 🎯 NF4 Lookup Optimization:\n")
                f.write("   - Current: Tree-based lookup in registers\n")
                f.write("   - Target: Constant memory lookup table\n")
                f.write("   - Expected: Reduced register pressure\n\n")
                
                f.write("2. 🎯 Scaling Factor Optimization:\n")
                f.write("   - Current: Global memory access per element\n")
                f.write("   - Target: Shared memory cooperative loading\n")
                f.write("   - Expected: Better memory bandwidth utilization\n\n")
                
                f.write("3. 🎯 Memory Access Optimization:\n")
                f.write("   - Current: Individual element access\n")
                f.write("   - Target: Vectorized float4 access patterns\n")
                f.write("   - Expected: Improved memory coalescing\n\n")
                
                f.write("VALIDATION METHODOLOGY:\n")
                f.write("- Use counters to verify operations are being tracked\n")
                f.write("- Use PyTorch profiler to measure end-to-end timing\n")
                f.write("- Compare before/after optimization implementations\n")
                f.write("- Focus on throughput (tokens/sec) improvement\n")
            else:
                f.write("Counter tracking not available.\n")
                f.write("Using PyTorch profiler timing only.\n")
                f.write("Consider enabling instrumentation for granular validation.\n")
        
        print(f"📊 Profiling report saved: {report_path}")
        return str(report_path)
    
    def run_profiling(self, model_name: str, batch_sizes: List[int] = None, tag: str = "") -> Dict[int, Dict[str, Any]]:
        """Run profiling across multiple batch sizes"""
        
        if batch_sizes is None:
            batch_sizes = [2, 4]  # Start with batch_size >= 2 for blockwise operations
        
        print(f"🚀 Starting BitsAndBytes profiling")
        print(f"Model: {model_name}")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Counters: {'Enabled' if self.use_counters else 'Disabled'}")
        if tag:
            print(f"Configuration tag: {tag}")
        
        # Warn about batch size for blockwise operations
        if min(batch_sizes) < 2:
            print("⚠️ Warning: Batch sizes < 2 may not trigger NF4 blockwise dequantization")
        
        # Load model ONCE and reuse for all batch sizes
        if not self.load_model_with_bnb(model_name):
            return {}
        
        results = {}
        
        for i, batch_size in enumerate(batch_sizes):
            try:
                print(f"\n📊 Profiling batch size {batch_size} (run {i+1}/{len(batch_sizes)})...")
                
                # Clear GPU state between runs
                if i > 0:  # Not needed for first run
                    print("   🧹 Clearing GPU state between runs...")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    time.sleep(1)  # Brief pause
                
                result = self.profile_with_counters(batch_size)
                result['model_name'] = model_name  # Add model name to result
                
                results[batch_size] = result
                
                # Print summary with validation
                tokens_sec = result['tokens_per_sec']
                if self.use_counters and 'operation_counters' in result:
                    counter_data = result['operation_counters']
                    if 'error' not in counter_data:
                        total_ops = counter_data['total_operations']
                        kernel_calls = counter_data['kernel_call_count']
                        print(f"✅ Batch {batch_size}: {tokens_sec:.1f} tok/s, {total_ops:,} ops, {kernel_calls} kernels")
                        
                        # Immediate validation against PyTorch profiler
                        pytorch_data = result.get('pytorch_profiling', {})
                        pytorch_kernel_count = 0
                        for op in pytorch_data.get('bnb_operations', []):
                            if 'kDequantizeBlockwise' in op['name']:
                                pytorch_kernel_count = op['count']
                                break
                        
                        if pytorch_kernel_count > 0 and kernel_calls != pytorch_kernel_count:
                            print(f"   ⚠️ Counter/PyTorch mismatch: {kernel_calls} vs {pytorch_kernel_count} kernels")
                        elif pytorch_kernel_count > 0:
                            print(f"   ✅ Counter/PyTorch agreement: {kernel_calls} kernels")
                            
                    else:
                        print(f"✅ Batch {batch_size}: {tokens_sec:.1f} tok/s, counter error: {counter_data['error']}")
                else:
                    pytorch_data = result.get('pytorch_profiling', {})
                    bnb_pct = pytorch_data.get('bnb_time_percentage', 0)
                    print(f"✅ Batch {batch_size}: {tokens_sec:.1f} tok/s, {bnb_pct:.1f}% BnB overhead")
                
                # Check for suspicious patterns immediately
                if i > 0:  # Compare with previous result
                    prev_batch = batch_sizes[i-1]
                    prev_result = results[prev_batch]
                    
                    # Check PyTorch profiler data for suspicious patterns
                    current_pytorch = result.get('pytorch_profiling', {})
                    prev_pytorch = prev_result.get('pytorch_profiling', {})
                    
                    current_bnb_ops = current_pytorch.get('bnb_operations', [])
                    prev_bnb_ops = prev_pytorch.get('bnb_operations', [])
                    
                    if current_bnb_ops and prev_bnb_ops:
                        current_kernel_count = current_bnb_ops[0].get('count', 0) if current_bnb_ops else 0
                        prev_kernel_count = prev_bnb_ops[0].get('count', 0) if prev_bnb_ops else 0
                        
                        if current_kernel_count == prev_kernel_count and current_kernel_count > 0:
                            print(f"   🚨 SUSPICIOUS: Identical PyTorch kernel counts ({current_kernel_count})")
                            print(f"       This suggests PyTorch profiler may also have accumulation issues")
                
                # Brief pause between runs to ensure separation
                if i < len(batch_sizes) - 1:
                    time.sleep(2)
                
            except Exception as e:
                print(f"❌ Batch {batch_size} failed: {e}")
                import traceback
                traceback.print_exc()
                results[batch_size] = {'error': str(e)}
        
        # Generate report with enhanced diagnostics
        if results:
            report_path = self.generate_profiling_report(results, tag)
            print(f"\n✅ Profiling complete!")
            print(f"📊 Report: {report_path}")
            
            # Immediate cross-validation summary
            print(f"\n🔍 Quick Validation Summary:")
            pytorch_kernel_counts = []
            counter_kernel_counts = []
            
            for batch_size, result in results.items():
                if 'error' not in result:
                    # PyTorch profiler kernel count
                    pytorch_data = result.get('pytorch_profiling', {})
                    bnb_ops = pytorch_data.get('bnb_operations', [])
                    pytorch_count = bnb_ops[0].get('count', 0) if bnb_ops else 0
                    pytorch_kernel_counts.append(pytorch_count)
                    
                    # Counter kernel count
                    counter_data = result.get('operation_counters', {})
                    counter_count = counter_data.get('kernel_call_count', 0) if 'error' not in counter_data else 0
                    counter_kernel_counts.append(counter_count)
                    
                    print(f"  Batch {batch_size}: PyTorch={pytorch_count}, Counters={counter_count} kernels")
            
            # Check for identical values
            if len(set(pytorch_kernel_counts)) == 1 and pytorch_kernel_counts[0] > 0:
                print(f"  🚨 PyTorch profiler shows identical kernel counts - profiler accumulation bug!")
            elif len(set(pytorch_kernel_counts)) > 1:
                print(f"  ✅ PyTorch profiler shows proper scaling")
                
            if len(set(counter_kernel_counts)) == 1 and counter_kernel_counts[0] > 0:
                print(f"  🚨 Counter data shows identical kernel counts - counter reset bug!")
            elif len(set(counter_kernel_counts)) > 1:
                print(f"  ✅ Counter data shows proper scaling")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="BitsAndBytes Profiler with Counters")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--batch_sizes", default="2,4", help="Comma-separated batch sizes (>=2 recommended for blockwise)")
    parser.add_argument("--output_dir", default="bnb_counter_profiling", help="Output directory")
    parser.add_argument("--counters", action="store_true", 
                       help="Enable operation counters (requires instrumented BitsAndBytes)")
    parser.add_argument("--tag", default="", help="Configuration tag for output files")
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(',')]
    
    # Validate batch sizes for blockwise operations
    if min(batch_sizes) < 2:
        print("⚠️ Warning: Batch sizes < 2 may not trigger NF4 blockwise dequantization kernels")
        print("   Consider using --batch_sizes 2,4 for better coverage")
    
    try:
        # Create profiler
        profiler = BnBCounterProfiler(args.output_dir, use_counters=args.counters)
        
        # Run profiling
        results = profiler.run_profiling(args.model, batch_sizes, args.tag)
        
        if not results:
            print("❌ No profiling results generated")
            return
        
        print("\n" + "=" * 50)
        print("✅ BITSANDBYTES PROFILING COMPLETE!")
        print("=" * 50)
        
        # Print key findings
        print("\n📈 Performance Results:")
        for batch_size, result in results.items():
            if 'error' not in result:
                tokens_sec = result.get('tokens_per_sec', 0)
                bnb_pct = result.get('pytorch_profiling', {}).get('bnb_time_percentage', 0)
                print(f"  Batch {batch_size}: {tokens_sec:.1f} tok/s, {bnb_pct:.1f}% BnB overhead")
        
        if profiler.use_counters:
            print("\n🔢 Operation Validation:")
            for batch_size, result in results.items():
                if 'operation_counters' in result and 'error' not in result['operation_counters']:
                    counters = result['operation_counters']
                    print(f"  Batch {batch_size}: {counters['total_operations']:,} ops, {counters['kernel_call_count']} kernels")
            
            print("\n🎯 Use counters to validate optimization effectiveness:")
            print("  • Track operation count changes after optimizations")
            print("  • Verify kernel invocation patterns")
            print("  • Measure throughput improvements")
        
        print(f"\n📊 Detailed analysis: {profiler.output_dir}/*_report_*.txt")
        
    except KeyboardInterrupt:
        print("\n⏹️ Profiling interrupted")
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()