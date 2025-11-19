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
    print("[OK] Kernel counters available")
except ImportError:
    COUNTERS_AVAILABLE = False
    print("[WARNING] Kernel counters not available - using PyTorch profiling only")

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
                # Check if instrumentation is actually available
                if self.counter_extractor.instrumentation_available:
                    print("Operation counter tracking enabled")
                else:
                    print("Instrumentation not available - using PyTorch profiling only")
                    self.use_counters = False
            except Exception as e:
                print(f"Failed to initialize counter extractor: {e}")
                self.use_counters = False
        
        if not self.use_counters:
            print("Using PyTorch profiling only")
        
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
    
    def calculate_memory_bandwidth_efficiency(self, counters: Dict[str, Any], kernel_time_us: float) -> Dict[str, float]:
        """Calculate memory bandwidth utilization metrics"""
        
        results = {}
        
        if counters.get('bytes_loaded', 0) > 0 and kernel_time_us > 0:
            # Convert microseconds to seconds
            kernel_time_s = kernel_time_us / 1e6
            
            # Calculate achieved bandwidth
            bytes_gb = counters['bytes_loaded'] / 1e9
            bandwidth_gb_s = bytes_gb / kernel_time_s
            
            # Determine theoretical bandwidth based on GPU
            if torch.cuda.is_available():
                device_props = torch.cuda.get_device_properties(0)
                device_name = device_props.name.lower()
                
                if 'a100' in device_name:
                    theoretical_bandwidth = 1935  # GB/s for A100-80GB
                elif 'v100' in device_name:
                    theoretical_bandwidth = 900   # GB/s for V100
                elif 'a40' in device_name:
                    theoretical_bandwidth = 696   # GB/s for A40
                elif 'a10' in device_name:
                    theoretical_bandwidth = 600   # GB/s for A10
                else:
                    theoretical_bandwidth = 500   # Conservative default
            else:
                theoretical_bandwidth = 500
            
            efficiency = (bandwidth_gb_s / theoretical_bandwidth) * 100
            
            results = {
                'bandwidth_gb_s': bandwidth_gb_s,
                'theoretical_bandwidth_gb_s': theoretical_bandwidth,
                'bandwidth_efficiency': efficiency,
                'total_gb_transferred': bytes_gb
            }
        
        return results

    def load_model_with_bnb(self, model_name: str) -> bool:
        """Load model with BitsAndBytes quantization"""
        
        print(f"[Loading] Loading {model_name} with BitsAndBytes...")
        
        try:
            from transformers import BitsAndBytesConfig
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"   [Cleanup] GPU memory before loading: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
            
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
            
            print(f"   [Stats] Quantization: {quantized_layers}/{total_layers} layers")
            
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.memory_allocated() / 1024**3
                print(f"   [Stats] GPU memory after loading: {gpu_memory_gb:.1f} GB")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False
    
    def profile_with_counters(self, batch_size: int, max_tokens: int = 100) -> Dict[str, Any]:
        """Profile using PyTorch profiler and operation counters"""
        
        print(f"[Profiling] Profiling - Batch size {batch_size}")
        
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
        print("   [Warmup] Warming up...")
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
        
        # ENHANCED RESET LOGIC WITH RETRY MECHANISM
        if self.use_counters:
            print("   [Reset] Resetting counters with enhanced verification...")
            
            # Attempt 1: Normal reset
            reset_success = self.counter_extractor.reset_counters()
            
            if not reset_success:
                print("   [Retry] Retry 1: Force GPU memory clear + reset...")
                # Attempt 2: Force memory clear first
                self.counter_extractor.force_gpu_memory_clear()
                time.sleep(0.5)  # Longer pause for large models
                reset_success = self.counter_extractor.reset_counters()
                
                if not reset_success:
                    print("   [Retry] Retry 2: Device reset + counter reset...")
                    # Attempt 3: More aggressive approach
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    time.sleep(1.0)  # Even longer pause
                    reset_success = self.counter_extractor.reset_counters()
                    
                    if not reset_success:
                        print("   [CRITICAL] CRITICAL: All counter reset attempts failed!")
                        print("   [WARNING] Results will be cumulative - interpret with caution")
                        # Continue anyway - PyTorch profiler still works
        
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
        time_to_first_token = total_time / output_tokens if output_tokens > 0 else 0
        tokens_per_batch_item = total_tokens / batch_size
        time_per_batch_item = total_time / batch_size
        
        print(f"   [Speed] Generated {output_tokens} new tokens in {total_time:.2f}s")
        print(f"   [Throughput] Throughput: {tokens_per_sec:.1f} total tokens/sec, {new_tokens_per_sec:.1f} new tokens/sec")
        print(f"   [Latency] Latency: {total_time:.2f}s total, {time_per_batch_item:.2f}s per batch item")
        print(f"   [Efficiency] Efficiency: {tokens_per_batch_item:.1f} tokens per batch item")
        
        # Memory stats
        memory_stats = {
            'peak_memory_mb': torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            'current_memory_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
        }
        
        # PyTorch profiler analysis
        pytorch_analysis = self._analyze_pytorch_profiling(prof)
        
        # Counter analysis with validation
        # Counter analysis if available
        counter_analysis = {}
        if self.use_counters:
            try:
                counter_analysis = self.counter_extractor.get_comprehensive_counter_report()
                if counter_analysis.get('instrumentation_available', False):
                    total_ops = counter_analysis.get('total_operations', 0)
                    print(f"   Operations counted: {total_ops:,}")
                    
                    if total_ops > 0:
                        nf4_count = counter_analysis.get('nf4_lookup_count', 0)
                        kernel_count = counter_analysis.get('kernel_call_count', 0)
                        
                        if kernel_count > 0:
                            ops_per_kernel = total_ops / kernel_count
                            nf4_per_kernel = nf4_count / kernel_count
                            
                            print(f"   {ops_per_kernel:,.0f} ops/kernel, {nf4_per_kernel:,.0f} NF4/kernel")
                            
                            if ops_per_kernel < 1000:
                                print(f"   Very low ops/kernel - likely counter reset issues")
                            if nf4_per_kernel < 100:
                                print(f"   Very low NF4/kernel - may not be hitting blockwise kernels")
                            
                            if kernel_count % 864 == 0 and kernel_count > 864:
                                multiple = kernel_count // 864
                                print(f"   ACCUMULATION DETECTED: {multiple}x base kernel count")
                                print(f"       This indicates counter reset failed {multiple-1} times")
                            
                            counter_analysis['reset_success'] = reset_success
                            counter_analysis['reset_attempts'] = 3 if not reset_success else 1
                else:
                    print("   Instrumentation not available")
                    counter_analysis = {'instrumentation_available': False}
                                
            except Exception as e:
                print(f"   Failed to extract counters: {e}")
                counter_analysis = {'error': str(e), 'instrumentation_available': False}
        else:
            counter_analysis = {'instrumentation_available': False}

        # Rest of the function remains the same...
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
            'end_to_end_latency_s': total_time,
            'latency_per_batch_item_s': time_per_batch_item,
            'tokens_per_batch_item': tokens_per_batch_item,
            'time_to_first_token_approx_s': time_to_first_token,
        }
    
    def profile_with_responses(self, batch_size: int, max_tokens: int = 100, save_responses: bool = True) -> Dict[str, Any]:
        """Profile with response generation and optional response logging"""
        
        print(f"[Profiling] Profiling with response logging - Batch size {batch_size}")
        
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
        print("   [Warmup] Warming up...")
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
        
        # ENHANCED RESET LOGIC (same as existing)
        if self.use_counters:
            print("   [Reset] Resetting counters with enhanced verification...")
            reset_success = self.counter_extractor.reset_counters()
            
            if not reset_success:
                print("   [Retry] Retry 1: Force GPU memory clear + reset...")
                self.counter_extractor.force_gpu_memory_clear()
                time.sleep(0.5)
                reset_success = self.counter_extractor.reset_counters()
                
                if not reset_success:
                    print("   [Retry] Retry 2: Device reset + counter reset...")
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    time.sleep(1.0)
                    reset_success = self.counter_extractor.reset_counters()
                    
                    if not reset_success:
                        print("   [CRITICAL] CRITICAL: All counter reset attempts failed!")
        
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
        
        # Store responses for logging
        generated_responses = []
        input_prompts = prompts.copy()
        
        # Run profiling with response capture
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
        
        # Decode responses for logging
        if save_responses:
            for i in range(batch_size):
                input_length = inputs['input_ids'][i].shape[0]
                full_response = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
                
                input_text = self.tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True)
                generated_text = full_response[len(input_text):].strip()
                
                response_entry = {
                    'prompt_id': i,
                    'input_prompt': input_prompts[i],
                    'generated_response': generated_text,
                    'full_response': full_response,
                    'input_tokens': input_length,
                    'total_tokens': len(outputs[i]),
                    'generated_tokens': len(outputs[i]) - input_length
                }
                generated_responses.append(response_entry)
        
        # Calculate metrics (same as existing profile_with_counters)
        total_time = end_time - start_time
        output_tokens = outputs.numel() - input_tokens
        total_tokens = outputs.numel()
        
        tokens_per_sec = total_tokens / total_time
        new_tokens_per_sec = output_tokens / total_time
        
        time_to_first_token = total_time / output_tokens if output_tokens > 0 else 0
        tokens_per_batch_item = total_tokens / batch_size
        time_per_batch_item = total_time / batch_size
        
        print(f"   [Speed] Generated {output_tokens} new tokens in {total_time:.2f}s")
        print(f"   [Throughput] Throughput: {tokens_per_sec:.1f} total tokens/sec, {new_tokens_per_sec:.1f} new tokens/sec")
        print(f"   [Latency] Latency: {total_time:.2f}s total, {time_per_batch_item:.2f}s per batch item")
        print(f"   [Efficiency] Efficiency: {tokens_per_batch_item:.1f} tokens per batch item")
        
        # Memory stats
        memory_stats = {
            'peak_memory_mb': torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            'current_memory_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
        }
        
        # PyTorch profiler analysis (use existing method)
        pytorch_analysis = self._analyze_pytorch_profiling(prof)
        
        # Counter analysis (same as existing)
        # Counter analysis if available
        counter_analysis = {}
        if self.use_counters:
            try:
                counter_analysis = self.counter_extractor.get_comprehensive_counter_report()
                if counter_analysis.get('instrumentation_available', False):
                    total_ops = counter_analysis.get('total_operations', 0)
                    print(f"   Operations counted: {total_ops:,}")
                    
                    if total_ops > 0:
                        nf4_count = counter_analysis.get('nf4_lookup_count', 0)
                        kernel_count = counter_analysis.get('kernel_call_count', 0)
                        
                        if kernel_count > 0:
                            ops_per_kernel = total_ops / kernel_count
                            nf4_per_kernel = nf4_count / kernel_count
                            
                            print(f"   {ops_per_kernel:,.0f} ops/kernel, {nf4_per_kernel:,.0f} NF4/kernel")
                            
                            if ops_per_kernel < 1000:
                                print(f"   Very low ops/kernel - likely counter reset issues")
                            if nf4_per_kernel < 100:
                                print(f"   Very low NF4/kernel - may not be hitting blockwise kernels")
                            
                            if kernel_count % 864 == 0 and kernel_count > 864:
                                multiple = kernel_count // 864
                                print(f"   ACCUMULATION DETECTED: {multiple}x base kernel count")
                                print(f"       This indicates counter reset failed {multiple-1} times")
                            
                            counter_analysis['reset_success'] = reset_success
                            counter_analysis['reset_attempts'] = 3 if not reset_success else 1
                else:
                    print("   Instrumentation not available")
                    counter_analysis = {'instrumentation_available': False}
                                
            except Exception as e:
                print(f"   Failed to extract counters: {e}")
                counter_analysis = {'error': str(e), 'instrumentation_available': False}
        else:
            counter_analysis = {'instrumentation_available': False}

        # Save responses to JSON if requested
        if save_responses and generated_responses:
            self._save_responses_json(batch_size, generated_responses, total_time)
        
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
            'end_to_end_latency_s': total_time,
            'latency_per_batch_item_s': time_per_batch_item,
            'tokens_per_batch_item': tokens_per_batch_item,
            'time_to_first_token_approx_s': time_to_first_token,
            'generated_responses': generated_responses if save_responses else [],
            'response_count': len(generated_responses)
        }

    def _save_responses_json(self, batch_size: int, responses: List[Dict], total_time: float):
        """Save generated responses to JSON file - JSON ONLY"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        responses_file = self.output_dir / f"responses_batch_{batch_size}_{timestamp}.json"
        
        response_data = {
            'metadata': {
                'batch_size': batch_size,
                'timestamp': datetime.now().isoformat(),
                'total_generation_time_s': total_time,
                'model_name': getattr(self.model, 'name_or_path', 'unknown'),
                'response_count': len(responses),
                'avg_time_per_response_s': total_time / len(responses) if responses else 0
            },
            'responses': responses
        }
        
        with open(responses_file, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False)
        
        print(f"   [Saved] Responses saved: {responses_file}")

    def run_profiling_with_responses(self, model_name: str, batch_sizes: List[int] = None, tag: str = "", save_responses: bool = True) -> Dict[int, Dict[str, Any]]:
        """Run profiling across multiple batch sizes with response generation"""
        
        if batch_sizes is None:
            batch_sizes = [2, 4]
        
        print(f"[Starting] Starting BitsAndBytes profiling with response generation")
        print(f"Model: {model_name}")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Counters: {'Enabled' if self.use_counters else 'Disabled'}")
        print(f"Response logging: {'Enabled' if save_responses else 'Disabled'}")
        if tag:
            print(f"Configuration tag: {tag}")
        
        # Load model
        if not self.load_model_with_bnb(model_name):
            return {}
        
        results = {}
        
        for i, batch_size in enumerate(batch_sizes):
            try:
                print(f"\n[Profiling] Profiling batch size {batch_size} (run {i+1}/{len(batch_sizes)})...")
                
                if i > 0:
                    print("   [Cleanup] Clearing GPU state between runs...")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    time.sleep(1)
                
                # Use the new method with response logging
                result = self.profile_with_responses(batch_size, save_responses=save_responses)
                result['model_name'] = model_name
                
                results[batch_size] = result
                
                # Print summary
                tokens_sec = result['tokens_per_sec']
                response_count = result['response_count']
                
                if self.use_counters and 'operation_counters' in result:
                    counter_data = result['operation_counters']
                    if counter_data.get('instrumentation_available', False):
                        total_ops = counter_data.get('total_operations', 0)
                        kernel_calls = counter_data.get('kernel_call_count', 0)
                        print(f"[OK] Batch {batch_size}: {tokens_sec:.1f} tok/s, {total_ops:,} ops, {kernel_calls} kernels")
                    else:
                        print(f"[OK] Batch {batch_size}: {tokens_sec:.1f} tok/s, instrumentation not available")
                else:
                    pytorch_data = result.get('pytorch_profiling', {})
                    bnb_pct = pytorch_data.get('bnb_time_percentage', 0)
                    print(f"[OK] Batch {batch_size}: {tokens_sec:.1f} tok/s, {bnb_pct:.1f}% BnB overhead")
                
                if i < len(batch_sizes) - 1:
                    time.sleep(2)
                    
            except Exception as e:
                print(f"[ERROR] Batch {batch_size} failed: {e}")
                import traceback
                traceback.print_exc()
                results[batch_size] = {'error': str(e)}
        
        # Generate reports
        if results:
            report_path = self.generate_profiling_report(results, tag)
            print(f"\n[Complete] Profiling complete!")
            print(f"[Report] Performance report: {report_path}")
            
            if save_responses:
                print(f"[Saved] Generated responses saved to: {self.output_dir}/responses_batch_*")
                
                # Create consolidated responses file
                consolidated_file = self.output_dir / f"all_responses_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                consolidated_data = {
                    'metadata': {
                        'model_name': model_name,
                        'timestamp': datetime.now().isoformat(),
                        'tag': tag,
                        'batch_sizes_tested': batch_sizes
                    },
                    'responses_by_batch': {}
                }
                
                for batch_size, result in results.items():
                    if 'generated_responses' in result:
                        consolidated_data['responses_by_batch'][batch_size] = {
                            'batch_size': batch_size,
                            'performance_metrics': {
                                'tokens_per_sec': result.get('tokens_per_sec', 0),
                                'new_tokens_per_sec': result.get('new_tokens_per_sec', 0),
                                'total_time_s': result.get('total_time_s', 0),
                                'latency_per_item_s': result.get('latency_per_batch_item_s', 0)
                            },
                            'responses': result['generated_responses']
                        }
                
                with open(consolidated_file, 'w', encoding='utf-8') as f:
                    json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
                
                print(f"[Consolidated] Consolidated responses: {consolidated_file}")
        
        return results


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
            dataset = load_dataset("cais/mmlu", "all", split="test")
            samples = dataset.select(range(min(batch_size, len(dataset))))
            
            prompts = []
            for sample in samples:
                prompt = f"Solve this math problem step by step:\n\n{sample['question']}\n\nAnswer:"
                prompts.append(prompt)
            
            return prompts
            
        except Exception as e:
            print(f"[WARNING] Using fallback prompts: {e}")
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
        """Generate comprehensive profiling report with fallback for missing instrumentation"""
        
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
            
            # Check if any results have instrumentation
            has_instrumentation = False
            for result in results.values():
                if 'operation_counters' in result:
                    counter_data = result['operation_counters']
                    if counter_data.get('instrumentation_available', False):
                        has_instrumentation = True
                        break
            
            if has_instrumentation:
                f.write("Counter tracking: Enabled\n\n")
            else:
                f.write("Counter tracking: Not available (compiled without BNB_ENABLE_INSTRUMENTATION)\n")
                f.write("Report will focus on PyTorch profiler analysis\n\n")
            
            # Performance Summary
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 20 + "\n")
            
            # Check for identical counter values (indicates reset bug) - only if instrumentation available
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
                
                # Basic operation counts - only if instrumentation available
                if has_instrumentation and 'operation_counters' in result:
                    counters = result['operation_counters']
                    if counters.get('instrumentation_available', False):
                        f.write(f"  NF4 lookups: {counters.get('nf4_lookup_count', 0):,}\n")
                        f.write(f"  Scaling accesses: {counters.get('scaling_factor_count', 0):,}\n")
                        f.write(f"  Memory accesses: {counters.get('memory_access_count', 0):,}\n")
                        f.write(f"  Kernel calls: {counters.get('kernel_call_count', 0):,}\n")
                        
                        # Track for duplicate detection
                        counter_signature = (
                            counters.get('nf4_lookup_count', 0),
                            counters.get('scaling_factor_count', 0),
                            counters.get('memory_access_count', 0),
                            counters.get('kernel_call_count', 0)
                        )
                        if counter_signature in counter_values_check:
                            counter_values_check[counter_signature].append(batch_size)
                        else:
                            counter_values_check[counter_signature] = [batch_size]
                f.write("\n")
            
            # Check for duplicate counter values - only if instrumentation available
            if has_instrumentation:
                duplicates_found = False
                for signature, batch_sizes in counter_values_check.items():
                    if len(batch_sizes) > 1 and any(x > 0 for x in signature):
                        duplicates_found = True
                        f.write(f"CRITICAL ISSUE: Identical counter values for batches {batch_sizes}\n")
                        f.write(f"   Values: NF4={signature[0]:,}, Scaling={signature[1]:,}, Memory={signature[2]:,}, Kernels={signature[3]:,}\n")
                        f.write(f"   This indicates counter reset failure between batch runs!\n\n")
                
                if duplicates_found:
                    f.write("COUNTER RESET BUG DETECTED!\n")
                    f.write("   The profiler is not properly resetting counters between batch sizes.\n")
                    f.write("   All counter data in this report may be cumulative rather than per-batch.\n")
                    f.write("   Recommendation: Fix counter reset logic and re-run profiling.\n\n")
            
            # Advanced Performance Metrics - only if instrumentation available
            if has_instrumentation:
                f.write("ADVANCED PERFORMANCE METRICS:\n")
                f.write("-" * 30 + "\n")
                
                for batch_size, result in results.items():
                    f.write(f"\nBatch {batch_size} Advanced Analysis:\n")
                    
                    if 'operation_counters' in result:
                        counters = result['operation_counters']
                        if counters.get('instrumentation_available', False):
                            # Branch divergence analysis
                            divergence_rate = counters.get('divergence_rate', 0)
                            f.write(f"  Branch divergence rate: {divergence_rate:.1f}%\n")
                            if divergence_rate > 10:
                                f.write(f"    HIGH DIVERGENCE - Constant memory optimization will help!\n")
                                f.write(f"    Expected improvement: {divergence_rate:.0f}% -> 0% divergence\n")
                            
                            # Memory coalescing analysis
                            coalescing_eff = counters.get('coalescing_efficiency', 0)
                            f.write(f"  Memory coalescing efficiency: {coalescing_eff:.1f}%\n")
                            if coalescing_eff < 70:
                                f.write(f"    POOR COALESCING - Vectorized access patterns needed!\n")
                                f.write(f"    Expected improvement: {coalescing_eff:.0f}% -> 95%+ efficiency\n")
                            
                            # Cache efficiency analysis
                            accesses_per_line = counters.get('accesses_per_cache_line', 0)
                            f.write(f"  Cache efficiency: {accesses_per_line:.1f} accesses per cache line\n")
                            if accesses_per_line < 4:
                                f.write(f"    INEFFICIENT CACHE USAGE - Shared memory will help!\n")
                                f.write(f"    Expected improvement: {accesses_per_line:.1f} -> 16+ accesses/line\n")
                            
                            # Memory bandwidth analysis
                            pytorch_data = result.get('pytorch_profiling', {})
                            if pytorch_data.get('bnb_total_time_us', 0) > 0:
                                bandwidth_metrics = self.calculate_memory_bandwidth_efficiency(
                                    counters,
                                    pytorch_data['bnb_total_time_us']
                                )
                                if bandwidth_metrics:
                                    f.write(f"  Memory bandwidth: {bandwidth_metrics['bandwidth_gb_s']:.1f} GB/s\n")
                                    f.write(f"  Bandwidth efficiency: {bandwidth_metrics['bandwidth_efficiency']:.1f}%\n")
                                    f.write(f"  Total data transferred: {bandwidth_metrics['total_gb_transferred']:.2f} GB\n")
                                    
                                    if bandwidth_metrics['bandwidth_efficiency'] < 30:
                                        f.write(f"    LOW BANDWIDTH UTILIZATION!\n")
                                        f.write(f"    Theoretical maximum: {bandwidth_metrics['theoretical_bandwidth_gb_s']:.0f} GB/s\n")
                                        f.write(f"    Current utilization: {bandwidth_metrics['bandwidth_efficiency']:.1f}%\n")
                                        f.write(f"    Target after optimization: 80%+ utilization\n")
            
            # Latency scaling analysis
            f.write("\nLATENCY SCALING ANALYSIS:\n")
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
                    
                    f.write(f"Batch {prev_batch} -> {batch_size} (expected {batch_scaling:.1f}x scaling):\n")
                    f.write(f"  Throughput scaling: {throughput_scaling:.2f}x\n")
                    f.write(f"  Latency scaling: {latency_scaling:.2f}x\n")
                    
                    # Counter scaling analysis - only if instrumentation available
                    if has_instrumentation and 'operation_counters' in current_result and 'operation_counters' in prev_result:
                        curr_counters = current_result['operation_counters']
                        prev_counters = prev_result['operation_counters']
                        
                        if curr_counters.get('instrumentation_available', False) and prev_counters.get('instrumentation_available', False):
                            # Check divergence scaling
                            curr_div = curr_counters.get('divergence_rate', 0)
                            prev_div = prev_counters.get('divergence_rate', 0)
                            f.write(f"  Divergence rate: {prev_div:.1f}% -> {curr_div:.1f}%\n")
                            
                            # Check efficiency scaling
                            curr_coal = curr_counters.get('coalescing_efficiency', 0)
                            prev_coal = prev_counters.get('coalescing_efficiency', 0)
                            f.write(f"  Coalescing efficiency: {prev_coal:.1f}% -> {curr_coal:.1f}%\n")
            
            # Detailed operation analysis
            f.write("\nDETAILED OPERATION ANALYSIS:\n")
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
                        f.write(f"     Avg time: {op['avg_time_us']:.1f} µs\n\n")
                else:
                    f.write("No BitsAndBytes operations detected in PyTorch profiler\n\n")
                
                # Counter breakdown with advanced metrics - only if instrumentation available
                if has_instrumentation and 'operation_counters' in result:
                    counters = result['operation_counters']
                    if counters.get('instrumentation_available', False):
                        f.write("Operation Counter Breakdown:\n")
                        f.write(f"  NF4 lookups: {counters.get('nf4_lookup_count', 0):,}\n")
                        f.write(f"  Scaling factor accesses: {counters.get('scaling_factor_count', 0):,}\n")
                        f.write(f"  Memory accesses: {counters.get('memory_access_count', 0):,}\n")
                        f.write(f"  Kernel calls: {counters.get('kernel_call_count', 0):,}\n")
                        f.write(f"  Total operations: {counters.get('total_operations', 0):,}\n")
                        
                        f.write(f"\nAdvanced Counters:\n")
                        f.write(f"  Warp divergence events: {counters.get('warp_divergence_count', 0):,}\n")
                        f.write(f"  Cache line loads: {counters.get('cache_line_loads', 0):,}\n")
                        f.write(f"  Coalesced loads: {counters.get('coalesced_loads', 0):,}\n")
                        f.write(f"  Scattered loads: {counters.get('scattered_loads', 0):,}\n")
                        f.write(f"  Total bytes loaded: {counters.get('bytes_loaded', 0):,}\n")
                        
                        # Calculate ratios
                        kernel_calls = counters.get('kernel_call_count', 0)
                        if kernel_calls > 0:
                            f.write(f"\nAverages per kernel call:\n")
                            f.write(f"  NF4 lookups/kernel: {counters.get('nf4_lookup_count', 0) / kernel_calls:,.1f}\n")
                            f.write(f"  Scaling accesses/kernel: {counters.get('scaling_factor_count', 0) / kernel_calls:,.1f}\n")
                            f.write(f"  Memory accesses/kernel: {counters.get('memory_access_count', 0) / kernel_calls:,.1f}\n")
                            f.write(f"  Bytes/kernel: {counters.get('bytes_loaded', 0) / kernel_calls:,.0f}\n")
                            
                            # Unusual ratio detection
                            nf4_to_scaling = counters.get('nf4_lookup_count', 0) / max(counters.get('scaling_factor_count', 1), 1)
                            f.write(f"\nRatio Analysis:\n")
                            f.write(f"  NF4:Scaling ratio: {nf4_to_scaling:.1f}:1\n")
                            if abs(nf4_to_scaling - 16) > 2:
                                f.write(f"    Unusual ratio! Expected ~16:1 for standard NF4\n")
                                f.write(f"    May indicate non-standard quantization or measurement issue\n")
            
            # Optimization recommendations
            f.write("\nOPTIMIZATION ANALYSIS:\n")
            f.write("-" * 25 + "\n")
            
            if has_instrumentation:
                f.write("Based on operation counters:\n\n")
                
                f.write("1. NF4 Lookup Optimization:\n")
                f.write("   - Current: Tree-based lookup in registers\n")
                f.write("   - Target: Constant memory lookup table\n")
                f.write("   - Expected: Reduced register pressure\n\n")
                
                f.write("2. Scaling Factor Optimization:\n")
                f.write("   - Current: Global memory access per element\n")
                f.write("   - Target: Shared memory cooperative loading\n")
                f.write("   - Expected: Better memory bandwidth utilization\n\n")
                
                f.write("3. Memory Access Optimization:\n")
                f.write("   - Current: Individual element access\n")
                f.write("   - Target: Vectorized float4 access patterns\n")
                f.write("   - Expected: Improved memory coalescing\n\n")
                
                # Add quantitative optimization targets
                f.write("QUANTITATIVE OPTIMIZATION TARGETS:\n")
                
                # Find worst metrics across all batches
                worst_divergence = 0
                worst_coalescing = 100
                worst_bandwidth = 100
                
                for result in results.values():
                    if 'operation_counters' in result:
                        counters = result['operation_counters']
                        if counters.get('instrumentation_available', False):
                            worst_divergence = max(worst_divergence, counters.get('divergence_rate', 0))
                            worst_coalescing = min(worst_coalescing, counters.get('coalescing_efficiency', 100))
                            
                            pytorch_data = result.get('pytorch_profiling', {})
                            if pytorch_data.get('bnb_total_time_us', 0) > 0:
                                bandwidth_metrics = self.calculate_memory_bandwidth_efficiency(
                                    counters,
                                    pytorch_data['bnb_total_time_us']
                                )
                                if bandwidth_metrics:
                                    worst_bandwidth = min(worst_bandwidth, bandwidth_metrics.get('bandwidth_efficiency', 100))
                
                f.write(f"  Branch divergence: {worst_divergence:.1f}% -> 0%\n")
                f.write(f"  Memory coalescing: {worst_coalescing:.1f}% -> 95%+\n")
                f.write(f"  Bandwidth utilization: {worst_bandwidth:.1f}% -> 80%+\n")
                
                # Calculate potential speedup
                divergence_speedup = 1 + (worst_divergence / 100) * 15  # Up to 16x for worst case
                coalescing_speedup = 95 / max(worst_coalescing, 1)
                bandwidth_speedup = 80 / max(worst_bandwidth, 1)
                
                combined_speedup = (divergence_speedup + coalescing_speedup + bandwidth_speedup) / 3
                
                f.write(f"\n  Estimated kernel speedup: {combined_speedup:.1f}x\n")
                f.write(f"  Estimated end-to-end improvement: {combined_speedup * 0.3:.0f}% faster\n")
            else:
                f.write("Based on PyTorch profiler analysis:\n\n")
                
                # Analyze PyTorch profiler results for recommendations
                avg_bnb_overhead = sum(
                    result.get('pytorch_profiling', {}).get('bnb_time_percentage', 0) 
                    for result in results.values()
                ) / len(results)
                
                f.write(f"Average BnB overhead: {avg_bnb_overhead:.1f}%\n\n")
                
                if avg_bnb_overhead > 20:
                    f.write("1. High BnB overhead suggests optimization opportunities:\n")
                    f.write("   - NF4 lookup table optimization\n")
                    f.write("   - Memory access pattern improvements\n")
                    f.write("   - Shared memory utilization\n\n")
                    f.write("2. To get detailed optimization insights:\n")
                    f.write("   - Recompile BitsAndBytes with BNB_ENABLE_INSTRUMENTATION=1\n")
                    f.write("   - Re-run profiling to get operation counter data\n\n")
                else:
                    f.write("1. BnB overhead is reasonable\n")
                    f.write("2. Focus on general PyTorch optimizations\n")
                    f.write("3. Consider enabling instrumentation for detailed analysis\n\n")
            
            f.write("\nVALIDATION METHODOLOGY:\n")
            f.write("- Use PyTorch profiler to measure end-to-end timing\n")
            f.write("- Compare before/after optimization implementations\n")
            f.write("- Focus on throughput (tokens/sec) improvement\n")
            if has_instrumentation:
                f.write("- Use counters to verify operations are being tracked\n")
                f.write("- Monitor divergence and coalescing efficiency improvements\n")
            else:
                f.write("- Monitor BnB overhead percentage reduction\n")
                f.write("- Enable instrumentation for detailed kernel validation\n")
        
        print(f"Profiling report saved: {report_path}")
        return str(report_path)
    
    def run_profiling(self, model_name: str, batch_sizes: List[int] = None, tag: str = "") -> Dict[int, Dict[str, Any]]:
        """Run profiling across multiple batch sizes"""
        
        if batch_sizes is None:
            batch_sizes = [2, 4]  # Start with batch_size >= 2 for blockwise operations
        
        print(f"[Starting] Starting BitsAndBytes profiling")
        print(f"Model: {model_name}")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Counters: {'Enabled' if self.use_counters else 'Disabled'}")
        if tag:
            print(f"Configuration tag: {tag}")
        
        # Warn about batch size for blockwise operations
        if min(batch_sizes) < 2:
            print("[WARNING] Warning: Batch sizes < 2 may not trigger NF4 blockwise dequantization")
        
        # Load model ONCE and reuse for all batch sizes
        if not self.load_model_with_bnb(model_name):
            return {}
        
        results = {}
        
        for i, batch_size in enumerate(batch_sizes):
            try:
                print(f"\n[Profiling] Profiling batch size {batch_size} (run {i+1}/{len(batch_sizes)})...")
                
                # Clear GPU state between runs
                if i > 0:  # Not needed for first run
                    print("   [Cleanup] Clearing GPU state between runs...")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    time.sleep(1)  # Brief pause
                
                result = self.profile_with_counters(batch_size)
                result['model_name'] = model_name  # Add model name to result
                
                results[batch_size] = result
                
                # Print summary with validation
                # Print summary
                tokens_sec = result['tokens_per_sec']
                if self.use_counters and 'operation_counters' in result:
                    counter_data = result['operation_counters']
                    if counter_data.get('instrumentation_available', False):
                        total_ops = counter_data.get('total_operations', 0)
                        kernel_calls = counter_data.get('kernel_call_count', 0)
                        print(f"Batch {batch_size}: {tokens_sec:.1f} tok/s, {total_ops:,} ops, {kernel_calls} kernels")
                    else:
                        print(f"Batch {batch_size}: {tokens_sec:.1f} tok/s, instrumentation not available")
                else:
                    pytorch_data = result.get('pytorch_profiling', {})
                    bnb_pct = pytorch_data.get('bnb_time_percentage', 0)
                    print(f"Batch {batch_size}: {tokens_sec:.1f} tok/s, {bnb_pct:.1f}% BnB overhead")
                
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
                            print(f"   [CRITICAL] SUSPICIOUS: Identical PyTorch kernel counts ({current_kernel_count})")
                            print(f"       This suggests PyTorch profiler may also have accumulation issues")
                
                # Brief pause between runs to ensure separation
                if i < len(batch_sizes) - 1:
                    time.sleep(2)
                
            except Exception as e:
                print(f"[ERROR] Batch {batch_size} failed: {e}")
                import traceback
                traceback.print_exc()
                results[batch_size] = {'error': str(e)}
        
        # Generate report with enhanced diagnostics
        if results:
            report_path = self.generate_profiling_report(results, tag)
            print(f"\n[Complete] Profiling complete!")
            print(f"[Report] Report: {report_path}")
            
            # Immediate cross-validation summary
            print(f"\n[Validation] Quick Validation Summary:")
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
                print(f"  [CRITICAL] PyTorch profiler shows identical kernel counts - profiler accumulation bug!")
            elif len(set(pytorch_kernel_counts)) > 1:
                print(f"  [OK] PyTorch profiler shows proper scaling")
                
            if len(set(counter_kernel_counts)) == 1 and counter_kernel_counts[0] > 0:
                print(f"  [CRITICAL] Counter data shows identical kernel counts - counter reset bug!")
            elif len(set(counter_kernel_counts)) > 1:
                print(f"  [OK] Counter data shows proper scaling")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="BitsAndBytes Profiler with Response Generation")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--batch_sizes", default="2,4", help="Comma-separated batch sizes")
    parser.add_argument("--output_dir", default="bnb_counter_profiling", help="Output directory")
    parser.add_argument("--counters", action="store_true", help="Enable operation counters")
    parser.add_argument("--tag", default="", help="Configuration tag for output files")
    parser.add_argument("--save_responses", action="store_true", help="Save generated responses to JSON")  # NEW LINE
    
    args = parser.parse_args()
    
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(',')]
    
    if min(batch_sizes) < 2:
        print("[WARNING] Warning: Batch sizes < 2 may not trigger NF4 blockwise dequantization kernels")
    
    try:
        profiler = BnBCounterProfiler(args.output_dir, use_counters=args.counters)
        
        # MODIFIED: Choose method based on save_responses flag
        if args.save_responses:
            results = profiler.run_profiling_with_responses(args.model, batch_sizes, args.tag, save_responses=True)
        else:
            results = profiler.run_profiling(args.model, batch_sizes, args.tag)
        
        if not results:
            print("[ERROR] No profiling results generated")
            return
        
        print("\n" + "=" * 50)
        print("[Complete] PROFILING COMPLETE!")
        print("=" * 50)
        
        print("\n[Results] Performance Results:")
        for batch_size, result in results.items():
            if 'error' not in result:
                tokens_sec = result.get('tokens_per_sec', 0)
                bnb_pct = result.get('pytorch_profiling', {}).get('bnb_time_percentage', 0)
                response_count = result.get('response_count', 0)
                print(f"  Batch {batch_size}: {tokens_sec:.1f} tok/s, {bnb_pct:.1f}% BnB overhead, {response_count} responses")
        
        if args.save_responses:
            print(f"\n[Saved] Generated responses saved to JSON files in: {args.output_dir}/")
            print("   - Individual batch files: responses_batch_*.json")
            print("   - Consolidated file: all_responses_*.json")
        
    except KeyboardInterrupt:
        print("\n[Interrupted] Profiling interrupted")
    except Exception as e:
        print(f"[ERROR] Profiling failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()