#!/usr/bin/env python3
"""
simple_kernel_profiler.py

Robust kernel profiler using only stable PyTorch profiler APIs.
This version focuses on getting actionable insights without API compatibility issues.

Usage:
    python simple_kernel_profiler.py --batch_sizes 1,4,16
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import torch
import torch.profiler
from datasets import load_dataset

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model_loader import ModelLoader

class SimpleKernelProfiler:
    """
    Simplified, robust kernel profiler using stable PyTorch APIs
    """
    
    def __init__(self, output_dir: str = "simple_kernel_profiling"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set reproducible seed
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        print("🎲 Set seed to 42 for reproducible results")
        
        print("🔄 Loading model for kernel profiling...")
        self.tokenizer, self.model = ModelLoader.load_model()
        print("✅ Model loaded")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_gsm8k_samples(self, num_samples: int = 8) -> List[str]:
        """Load GSM8K samples for profiling"""
        try:
            dataset = load_dataset("gsm8k", "main", split="test")
            samples = dataset.select(range(min(num_samples, len(dataset))))
            
            prompts = []
            for sample in samples:
                prompt = f"Solve this math problem step by step:\n\n{sample['question']}\n\nAnswer:"
                prompts.append(prompt)
            
            print(f"✅ Loaded {len(prompts)} GSM8K prompts")
            return prompts
            
        except Exception as e:
            print(f"⚠️ Failed to load GSM8K, using fallback prompts: {e}")
            return [
                "What is 25 + 37?",
                "Calculate 12 × 8 + 15",
                "If 5 apples cost $2, how much do 20 apples cost?",
                "Find the area of a rectangle with length 8 and width 6."
            ][:num_samples]
    
    def profile_single_batch(self, batch_size: int, max_tokens: int = 150) -> Dict[str, Any]:
        """Profile a single batch size using stable PyTorch profiler APIs"""
        
        print(f"🔍 Profiling batch size {batch_size}...")
        
        # Set seed for reproducible results
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        # Load test prompts
        prompts = self.load_gsm8k_samples(batch_size)
        
        # Prepare batch
        inputs = self.tokenizer(
            prompts,
            padding='longest',
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Warmup run with same settings
        print("  Warmup...")
        with torch.no_grad():
            _ = self.model.generate(
                **inputs, 
                max_new_tokens=10, 
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        
        # Clear memory stats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Main profiling run
        print("  Profiling...")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,  # Disable to avoid overhead
            with_flops=True,
            with_modules=False,  # Disable for stability
        ) as prof:
            
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # Use deterministic generation
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.0  # Remove invalid generation flags
                )
            
            torch.cuda.synchronize()
            end_time = time.time()
        
        # Collect memory stats
        memory_stats = {
            'peak_memory_mb': torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            'current_memory_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
        }
        
        # Export trace
        trace_file = self.output_dir / f"batch_{batch_size}_trace.json"
        try:
            prof.export_chrome_trace(str(trace_file))
            print(f"  ✅ Trace saved: {trace_file}")
        except Exception as e:
            print(f"  ⚠️ Trace export failed: {e}")
            trace_file = "trace_export_failed"
        
        # Analyze profiling results using stable APIs
        analysis = self._analyze_profiling_results(prof, batch_size)
        
        return {
            'batch_size': batch_size,
            'total_time_ms': (end_time - start_time) * 1000,
            'memory_stats': memory_stats,
            'analysis': analysis,
            'trace_file': str(trace_file)
        }
    
    def _analyze_profiling_results(self, prof: torch.profiler.profile, batch_size: int) -> Dict[str, Any]:
        """Analyze profiling results using only stable APIs"""
        
        # Get key averages - this is the most stable API
        key_averages = prof.key_averages()
        
        # Extract top CUDA operations
        cuda_ops = []
        cpu_ops = []
        
        for event in key_averages:
            # Use device_time instead of deprecated cuda_time
            device_time = getattr(event, 'device_time', 0)
            device_time_total = getattr(event, 'device_time_total', 0)
            cpu_time = getattr(event, 'cpu_time', 0)
            cpu_time_total = getattr(event, 'cpu_time_total', 0)
            
            event_data = {
                'name': event.key,
                'cpu_time_us': cpu_time,
                'device_time_us': device_time,  # Use device_time instead of cuda_time
                'cpu_time_total': cpu_time_total,
                'device_time_total': device_time_total,  # Use device_time_total
                'count': event.count,
                'input_shapes': getattr(event, 'input_shapes', []),
                'flops': getattr(event, 'flops', 0) if hasattr(event, 'flops') else 0
            }
            
            # Classify as CUDA or CPU operation
            if device_time > 0 or 'cuda' in event.key.lower() or 'kernel' in event.key.lower():
                event_data['operation_type'] = self._classify_operation(event.key)
                cuda_ops.append(event_data)
            else:
                cpu_ops.append(event_data)
        
        # Sort by device time (CUDA time)
        cuda_ops.sort(key=lambda x: x.get('device_time_total', 0), reverse=True)
        cpu_ops.sort(key=lambda x: x.get('cpu_time_total', 0), reverse=True)
        
        # Get formatted table - try both old and new API
        try:
            # Try new API first
            table_cuda = prof.key_averages().table(sort_by="device_time_total", row_limit=15)
        except:
            try:
                # Fallback to old API
                table_cuda = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
            except Exception as e:
                print(f"⚠️ Warning: Could not generate CUDA table: {e}")
                table_cuda = "CUDA table generation failed"
        
        try:
            table_cpu = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
        except Exception as e:
            print(f"⚠️ Warning: Could not generate CPU table: {e}")
            table_cpu = "CPU table generation failed"
        
        # Generate insights with safe division
        insights = self._generate_insights(cuda_ops, cpu_ops, batch_size)
        
        # Safe summation
        total_cuda_time = sum(op.get('device_time_total', 0) for op in cuda_ops)
        total_cpu_time = sum(op.get('cpu_time_total', 0) for op in cpu_ops)
        
        return {
            'top_cuda_ops': cuda_ops[:15],
            'top_cpu_ops': cpu_ops[:10],
            'table_cuda': table_cuda,
            'table_cpu': table_cpu,
            'insights': insights,
            'total_cuda_time_us': total_cuda_time,
            'total_cpu_time_us': total_cpu_time
        }
    
    def _classify_operation(self, op_name: str) -> str:
        """Classify operation type for optimization targeting"""
        op_lower = op_name.lower()
        
        # Matrix operations (usually most expensive)
        if any(x in op_lower for x in ['gemm', 'sgemm', 'hgemm', 'matmul', 'mm', 'bmm']):
            return 'matrix_multiply'
        
        # Attention operations
        elif any(x in op_lower for x in ['attention', 'softmax', 'scaled_dot_product']):
            return 'attention'
        
        # Memory operations
        elif any(x in op_lower for x in ['copy', 'clone', 'cat', 'stack', 'view', 'reshape']):
            return 'memory_reshape'
        
        # Normalization
        elif any(x in op_lower for x in ['norm', 'layer_norm', 'rms_norm', 'batch_norm']):
            return 'normalization'
        
        # Activation functions
        elif any(x in op_lower for x in ['gelu', 'relu', 'silu', 'swish', 'tanh', 'sigmoid']):
            return 'activation'
        
        # Embedding operations
        elif any(x in op_lower for x in ['embedding', 'embed']):
            return 'embedding'
        
        # Element-wise operations
        elif any(x in op_lower for x in ['add', 'mul', 'div', 'sub', 'elementwise']):
            return 'elementwise'
        
        else:
            return 'other'
    
    def _generate_insights(self, cuda_ops: List[Dict], cpu_ops: List[Dict], batch_size: int) -> List[str]:
        """Generate actionable optimization insights with safe division"""
        insights = []
        
        if not cuda_ops:
            insights.append("⚠️ No CUDA operations detected - model may be running on CPU")
            return insights
        
        # Total time analysis with safe division
        total_cuda_time = sum(op.get('device_time_total', 0) for op in cuda_ops)
        total_cpu_time = sum(op.get('cpu_time_total', 0) for op in cpu_ops)
        
        if total_cuda_time == 0:
            insights.append("⚠️ No CUDA time recorded - profiling may have failed")
            return insights
        
        if total_cpu_time > total_cuda_time and total_cuda_time > 0:
            cpu_ratio = total_cpu_time / total_cuda_time if total_cuda_time > 0 else float('inf')
            insights.append(f"🔴 CPU overhead is high ({total_cpu_time/1000:.1f}ms CPU vs {total_cuda_time/1000:.1f}ms CUDA, ratio: {cpu_ratio:.1f}x)")
            insights.append("   → Consider CUDA graphs or larger batch sizes")
        
        # Top operation analysis
        if cuda_ops:
            top_op = cuda_ops[0]
            op_type = top_op.get('operation_type', 'unknown')
            op_time_ms = top_op.get('device_time_total', 0) / 1000
            op_percentage = (top_op.get('device_time_total', 0) / total_cuda_time) * 100 if total_cuda_time > 0 else 0
            
            insights.append(f"🎯 Top bottleneck: {top_op['name'][:50]}...")
            insights.append(f"   Type: {op_type}, Time: {op_time_ms:.1f}ms ({op_percentage:.1f}% of GPU time)")
            
            # Specific recommendations
            if op_type == 'matrix_multiply':
                insights.append("   → Optimization: Use mixed precision (FP16), optimize tensor shapes")
            elif op_type == 'attention':
                insights.append("   → Optimization: Consider Flash Attention, fused attention kernels")
            elif op_type == 'memory_reshape':
                insights.append("   → Optimization: Reduce tensor copying, use in-place operations")
            elif op_type == 'normalization':
                insights.append("   → Optimization: Use fused normalization kernels")
        
        # Memory efficiency with safe division
        top_5_ops = cuda_ops[:5]
        top_5_time = sum(op.get('device_time_total', 0) for op in top_5_ops)
        top_5_percentage = (top_5_time / total_cuda_time) * 100 if total_cuda_time > 0 else 0
        
        insights.append(f"📊 Top 5 operations account for {top_5_percentage:.1f}% of GPU time")
        
        if top_5_percentage > 80:
            insights.append("   → High concentration - focus optimization on these operations")
        elif top_5_percentage > 0:
            insights.append("   → Distributed workload - consider overall architecture optimization")
        
        # Batch size specific insights
        if batch_size == 1:
            insights.append("⚡ Small batch size may have high overhead - test larger batches")
        elif batch_size >= 16:
            insights.append("🔋 Large batch size - monitor memory usage and consider memory optimization")
        
        return insights
    
    def run_batch_comparison(self, batch_sizes: List[int]) -> Dict[str, Any]:
        """Run profiling across multiple batch sizes"""
        
        print(f"🚀 Running batch size comparison: {batch_sizes}")
        
        results = {}
        
        for batch_size in batch_sizes:
            try:
                result = self.profile_single_batch(batch_size)
                results[batch_size] = result
                
                # Brief summary
                time_ms = result['total_time_ms']
                time_per_sample = time_ms / batch_size
                memory_mb = result['memory_stats']['peak_memory_mb']
                
                print(f"✅ Batch {batch_size}: {time_ms:.0f}ms total, {time_per_sample:.0f}ms/sample, {memory_mb:.0f}MB")
                
                # Pause between runs
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Batch size {batch_size} failed: {e}")
                results[batch_size] = {'error': str(e)}
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive profiling report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"kernel_profiling_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("KERNEL-LEVEL PROFILING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Batch sizes: {list(results.keys())}\n\n")
            
            # Performance summary
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 25 + "\n")
            
            for batch_size, result in results.items():
                if 'error' in result:
                    f.write(f"Batch {batch_size:2d}: ERROR - {result['error']}\n")
                    continue
                
                time_ms = result['total_time_ms']
                time_per_sample = time_ms / batch_size
                memory_mb = result['memory_stats']['peak_memory_mb']
                
                f.write(f"Batch {batch_size:2d}: {time_ms:7.0f}ms total, "
                       f"{time_per_sample:6.0f}ms/sample, {memory_mb:6.0f}MB peak\n")
            
            f.write("\n")
            
            # Detailed analysis for each batch size
            for batch_size, result in results.items():
                if 'error' in result:
                    continue
                    
                f.write(f"DETAILED ANALYSIS - BATCH SIZE {batch_size}:\n")
                f.write("-" * 40 + "\n")
                
                analysis = result['analysis']
                
                # Top CUDA operations
                f.write("Top CUDA Operations:\n")
                for i, op in enumerate(analysis['top_cuda_ops'][:10], 1):
                    op_name = op['name'][:50]
                    op_type = op.get('operation_type', 'unknown')
                    device_time_ms = op.get('device_time_total', 0) / 1000  # Use device_time_total
                    count = op.get('count', 0)
                    
                    f.write(f"{i:2d}. {op_name}\n")
                    f.write(f"    Type: {op_type}, Time: {device_time_ms:.2f}ms, Calls: {count}\n\n")
                
                # Insights
                f.write("Optimization Insights:\n")
                for insight in analysis['insights']:
                    f.write(f"{insight}\n")
                
                f.write("\n" + "="*50 + "\n\n")
            
            # Overall recommendations
            f.write("OVERALL RECOMMENDATIONS:\n")
            f.write("-" * 30 + "\n")
            f.write("1. Focus on the top 3-5 operations for maximum impact\n")
            f.write("2. Use Chrome tracing (chrome://tracing) to visualize execution timeline\n")
            f.write("3. Test different batch sizes to find optimal throughput/memory balance\n")
            f.write("4. Consider model compilation with torch.compile() for kernel fusion\n")
            f.write("5. Monitor CPU vs GPU time ratio - high CPU time indicates overhead\n\n")
            
            f.write(f"Trace files: {self.output_dir}/*_trace.json\n")
            f.write("Load these in Chrome's tracing tool for detailed timeline analysis.\n")
        
        print(f"📊 Report saved to: {report_path}")
        return str(report_path)


def main():
    parser = argparse.ArgumentParser(description="Simple, robust kernel profiler")
    parser.add_argument("--batch_sizes", default="1,4,8", 
                       help="Comma-separated batch sizes to test")
    parser.add_argument("--max_tokens", type=int, default=150,
                       help="Maximum tokens to generate per sample")
    parser.add_argument("--output_dir", default="simple_kernel_profiling",
                       help="Output directory")
    
    args = parser.parse_args()
    
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(',')]
    
    print("🔬 SIMPLE KERNEL PROFILER")
    print("=" * 35)
    print(f"Batch sizes: {batch_sizes}")
    print(f"Max tokens: {args.max_tokens}")
    print()
    
    profiler = SimpleKernelProfiler(args.output_dir)
    
    try:
        # Run profiling
        results = profiler.run_batch_comparison(batch_sizes)
        
        # Generate report
        report_path = profiler.generate_report(results)
        
        print("\n" + "=" * 50)
        print("✅ PROFILING COMPLETE!")
        print("=" * 50)
        print(f"📊 Report: {report_path}")
        print(f"📈 Trace files: {profiler.output_dir}/*_trace.json")
        print("\n💡 Next steps:")
        print("1. Read the text report for optimization insights")
        print("2. Load trace files in Chrome (chrome://tracing) for visual analysis")
        print("3. Focus on the top operations for maximum optimization impact")
        
    except KeyboardInterrupt:
        print("\n⏹️ Profiling interrupted")
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()