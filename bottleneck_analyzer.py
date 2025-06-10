#!/usr/bin/env python3
"""
bottleneck_analyzer.py

Main script for identifying performance bottlenecks and optimization opportunities
in your Qwen2 model. Run this to find the operations that need optimization.

Usage:
    python bottleneck_analyzer.py --mode establish_baseline
    python bottleneck_analyzer.py --mode analyze_bottlenecks
    python bottleneck_analyzer.py --mode test_optimization --optimization_name "flash_attention"
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List
import torch

# Import your modules
from model_loader import load_model_for_optimization, ModelLoader
import logging_config

logger = logging.getLogger(__name__)


class BottleneckAnalyzer:
    """
    Main analyzer class for finding and optimizing performance bottlenecks
    """
    
    def __init__(self, output_dir: str = "optimization_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Test configurations
        self.test_prompts = [
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate fibonacci numbers.",
            "What are the main causes of climate change?",
            "Describe the process of machine learning model training.",
            "How does neural network backpropagation work?"
        ]
        
        # Load model once for all tests
        logger.info("Loading model for optimization analysis...")
        self.tokenizer, self.model = load_model_for_optimization()
        logger.info("✅ Model loaded successfully")
    
    def run_inference_suite(self, num_runs: int = 5, max_new_tokens: int = 100) -> None:
        """
        Run a comprehensive inference suite to collect profiling data
        """
        logger.info(f"🚀 Running inference suite: {num_runs} runs x {len(self.test_prompts)} prompts")
        
        # Warmup runs
        logger.info("Performing warmup...")
        for prompt in self.test_prompts[:2]:  # Use first 2 prompts for warmup
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = self.model.generate(**inputs, max_new_tokens=50, do_sample=False)
            torch.cuda.synchronize()
        
        logger.info("Warmup complete. Starting profiled runs...")
        
        # Reset stats after warmup
        ModelLoader.reset_profiling_stats()
        
        # Profiled inference runs
        for run in range(num_runs):
            logger.info(f"Run {run + 1}/{num_runs}")
            
            for i, prompt in enumerate(self.test_prompts):
                inputs = self.tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, 
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # Deterministic for consistent profiling
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Synchronize to ensure timing accuracy
                torch.cuda.synchronize()
                
                logger.debug(f"  Completed prompt {i+1}/{len(self.test_prompts)}")
        
        logger.info("✅ Inference suite completed")
    
    def establish_baseline(self, baseline_name: str = "baseline") -> Dict[str, Any]:
        """
        Establish performance baseline by running inference and analyzing results
        """
        logger.info(f"📊 Establishing baseline: {baseline_name}")
        
        # Run inference to collect data
        self.run_inference_suite()
        
        # Establish baseline
        baseline_results = ModelLoader.establish_baseline(baseline_name)
        
        # Save baseline
        baseline_file = self.output_dir / f"baseline_{baseline_name}_{int(time.time())}.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline_results, f, indent=2, default=str)
        
        logger.info(f"💾 Baseline saved to: {baseline_file}")
        
        # Print summary
        self._print_bottleneck_summary(baseline_results)
        
        return baseline_results
    
    def analyze_current_performance(self) -> Dict[str, Any]:
        """
        Analyze current model performance without establishing baseline
        """
        logger.info("🔍 Analyzing current performance...")
        
        # Run inference to collect data
        self.run_inference_suite()
        
        # Get analysis
        results = ModelLoader.get_bottleneck_analysis()
        
        # Save results
        results_file = self.output_dir / f"analysis_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Analysis saved to: {results_file}")
        
        # Print summary
        self._print_bottleneck_summary(results)
        
        return results
    
    def test_optimization(self, optimization_name: str) -> Dict[str, Any]:
        """
        Test an optimization against the established baseline
        """
        logger.info(f"🧪 Testing optimization: {optimization_name}")
        
        # Run inference with the optimization
        self.run_inference_suite()
        
        # Compare with baseline
        comparison = ModelLoader.test_optimization(optimization_name)
        
        # Save comparison
        comparison_file = self.output_dir / f"optimization_{optimization_name}_{int(time.time())}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        logger.info(f"💾 Optimization results saved to: {comparison_file}")
        
        # Print comparison summary
        self._print_optimization_summary(comparison)
        
        return comparison
    
    def _print_bottleneck_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of bottleneck analysis"""
        print("\n" + "="*80)
        print("🎯 BOTTLENECK ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"📈 Total Execution Time: {results['total_execution_time_ms']:.2f} ms")
        print(f"🔢 Operations Profiled: {results['total_operations_profiled']}")
        print(f"📞 Total Function Calls: {results['total_function_calls']}")
        
        print("\n🏆 TOP 10 BOTTLENECKS (Optimization Targets):")
        print("-" * 80)
        print(f"{'Rank':<4} {'Operation':<35} {'Time (ms)':<10} {'%':<6} {'Calls':<8} {'Priority':<8}")
        print("-" * 80)
        
        for i, bottleneck in enumerate(results['bottlenecks'][:10], 1):
            print(f"{i:<4} {bottleneck['operation'][:34]:<35} "
                  f"{bottleneck['total_time_ms']:<10.2f} "
                  f"{bottleneck['contribution_percent']:<6.1f} "
                  f"{bottleneck['call_count']:<8} "
                  f"{bottleneck['optimization_potential']:<8}")
        
        # Layer analysis
        if results.get('layer_analysis', {}).get('outlier_layers'):
            print("\n⚠️  PROBLEMATIC LAYERS:")
            print("-" * 50)
            for outlier in results['layer_analysis']['outlier_layers']:
                print(f"  {outlier['layer']}: {outlier['total_time_ms']:.2f} ms "
                      f"(+{outlier['deviation_from_avg']:.2f} ms from avg)")
        
        # Operation type breakdown
        if results.get('operation_type_analysis'):
            print("\n📊 OPERATION TYPE BREAKDOWN:")
            print("-" * 50)
            for op_type, stats in results['operation_type_analysis'].items():
                print(f"  {op_type.title()}: {stats['contribution_percent']:.1f}% "
                      f"({stats['total_time_ms']:.2f} ms)")
        
        # Optimization recommendations
        if results.get('optimization_recommendations'):
            print("\n💡 OPTIMIZATION RECOMMENDATIONS:")
            print("-" * 50)
            for i, rec in enumerate(results['optimization_recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*80)
    
    def _print_optimization_summary(self, comparison: Dict[str, Any]):
        """Print a formatted summary of optimization comparison"""
        print("\n" + "="*80)
        print(f"🧪 OPTIMIZATION TEST RESULTS: {comparison['optimization_name']}")
        print("="*80)
        
        speedup = comparison['overall_speedup']
        time_saved = comparison['time_saved_ms']
        
        if speedup > 1.0:
            print(f"✅ IMPROVEMENT DETECTED!")
            print(f"   Overall Speedup: {speedup:.2f}x")
            print(f"   Time Saved: {time_saved:.2f} ms ({time_saved/comparison['baseline_total_time_ms']*100:.1f}%)")
        else:
            print(f"❌ PERFORMANCE REGRESSION")
            print(f"   Slowdown: {1/speedup:.2f}x")
            print(f"   Time Lost: {abs(time_saved):.2f} ms ({abs(time_saved)/comparison['baseline_total_time_ms']*100:.1f}%)")
        
        print(f"\nBaseline Total Time: {comparison['baseline_total_time_ms']:.2f} ms")
        print(f"Current Total Time:  {comparison['current_total_time_ms']:.2f} ms")
        
        # Show top operation improvements/regressions
        if comparison.get('operation_improvements'):
            improvements = sorted(
                comparison['operation_improvements'].items(),
                key=lambda x: x[1]['speedup'],
                reverse=True
            )
            
            print("\n🎯 TOP OPERATION IMPROVEMENTS:")
            print("-" * 70)
            print(f"{'Operation':<35} {'Speedup':<8} {'Time Saved (ms)':<15}")
            print("-" * 70)
            
            for op_name, stats in improvements[:5]:
                if stats['speedup'] > 1.0:
                    print(f"{op_name[:34]:<35} {stats['speedup']:<8.2f} {stats['time_saved_ms']:<15.2f}")
            
            print("\n⚠️  TOP OPERATION REGRESSIONS:")
            print("-" * 70)
            for op_name, stats in improvements[-5:]:
                if stats['speedup'] < 1.0:
                    print(f"{op_name[:34]:<35} {stats['speedup']:<8.2f} {stats['time_saved_ms']:<15.2f}")
        
        print("\n" + "="*80)
    
    def generate_optimization_report(self) -> str:
        """Generate a comprehensive optimization report"""
        logger.info("📋 Generating comprehensive optimization report...")
        
        # Analyze current performance
        results = self.analyze_current_performance()
        
        # Generate detailed report
        report_lines = [
            "QWEN2 MODEL OPTIMIZATION REPORT",
            "=" * 50,
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {self.model.config.name_or_path if hasattr(self.model, 'config') else 'Unknown'}",
            "",
            "EXECUTIVE SUMMARY:",
            f"• Total execution time: {results['total_execution_time_ms']:.2f} ms",
            f"• Operations profiled: {results['total_operations_profiled']}",
            f"• Function calls: {results['total_function_calls']:,}",
            "",
            "TOP OPTIMIZATION TARGETS:",
        ]
        
        for i, bottleneck in enumerate(results['bottlenecks'][:5], 1):
            report_lines.append(
                f"{i}. {bottleneck['operation']} - {bottleneck['contribution_percent']:.1f}% "
                f"({bottleneck['total_time_ms']:.2f} ms) - {bottleneck['optimization_potential']} priority"
            )
        
        report_lines.extend([
            "",
            "OPTIMIZATION RECOMMENDATIONS:",
        ])
        
        for i, rec in enumerate(results.get('optimization_recommendations', []), 1):
            report_lines.append(f"{i}. {rec}")
        
        # Save report
        report_content = "\n".join(report_lines)
        report_file = self.output_dir / f"optimization_report_{int(time.time())}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"📄 Comprehensive report saved to: {report_file}")
        
        # Also print to console
        print("\n" + report_content)
        
        return str(report_file)


def main():
    """Main CLI interface for bottleneck analysis"""
    parser = argparse.ArgumentParser(description="Qwen2 Model Bottleneck Analyzer")
    parser.add_argument(
        "--mode", 
        choices=["establish_baseline", "analyze", "test_optimization", "report"],
        required=True,
        help="Analysis mode to run"
    )
    parser.add_argument(
        "--baseline_name", 
        default="baseline",
        help="Name for the baseline (when establishing baseline)"
    )
    parser.add_argument(
        "--optimization_name",
        help="Name of the optimization being tested (required for test_optimization mode)"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of inference runs for profiling"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate per run"
    )
    parser.add_argument(
        "--output_dir",
        default="optimization_reports",
        help="Directory to save reports"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = BottleneckAnalyzer(output_dir=args.output_dir)
    
    try:
        if args.mode == "establish_baseline":
            print(f"🎯 Establishing baseline: {args.baseline_name}")
            analyzer.establish_baseline(args.baseline_name)
            
        elif args.mode == "analyze":
            print("🔍 Analyzing current performance...")
            analyzer.analyze_current_performance()
            
        elif args.mode == "test_optimization":
            if not args.optimization_name:
                raise ValueError("--optimization_name is required for test_optimization mode")
            print(f"🧪 Testing optimization: {args.optimization_name}")
            analyzer.test_optimization(args.optimization_name)
            
        elif args.mode == "report":
            print("📋 Generating comprehensive optimization report...")
            analyzer.generate_optimization_report()
            
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise
    
    print("\n✅ Analysis complete! Check the output directory for detailed reports.")


if __name__ == "__main__":
    main()