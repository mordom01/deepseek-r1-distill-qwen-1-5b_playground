#!/usr/bin/env python3
"""
multi_model_profiler.py

Kernel profiler for different model architectures including MoE models.
Supports various model sizes and architectures with optimized profiling settings.

Usage:
    python multi_model_profiler.py --model deepseek-r1-14b --batch_sizes 1,2,4
    python multi_model_profiler.py --model qwen2.5-14b --batch_sizes 1,2
    python multi_model_profiler.py --list_models
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

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error reporting

def setup_local_cache():
    """Setup local HuggingFace cache directory"""
    import os
    from pathlib import Path
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.parent.parent.absolute()
    local_cache = script_dir / "huggingface_cache"
    
    if local_cache.exists():
        os.environ['HF_HOME'] = str(local_cache)
        os.environ['HUGGINGFACE_HUB_CACHE'] = str(local_cache / "hub")
        print(f"✅ Using local HuggingFace cache: {local_cache}")
        return True
    else:
        print(f"❌ Local cache directory not found: {local_cache}")
        return False

class ModelConfig:
    """Configuration for different models"""
    
    def __init__(self, 
                 model_name: str,
                 hf_name: str,
                 architecture: str,
                 approx_size: str,
                 company: str,
                 release_date: str,
                 special_features: str = "",
                 max_batch_size: int = 8,
                 max_tokens: int = 100,
                 memory_efficient: bool = True,
                 is_moe: bool = False):
        self.model_name = model_name
        self.hf_name = hf_name
        self.architecture = architecture
        self.approx_size = approx_size
        self.company = company
        self.release_date = release_date
        self.special_features = special_features
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens
        self.memory_efficient = memory_efficient
        self.is_moe = is_moe

# Model registry with LATEST available models (Updated June 2025)
MODEL_REGISTRY = {
    
    # ================================================================================
    # CATEGORY 1: ~7-8B PARAMETER MODELS (Latest 2024-2025) 🎯
    # Perfect for finding common bottlenecks across different architectures
    # ================================================================================
    
    "qwen2.5-7b": ModelConfig(
        model_name="Qwen2.5-7B-Instruct", 
        hf_name="Qwen/Qwen2.5-7B-Instruct",
        architecture="Qwen2.5 Transformer",
        approx_size="7B",
        company="Alibaba",
        release_date="September 2024",
        special_features="Enhanced coding & math, 128K context",
        max_batch_size=8,
        max_tokens=120,
        is_moe=False
    ),

    "mixtral-8x7b": ModelConfig(
    model_name="Mixtral-8x7B-Instruct",
    hf_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    architecture="MoE Transformer (2 of 8 experts active)",
    approx_size="12.9B total, 2x7B active",
    company="Mistral",
    release_date="December 2023",
    special_features="MoE routing, efficient inference, 32K context",
    max_batch_size=6,
    max_tokens=120,
    is_moe=True
    ),

    "qwen1.5-moe-a2.7b": ModelConfig(
    model_name="Qwen1.5-MoE-A2.7B",
    hf_name="Qwen/Qwen1.5-MoE-A2.7B",
    architecture="MoE Transformer (8 experts, top‑2 routing)",
    approx_size="18.4B total, 2.7B active",
    company="Alibaba",
    release_date="March 2024",
    special_features="MoE with 2-of-8 expert activation, 32K context length, strong multilingual & reasoning performance",
    max_batch_size=8,
    max_tokens=120,
    is_moe=True
    ),
    
    "deepseek-r1-7b": ModelConfig(
        model_name="DeepSeek-R1-Distill-Qwen-7B",
        hf_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
        architecture="Qwen2.5 + R1 Chain-of-Thought",
        approx_size="7B",
        company="DeepSeek",
        release_date="December 2024", 
        special_features="Reasoning via reinforcement learning, O1-level performance",
        max_batch_size=6,
        max_tokens=100,
        is_moe=False
    ),
    
    "deepseek-r1-qwen3-8b": ModelConfig(
        model_name="DeepSeek-R1-0528-Qwen3-8B",
        hf_name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        architecture="Qwen3 + DeepSeek R1-0528",
        approx_size="8B", 
        company="DeepSeek",
        release_date="May 2025",
        special_features="Latest R1 distillation, SOTA reasoning for 8B",
        max_batch_size=8,
        max_tokens=120,
        is_moe=False
    ),
    
    "qwen3-8b": ModelConfig(
        model_name="Qwen3-8B",
        hf_name="Qwen/Qwen3-8B",
        architecture="Qwen3 Transformer",
        approx_size="8B",
        company="Alibaba",
        release_date="June 2025",
        special_features="Latest Qwen3 architecture, enhanced performance",
        max_batch_size=8,
        max_tokens=120,
        is_moe=False
    ),

    "qwen3-32b": ModelConfig(
        model_name="Qwen3-32B",
        hf_name="Qwen/Qwen3-32B",
        architecture="Qwen3 Transformer",
        approx_size="32B",
        company="Alibaba",
        release_date="June 2025",
        special_features="Latest Qwen3 architecture, enhanced performance",
        max_batch_size=16,
        max_tokens=120,
        is_moe=False
    ),

    "qwen3-30b-3a": ModelConfig(
        model_name="Qwen3-32B",
        hf_name="Qwen/Qwen3-30B-A3B",
        architecture="Qwen3 Transformer",
        approx_size="30B",
        company="Alibaba",
        release_date="June 2025",
        special_features="Latest Qwen3 MOE architecture, enhanced performance",
        max_batch_size=16,
        max_tokens=120,
        is_moe=True
    ),
    
    "gemma2-9b": ModelConfig(
        model_name="Gemma-2-9B-IT",
        hf_name="google/gemma-2-9b-it",
        architecture="Gemma-2 Transformer", 
        approx_size="9B",
        company="Google",
        release_date="June 2024",
        special_features="Distilled from Gemini, sliding window attention",
        max_batch_size=8,
        max_tokens=120,
        is_moe=False
    ),
    
    # ================================================================================
    # CATEGORY 2: ~11-14B PARAMETER MODELS (Latest 2024-2025) 📊
    # Mid-size comparison for scaling analysis
    # ================================================================================
    
    "qwen2.5-14b": ModelConfig(
        model_name="Qwen2.5-14B-Instruct",
        hf_name="Qwen/Qwen2.5-14B-Instruct", 
        architecture="Qwen2.5 Transformer",
        approx_size="14B",
        company="Alibaba",
        release_date="September 2024",
        special_features="Enhanced reasoning, structured output (JSON)",
        max_batch_size=4,
        max_tokens=100,
        is_moe=False
    ),
    
    "deepseek-r1-14b": ModelConfig(
        model_name="DeepSeek-R1-Distill-Qwen-14B", 
        hf_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        architecture="Qwen2.5 + R1 Chain-of-Thought",
        approx_size="14B",
        company="DeepSeek", 
        release_date="December 2024",
        special_features="Advanced reasoning, outperforms O1-mini",
        max_batch_size=4,
        max_tokens=100,
        is_moe=False
    ),
    
    "llama3.2-11b-vision": ModelConfig(
        model_name="Llama-3.2-11B-Vision-Instruct",
        hf_name="meta-llama/Llama-3.2-11B-Vision-Instruct", 
        architecture="Llama 3.1 + Vision Adapter",
        approx_size="11B",
        company="Meta",
        release_date="September 2024", 
        special_features="Multimodal (text+images), first Llama vision model",
        max_batch_size=4,
        max_tokens=100,
        is_moe=False
    ),
    
    "qwen2.5-coder-14b": ModelConfig(
        model_name="Qwen2.5-Coder-14B",
        hf_name="Qwen/Qwen2.5-Coder-14B",
        architecture="Qwen2.5 Coding Specialized", 
        approx_size="14B",
        company="Alibaba",
        release_date="September 2024",
        special_features="Code-specialized, 32K context, YaRN scaling",
        max_batch_size=4,
        max_tokens=100,
        is_moe=False
    ),
    
    # ================================================================================
    # CATEGORY 3: SMALL EFFICIENT MODELS (2-3B) for Memory Bandwidth Analysis ⚡
    # Perfect for detailed profiling with higher batch sizes
    # ================================================================================
    
    "gemma2-2b": ModelConfig(
        model_name="Gemma-2-2B-IT",
        hf_name="google/gemma-2-2b-it",
        architecture="Gemma-2 Transformer",
        approx_size="2B",
        company="Google", 
        release_date="June 2024",
        special_features="Highly optimized small model, knowledge distillation",
        max_batch_size=16,
        max_tokens=150,
        is_moe=False
    ),
    
    "deepseek-r1-1.5b": ModelConfig(
        model_name="DeepSeek-R1-Distill-Qwen-1.5B",
        hf_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
        architecture="Qwen2.5 + R1 Chain-of-Thought",
        approx_size="1.5B",
        company="DeepSeek",
        release_date="December 2024",
        special_features="Tiny reasoning model, exceptional efficiency",
        max_batch_size=16,
        max_tokens=150,
        is_moe=False
    ),
    
    "qwen2.5-3b": ModelConfig(
        model_name="Qwen2.5-3B-Instruct", 
        hf_name="Qwen/Qwen2.5-3B-Instruct",
        architecture="Qwen2.5 Transformer",
        approx_size="3B",
        company="Alibaba",
        release_date="September 2024", 
        special_features="Efficient mid-size, good performance/efficiency ratio",
        max_batch_size=12,
        max_tokens=140,
        is_moe=False
    ),
    
    # ================================================================================
    # CATEGORY 4: MIXTURE OF EXPERTS (MoE) MODELS 🔶
    # For analyzing expert routing and MoE-specific computational patterns
    # ================================================================================
    
    "mixtral-8x7b": ModelConfig(
        model_name="Mixtral-8x7B-Instruct-v0.1",
        hf_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        architecture="Mixtral MoE (8 experts, 2 active)",
        approx_size="47B total, 13B active",
        company="Mistral AI",
        release_date="December 2023",
        special_features="Sparse MoE, top-2 expert routing, outperforms Llama2-70B",
        max_batch_size=2,
        max_tokens=80,
        is_moe=True
    ),
    
    "mixtral-8x22b": ModelConfig(
        model_name="Mixtral-8x22B-Instruct-v0.1",
        hf_name="mistralai/Mixtral-8x22B-Instruct-v0.1",
        architecture="Mixtral MoE (8 experts, 2 active)",
        approx_size="141B total, 39B active",
        company="Mistral AI",
        release_date="April 2024",
        special_features="Large-scale MoE, superior performance/cost ratio",
        max_batch_size=1,
        max_tokens=60,
        is_moe=True
    ),
    
    # Note: These models may require special access or may not be publicly available
    # They're included for demonstration of MoE profiling capabilities
    "deepseek-v3-moe": ModelConfig(
        model_name="DeepSeek-V3-MoE",
        hf_name="deepseek-ai/deepseek-v3",  # Hypothetical - check actual availability
        architecture="DeepSeek MoE (Multi-Expert)",
        approx_size="671B total, 37B active",
        company="DeepSeek",
        release_date="2024",
        special_features="Massive MoE model, advanced expert routing",
        max_batch_size=1,
        max_tokens=50,
        is_moe=True
    ),
    
    "qwen2.5-moe-a14b": ModelConfig(
        model_name="Qwen2.5-MoE-A14B",
        hf_name="Qwen/Qwen2.5-MoE-A14B",  # Check if this exists
        architecture="Qwen2.5 MoE",
        approx_size="~100B total, 14B active",
        company="Alibaba",
        release_date="2024",
        special_features="Qwen MoE architecture, efficient expert routing",
        max_batch_size=2,
        max_tokens=70,
        is_moe=True
    ),
}

class MultiModelProfiler:
    """
    Advanced profiler supporting multiple model architectures
    """
    
    def __init__(self, output_dir: str = "multi_model_profiling"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set reproducible seed
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        print("🎲 Set seed to 42 for reproducible results")
        
        self.model = None
        self.tokenizer = None
        self.current_config = None
    
    def list_available_models(self):
        """List all available models in the registry"""
        print("\n📋 LATEST MODELS (June 2025):")
        print("=" * 70)
        
        # Group by type and highlight latest
        dense_models = []
        moe_models = []
        
        for key, config in MODEL_REGISTRY.items():
            if config.is_moe:
                moe_models.append((key, config))
            else:
                dense_models.append((key, config))
        
        print("\n🔥 DENSE MODELS:")
        print("-" * 50)
        print(f"{'Model Key':<25} | {'Model Name':<30} | {'Size':<20} | {'Max BS':<6}")
        print("-" * 50)
        for key, config in dense_models:
            print(f"{key:<25} | {config.model_name:<30} | {config.approx_size:<20} | {config.max_batch_size:<6}")
        
        print("\n🔶 MIXTURE OF EXPERTS (MoE) MODELS:")
        print("-" * 50)
        print(f"{'Model Key':<25} | {'Model Name':<30} | {'Size':<20} | {'Max BS':<6}")
        print("-" * 50)
        for key, config in moe_models:
            availability = "🔥" if "mixtral" in key else "⚠️"  # Mixtral models are definitely available
            print(f"{availability} {key:<23} | {config.model_name:<30} | {config.approx_size:<20} | {config.max_batch_size:<6}")
        
        print(f"\n💡 Usage: python {sys.argv[0]} --model <model_key> --batch_sizes 1,2,4")
        print("\n🔥 RECOMMENDED FOR PROFILING:")
        print("   Dense Models:")
        print("   • qwen3-8b: 🔥 LATEST Qwen3 architecture (June 2025)")
        print("   • qwen2.5-14b: Enhanced reasoning, structured output")
        print("   • deepseek-r1-14b: Advanced reasoning with R1 chain-of-thought")
        print("   • gemma2-9b: Google's efficient Gemma-2 architecture")
        print("\n   MoE Models:")
        print("   • mixtral-8x7b: Proven MoE model (47B total, 13B active)")
        print("   • mixtral-8x22b: Large-scale MoE (141B total, 39B active)")
        print("\n💡 MoE models show unique profiling patterns vs dense models!")
        print("⚠️ Some MoE models may require special access or may not be publicly available")
    
    def load_model(self, model_key: str) -> bool:
        """Load a specific model from the registry"""
        
        if model_key not in MODEL_REGISTRY:
            print(f"❌ Model '{model_key}' not found in registry")
            print("Use --list_models to see available models")
            return False
        
        config = MODEL_REGISTRY[model_key]
        self.current_config = config
        
        print(f"🔄 Loading {config.model_name}...")
        print(f"   HuggingFace: {config.hf_name}")
        print(f"   Architecture: {config.architecture}")
        print(f"   Size: {config.approx_size}")
        print(f"   MoE: {'Yes' if config.is_moe else 'No'}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.hf_name, 
                trust_remote_code=True
            )
            
            # FORCE 4-bit quantization more explicitly
            from transformers import BitsAndBytesConfig
            
            # Clear ALL GPU memory first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"   🧹 GPU memory before loading: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
            
            # Very explicit quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,  # Try bfloat16 instead of float16
            )
            
            # Adjust memory limits for MoE models
            max_gpu_memory = "25GB" if config.is_moe else "35GB"
            
            model_kwargs = {
                "quantization_config": quantization_config,
                "device_map": "auto",
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,  # Match quantization dtype
                "low_cpu_mem_usage": True,
                "max_memory": {0: max_gpu_memory, "cpu": "100GB"},
            }
            
            print(f"   🔧 Using EXPLICIT 4-bit quantization")
            print(f"   📊 Max GPU: {max_gpu_memory}, CPU offload: 100GB")
            if config.is_moe:
                print(f"   🔶 MoE model - using conservative memory limits")
            
            self.model = AutoModelForCausalLM.from_pretrained(config.hf_name, **model_kwargs)
            
            # Verify memory usage
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.memory_allocated() / 1024**3
                print(f"   📊 GPU memory after loading: {gpu_memory_gb:.1f} GB")
                
                expected_limit = 25 if config.is_moe else 35
                if gpu_memory_gb > expected_limit:
                    print("🚨 WARNING: Using more GPU memory than expected!")
                    print("   Quantization may not be working properly")
            
            return True

        except Exception as e:
            print(f"❌ Failed: {e}")
            if config.is_moe and "mixtral" not in model_key:
                print("💡 This MoE model may not be publicly available")
                print("   Try mixtral-8x7b or mixtral-8x22b instead")
            return False
    
    def load_gsm8k_samples(self, num_samples: int = 8) -> List[str]:
        """Load GSM8K samples optimized for different model types"""
        try:
            dataset = load_dataset("gsm8k", "main", split="test")
            samples = dataset.select(range(min(num_samples, len(dataset))))
            
            prompts = []
            for sample in samples:
                # Adjust prompt format based on model architecture
                if self.current_config and "llama" in self.current_config.architecture.lower():
                    # Llama format
                    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nSolve this math problem step by step:\n\n{sample['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                elif self.current_config and "qwen" in self.current_config.architecture.lower():
                    # Qwen format
                    prompt = f"<|im_start|>user\nSolve this math problem step by step:\n\n{sample['question']}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    # Generic format
                    prompt = f"Solve this math problem step by step:\n\n{sample['question']}\n\nAnswer:"
                
                prompts.append(prompt)
            
            print(f"✅ Loaded {len(prompts)} GSM8K prompts with {self.current_config.architecture} format")
            return prompts
            
        except Exception as e:
            print(f"⚠️ Failed to load GSM8K, using fallback prompts: {e}")
            return [
                "What is 25 + 37?",
                "Calculate 12 × 8 + 15", 
                "If 5 apples cost $2, how much do 20 apples cost?",
                "Find the area of a rectangle with length 8 and width 6."
            ][:num_samples]
    
    def profile_model_batch(self, batch_size: int) -> Dict[str, Any]:
        """Profile a specific batch size with model-specific optimizations"""
        
        if not self.model or not self.current_config:
            raise ValueError("No model loaded")
        
        # Adjust max_tokens based on model size and type
        max_tokens = min(self.current_config.max_tokens, 150)
        if self.current_config.is_moe:
            max_tokens = min(max_tokens, 80)  # More conservative for MoE
        
        print(f"🔍 Profiling {self.current_config.model_name} - Batch size {batch_size}")
        print(f"   Max tokens: {max_tokens}")
        if self.current_config.is_moe:
            print(f"   🔶 MoE model - expect expert routing overhead")
        
        # Set seed for reproducible results
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        # Load prompts
        prompts = self.load_gsm8k_samples(batch_size)
        
        # Prepare batch
        inputs = self.tokenizer(
            prompts,
            padding='longest',
            truncation=True,
            max_length=1024,  # Conservative for large models
            return_tensors="pt"
        )
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Warmup with shorter generation
        print("  Warmup...")
        with torch.no_grad():
            _ = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Main profiling run
        print("  Profiling...")
        
        # Adjust profiling settings for MoE models
        profiler_kwargs = {
            "activities": [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            "record_shapes": True,
            "profile_memory": True,
            "with_stack": False,
            "with_flops": True,
            "with_modules": False,
        }
        
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
        
        # Collect memory stats
        memory_stats = {
            'peak_memory_mb': torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            'current_memory_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
        }
        
        # Export trace
        trace_file = "trace_disabled"
        print(f"  ⏭️ Trace export skipped")
        
        # Analyze results
        analysis = self._analyze_model_results(prof, batch_size)
        
        return {
            'model_name': self.current_config.model_name,
            'model_key': [k for k, v in MODEL_REGISTRY.items() if v == self.current_config][0],
            'architecture': self.current_config.architecture,
            'is_moe': self.current_config.is_moe,
            'batch_size': batch_size,
            'total_time_ms': (end_time - start_time) * 1000,
            'memory_stats': memory_stats,
            'analysis': analysis,
            'trace_file': str(trace_file)
        }
    
    def _analyze_model_results(self, prof: torch.profiler.profile, batch_size: int) -> Dict[str, Any]:
        """Analyze profiling results with model-specific insights"""
        
        key_averages = prof.key_averages()
        
        cuda_ops = []
        cpu_ops = []
        
        for event in key_averages:
            device_time = getattr(event, 'device_time', 0)
            device_time_total = getattr(event, 'device_time_total', 0)
            cpu_time = getattr(event, 'cpu_time', 0)
            cpu_time_total = getattr(event, 'cpu_time_total', 0)
            
            event_data = {
                'name': event.key,
                'cpu_time_us': cpu_time,
                'device_time_us': device_time,
                'cpu_time_total': cpu_time_total,
                'device_time_total': device_time_total,
                'count': event.count,
                'input_shapes': getattr(event, 'input_shapes', []),
                'flops': getattr(event, 'flops', 0) if hasattr(event, 'flops') else 0
            }
            
            if device_time > 0 or 'cuda' in event.key.lower() or 'kernel' in event.key.lower():
                event_data['operation_type'] = self._classify_operation_advanced(event.key) 
                cuda_ops.append(event_data)
            else:
                cpu_ops.append(event_data)
        
        cuda_ops.sort(key=lambda x: x.get('device_time_total', 0), reverse=True)
        cpu_ops.sort(key=lambda x: x.get('cpu_time_total', 0), reverse=True)
        
        # Generate model-specific insights
        insights = self._generate_model_insights(cuda_ops, cpu_ops, batch_size)
        
        # Get table output
        try:
            table_cuda = prof.key_averages().table(sort_by="device_time_total", row_limit=15)
        except:
            try:
                table_cuda = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
            except:
                table_cuda = "Table generation failed"
        
        total_cuda_time = sum(op.get('device_time_total', 0) for op in cuda_ops)
        total_cpu_time = sum(op.get('cpu_time_total', 0) for op in cpu_ops)
        
        return {
            'top_cuda_ops': cuda_ops[:15],
            'top_cpu_ops': cpu_ops[:10],
            'table_cuda': table_cuda,
            'insights': insights,
            'total_cuda_time_us': total_cuda_time,
            'total_cpu_time_us': total_cpu_time,
            'moe_specific_analysis': self._analyze_moe_patterns(cuda_ops) if self.current_config.is_moe else None
        }
    
    def _classify_operation_advanced(self, op_name: str) -> str:
        """Advanced operation classification including MoE-specific operations"""
        op_lower = op_name.lower()
        
        # MoE-specific operations
        if any(x in op_lower for x in ['expert', 'router', 'gate', 'moe']):
            return 'moe_routing'
        elif any(x in op_lower for x in ['topk', 'top_k']):
            return 'moe_selection'
        
        # Standard classifications
        elif any(x in op_lower for x in ['gemm', 'sgemm', 'hgemm', 'matmul', 'mm', 'bmm']):
            return 'matrix_multiply'
        elif any(x in op_lower for x in ['attention', 'softmax', 'scaled_dot_product']):
            return 'attention'
        elif any(x in op_lower for x in ['copy', 'clone', 'cat', 'stack', 'view', 'reshape']):
            return 'memory_reshape' 
        elif any(x in op_lower for x in ['norm', 'layer_norm', 'rms_norm', 'batch_norm']):
            return 'normalization'
        elif any(x in op_lower for x in ['gelu', 'relu', 'silu', 'swish', 'tanh', 'sigmoid']):
            return 'activation'
        elif any(x in op_lower for x in ['embedding', 'embed']):
            return 'embedding'
        elif any(x in op_lower for x in ['add', 'mul', 'div', 'sub', 'elementwise']):
            return 'elementwise'
        else:
            return 'other'
    
    def _analyze_moe_patterns(self, cuda_ops: List[Dict]) -> Dict[str, Any]:
        """Analyze MoE-specific computational patterns"""
        
        moe_ops = [op for op in cuda_ops if op['operation_type'] in ['moe_routing', 'moe_selection']]
        
        if not moe_ops:
            return {'note': 'No MoE-specific operations detected in profiling'}
        
        total_moe_time = sum(op.get('device_time_total', 0) for op in moe_ops)
        total_time = sum(op.get('device_time_total', 0) for op in cuda_ops)
        
        moe_percentage = (total_moe_time / total_time) * 100 if total_time > 0 else 0
        
        return {
            'moe_operations_count': len(moe_ops),
            'moe_time_percentage': moe_percentage,
            'routing_overhead': total_moe_time / 1000,  # Convert to ms
            'optimization_potential': 'high' if moe_percentage > 15 else 'medium' if moe_percentage > 5 else 'low'
        }
    
    def _generate_model_insights(self, cuda_ops: List[Dict], cpu_ops: List[Dict], batch_size: int) -> List[str]:
        """Generate model-specific optimization insights"""
        insights = []
        
        if not cuda_ops:
            insights.append("⚠️ No CUDA operations detected - model may be running on CPU")
            return insights
        
        # Add model-specific context
        model_name = self.current_config.model_name
        is_moe = self.current_config.is_moe
        
        insights.append(f"🤖 Model: {model_name} {'(MoE)' if is_moe else '(Dense)'}")
        
        # Memory and compute analysis
        total_cuda_time = sum(op.get('device_time_total', 0) for op in cuda_ops)
        total_cpu_time = sum(op.get('cpu_time_total', 0) for op in cpu_ops)
        
        if total_cuda_time == 0:
            insights.append("⚠️ No CUDA time recorded - profiling may have failed")
            return insights
        
        # CPU overhead analysis
        if total_cpu_time > total_cuda_time and total_cuda_time > 0:
            cpu_ratio = total_cpu_time / total_cuda_time
            insights.append(f"🔴 High CPU overhead: {cpu_ratio:.1f}x GPU time")
            
            if is_moe:
                insights.append("   → MoE models have higher CPU overhead due to expert routing")
                insights.append("   → Consider CUDA graphs or expert caching optimizations")
            else:
                insights.append("   → Consider CUDA graphs or larger batch sizes")
        
        # Top operation analysis
        if cuda_ops:
            top_op = cuda_ops[0]
            op_type = top_op.get('operation_type', 'unknown')
            op_time_ms = top_op.get('device_time_total', 0) / 1000
            op_percentage = (top_op.get('device_time_total', 0) / total_cuda_time) * 100 if total_cuda_time > 0 else 0
            
            insights.append(f"🎯 Top bottleneck: {top_op['name'][:50]}...")
            insights.append(f"   Type: {op_type}, Time: {op_time_ms:.1f}ms ({op_percentage:.1f}% of GPU time)")
            
            # Model and operation specific recommendations
            if op_type == 'matrix_multiply':
                if is_moe:
                    insights.append("   → MoE optimization: Consider expert parallelization, sparse expert computation")
                else:
                    insights.append("   → Dense model optimization: Use mixed precision (FP16), optimize GEMM dimensions")
            elif op_type == 'moe_routing':
                insights.append("   → MoE routing optimization: Consider expert caching, routing efficiency improvements")
            elif op_type == 'attention':
                insights.append("   → Attention optimization: Consider Flash Attention, attention fusion")
        
        # MoE-specific insights
        if is_moe:
            moe_analysis = self._analyze_moe_patterns(cuda_ops)
            if moe_analysis and 'moe_time_percentage' in moe_analysis:
                moe_pct = moe_analysis['moe_time_percentage']
                insights.append(f"🔶 MoE routing overhead: {moe_pct:.1f}% of total GPU time")
                
                if moe_pct > 15:
                    insights.append("   → High MoE overhead - focus on expert routing optimization")
                elif moe_pct > 5:
                    insights.append("   → Moderate MoE overhead - consider expert load balancing")
                else:
                    insights.append("   → Low MoE overhead detected - may indicate sparse activations")
        
        # Batch size insights
        if batch_size == 1:
            insights.append("⚡ Batch size 1 - high per-sample overhead")
            if is_moe:
                insights.append("   → MoE models especially benefit from larger batch sizes")
        elif batch_size >= 8:
            insights.append("🔋 Large batch - monitor memory usage")
            if is_moe:
                insights.append("   → MoE models may have uneven expert utilization with large batches")
        
        return insights
    
    def run_model_comparison(self, batch_sizes: List[int]) -> Dict[str, Any]:
        """Run profiling across multiple batch sizes for the loaded model"""
        
        if not self.model or not self.current_config:
            raise ValueError("No model loaded. Use load_model() first.")
        
        # Validate batch sizes against model limits
        max_batch = self.current_config.max_batch_size
        valid_batch_sizes = [bs for bs in batch_sizes if bs <= max_batch]
        
        if len(valid_batch_sizes) < len(batch_sizes):
            excluded = [bs for bs in batch_sizes if bs > max_batch]
            print(f"⚠️ Excluding batch sizes {excluded} (max for this model: {max_batch})")
        
        if not valid_batch_sizes:
            print(f"❌ No valid batch sizes for {self.current_config.model_name}")
            return {}
        
        print(f"🚀 Profiling {self.current_config.model_name} with batch sizes: {valid_batch_sizes}")
        
        results = {}
        
        for batch_size in valid_batch_sizes:
            try:
                result = self.profile_model_batch(batch_size)
                results[batch_size] = result
                
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
    
    def generate_model_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive model profiling report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_safe = self.current_config.model_name.replace(' ', '_').replace('/', '_')
        report_path = self.output_dir / f"{model_name_safe}_profiling_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("MULTI-MODEL KERNEL PROFILING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model: {self.current_config.model_name}\n")
            f.write(f"Architecture: {self.current_config.architecture}\n")
            f.write(f"Size: {self.current_config.approx_size}\n")
            f.write(f"Type: {'Mixture of Experts (MoE)' if self.current_config.is_moe else 'Dense Model'}\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
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
                
                # Top operations
                f.write("Top GPU Operations:\n")
                for i, op in enumerate(analysis['top_cuda_ops'][:10], 1):
                    op_name = op['name'][:50]
                    op_type = op.get('operation_type', 'unknown')
                    device_time_ms = op.get('device_time_total', 0) / 1000
                    count = op.get('count', 0)
                    
                    f.write(f"{i:2d}. {op_name}\n")
                    f.write(f"    Type: {op_type}, Time: {device_time_ms:.2f}ms, Calls: {count}\n\n")
                
                # MoE-specific analysis
                if self.current_config.is_moe and analysis.get('moe_specific_analysis'):
                    moe_analysis = analysis['moe_specific_analysis']
                    f.write("MoE-Specific Analysis:\n")
                    f.write(f"  Routing overhead: {moe_analysis.get('moe_time_percentage', 0):.1f}% of GPU time\n")
                    f.write(f"  Optimization potential: {moe_analysis.get('optimization_potential', 'unknown')}\n\n")
                
                # Insights
                f.write("Optimization Insights:\n")
                for insight in analysis['insights']:
                    f.write(f"{insight}\n")
                
                f.write("\n" + "="*50 + "\n\n")
            
            # Model-specific recommendations
            f.write("MODEL-SPECIFIC RECOMMENDATIONS:\n")
            f.write("-" * 35 + "\n")
            
            if self.current_config.is_moe:
                f.write("MoE Model Optimizations:\n")
                f.write("• Focus on expert routing efficiency\n")
                f.write("• Consider expert load balancing\n")
                f.write("• Evaluate expert caching strategies\n")
                f.write("• Monitor expert utilization patterns\n")
                f.write("• Consider dynamic expert selection\n\n")
            else:
                f.write("Dense Model Optimizations:\n")
                f.write("• Focus on matrix multiplication kernels\n")
                f.write("• Use mixed precision (FP16) training/inference\n")
                f.write("• Optimize attention mechanisms\n")
                f.write("• Consider model parallelization\n")
                f.write("• Evaluate kernel fusion opportunities\n\n")
            
            f.write("General Recommendations:\n")
            f.write("• Use Chrome tracing for visual timeline analysis\n")
            f.write("• Focus optimization on top 3-5 operations\n")
            f.write("• Test different batch sizes for optimal throughput\n")
            f.write("• Consider torch.compile() for kernel fusion\n")
            f.write("• Monitor memory usage vs. batch size scaling\n\n")
            
            f.write(f"Trace files: {self.output_dir}/*_trace.json\n")
        
        print(f"📊 Report saved to: {report_path}")
        return str(report_path)


def main():

    if not setup_local_cache():
        print("Warning: Using default cache location")
        
    parser = argparse.ArgumentParser(description="Multi-model kernel profiler")
    parser.add_argument("--model", help="Model key to profile (use --list_models to see options)")
    parser.add_argument("--list_models", action="store_true", help="List available models")
    parser.add_argument("--batch_sizes", default="1,2,4", help="Comma-separated batch sizes")
    parser.add_argument("--output_dir", default="multi_model_profiling", help="Output directory")
    
    args = parser.parse_args()
    
    profiler = MultiModelProfiler(args.output_dir)
    
    if args.list_models:
        profiler.list_available_models()
        return
    
    if not args.model:
        print("Please specify --model or use --list_models to see available options")
        return
    
    try:
        # Load model
        if not profiler.load_model(args.model):
            return
        
        # Parse batch sizes
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(',')]
        
        # Run profiling
        results = profiler.run_model_comparison(batch_sizes)
        
        if not results:
            print("No profiling results generated")
            return
        
        # Generate report
        report_path = profiler.generate_model_report(results)
        
        print("\n" + "=" * 60)
        print("✅ MULTI-MODEL PROFILING COMPLETE!")
        print("=" * 60)
        print(f"📊 Report: {report_path}")
        print(f"📈 Trace files: {profiler.output_dir}/*_trace.json")
        print("\n💡 Key insights:")
        print("• Compare MoE vs Dense model computational patterns")
        print("• Focus on expert routing overhead in MoE models")
        print("• Use Chrome tracing for detailed timeline analysis")
        print("• Optimize the top 3-5 operations for maximum impact")
        
    except KeyboardInterrupt:
        print("\n⏹️ Profiling interrupted")
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()