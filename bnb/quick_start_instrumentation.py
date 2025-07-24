#!/usr/bin/env python3
"""
quick_start_instrumentation.py

Complete example showing the CUDA → C → Python pipeline in action.
Tests counter instrumentation and runs a simple profiling example.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time

def test_pipeline():
    """Test the complete CUDA → C → Python instrumentation pipeline"""
    
    print("🧪 TESTING BITSANDBYTES INSTRUMENTATION PIPELINE")
    print("=" * 55)
    
    # Step 1: Test counter bindings
    print("\n1️⃣ Testing Python → C interface...")
    try:
        import bnb_counter_bindings as bnb
        extractor = bnb.get_counter_extractor()
        
        if extractor.test_instrumentation():
            print("✅ Counter bindings working!")
        else:
            print("❌ Counter bindings failed!")
            return False
    except ImportError:
        print("❌ bnb_counter_bindings not found - check file placement")
        return False
    
    # Step 2: Load quantized model (triggers kernels)
    print("\n2️⃣ Loading quantized model to trigger CUDA kernels...")
    
    try:
        # Reset counters
        extractor.reset_counters()
        
        # Quantization config - ensure NF4 blockwise dequantization
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,  # Single quantization for simpler testing
            bnb_4bit_quant_type="nf4",        # NF4 to trigger our instrumented kernels
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.uint8,
        )
        
        # Use larger model that definitely has quantized layers
        # microsoft/DialoGPT-medium is ~350M params vs ~117M for small
        model_name = "microsoft/DialoGPT-medium"
        print(f"   Loading {model_name} (350M params) to ensure quantized operations...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Verify quantization worked
        quantized_layers = 0
        total_layers = 0
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                total_layers += 1
                if hasattr(module.weight, 'quant_type'):
                    quantized_layers += 1
        
        print(f"✅ Model loaded: {quantized_layers}/{total_layers} layers quantized")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Step 3: Run inference (triggers dequantization kernels)
    print("\n3️⃣ Running inference to trigger NF4 blockwise dequantization...")
    
    try:
        # Use batch size >= 2 to ensure blockwise dequantization is triggered
        prompts = [
            "Hello, how are you doing today?",
            "What is the weather like?", 
        ]
        print(f"   Using batch size {len(prompts)} to trigger blockwise operations...")
        
        inputs = tokenizer(
            prompts, 
            padding=True, 
            return_tensors="pt"
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print(f"   Input shape: {inputs['input_ids'].shape}")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,  # More tokens to trigger more dequantization
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(f"✅ Generated {len(responses)} responses:")
        for i, response in enumerate(responses):
            print(f"   {i+1}: {response[:80]}...")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False
    
    # Step 4: Extract and verify counters
    print("\n4️⃣ Extracting counters from CUDA → C → Python...")
    
    try:
        # Extract counter data from GPU
        extractor.extract_counter_data()
        
        # Get comprehensive report
        counters = extractor.get_comprehensive_counter_report()
        
        print("📊 Counter Results:")
        print(f"  NF4 lookups: {counters['nf4_lookup_count']:,}")
        print(f"  Scaling factor accesses: {counters['scaling_factor_count']:,}")
        print(f"  Memory accesses: {counters['memory_access_count']:,}")
        print(f"  Kernel calls: {counters['kernel_call_count']:,}")
        print(f"  Total operations: {counters['total_operations']:,}")
        
        # Detailed validation
        nf4_count = counters['nf4_lookup_count']
        kernel_count = counters['kernel_call_count']
        total_ops = counters['total_operations']
        
        print("\n🔍 Pipeline Validation:")
        
        if total_ops > 0:
            print("✅ Operations detected - pipeline working!")
            
            if nf4_count > 0:
                print("✅ NF4 lookups detected - blockwise dequantization working!")
                avg_nf4_per_kernel = nf4_count / kernel_count if kernel_count > 0 else 0
                print(f"   Average NF4 lookups per kernel: {avg_nf4_per_kernel:.1f}")
                
                if avg_nf4_per_kernel > 100:  # Reasonable threshold for blockwise operations
                    print("✅ High NF4 lookup density - likely hitting blockwise kernels!")
                else:
                    print("⚠️ Low NF4 lookup density - may be hitting different kernel path")
            else:
                print("⚠️ No NF4 lookups - kernels may not be NF4 type or not instrumented")
                
            if kernel_count > 0:
                print(f"✅ {kernel_count} kernel calls detected")
            else:
                print("⚠️ No kernel calls detected - instrumentation may not be in active kernels")
                
            return True
        else:
            print("⚠️ No operations detected")
            print("   Possible causes:")
            print("   - Kernels not using instrumented code paths")
            print("   - Model too small or not properly quantized")
            print("   - Batch size too small for blockwise operations")
            print("   - Instrumentation not compiled correctly")
            return False
            
    except Exception as e:
        print(f"❌ Counter extraction failed: {e}")
        return False

def run_full_profiler_example():
    """Run the full profiler script example"""
    
    print("\n🚀 RUNNING FULL PROFILER EXAMPLE")
    print("=" * 40)
    
    try:
        from bnb_enhanced_profiler_counters import BnBCounterProfiler
        
        # Create profiler
        profiler = BnBCounterProfiler(use_counters=True)
        
        # Run profiling on medium model with proper batch sizes
        print("   Using DialoGPT-medium with batch sizes [2, 4] to ensure blockwise operations...")
        
        results = profiler.run_profiling(
            model_name="microsoft/DialoGPT-medium",  # Larger model
            batch_sizes=[2, 4],  # Batch sizes >= 2 for blockwise
            tag="quickstart"
        )
        
        if results:
            print("✅ Full profiling completed!")
            print(f"📊 Results for {len(results)} batch sizes")
            
            for batch_size, result in results.items():
                if 'error' not in result:
                    tokens_sec = result.get('tokens_per_sec', 0)
                    counters = result.get('operation_counters', {})
                    
                    if 'error' not in counters:
                        total_ops = counters.get('total_operations', 0)
                        nf4_ops = counters.get('nf4_lookup_count', 0)
                        kernel_calls = counters.get('kernel_call_count', 0)
                        
                        print(f"  Batch {batch_size}: {tokens_sec:.1f} tok/s")
                        print(f"    Operations: {total_ops:,} total, {nf4_ops:,} NF4 lookups, {kernel_calls} kernels")
                        
                        # Validate we're hitting the right operations
                        if nf4_ops > 0:
                            print(f"    ✅ NF4 blockwise dequantization detected!")
                        else:
                            print(f"    ⚠️ No NF4 operations - may not be using blockwise kernels")
                    else:
                        print(f"  Batch {batch_size}: {tokens_sec:.1f} tok/s, counter error: {counters['error']}")
                else:
                    print(f"  Batch {batch_size}: Error - {result['error']}")
            
            return True
        else:
            print("❌ No profiling results")
            return False
            
    except ImportError:
        print("❌ Full profiler script not found")
        return False
    except Exception as e:
        print(f"❌ Full profiler failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test sequence"""
    
    print("🎯 BITSANDBYTES INSTRUMENTATION QUICK START")
    print("=" * 50)
    print("This script tests the complete instrumentation pipeline:")
    print("  CUDA kernels → C interface → Python bindings → Profiler")
    print()
    
    # Test basic pipeline
    if test_pipeline():
        print("\n🎉 SUCCESS: Basic instrumentation pipeline working!")
        
        # Test full profiler
        if run_full_profiler_example():
            print("\n🏆 COMPLETE SUCCESS: Full profiling system working!")
            print("\n📋 Next steps:")
            print("1. Use the profiler for baseline measurements")
            print("2. Implement your optimizations")
            print("3. Re-run profiler to validate improvements")
            print("4. Use counters to verify operation patterns")
        else:
            print("\n⚠️ Basic pipeline works, but full profiler had issues")
            print("Check bnb_enhanced_profiler_counters.py placement")
    else:
        print("\n❌ PIPELINE FAILED")
        print("\n🔧 Troubleshooting steps:")
        print("1. Compile BitsAndBytes with: export BNB_ENABLE_INSTRUMENTATION=1")
        print("2. Check kernels.cu has instrumentation code")
        print("3. Verify CMakeLists.txt has instrumentation flag")
        print("4. Use larger model (>=350M params) for more quantized layers")
        print("5. Use batch size >= 2 to trigger blockwise dequantization")
        print("6. Ensure bnb_counter_bindings.py is in Python path")
        print("7. Test with: python bnb_counter_bindings.py")

if __name__ == "__main__":
    main()