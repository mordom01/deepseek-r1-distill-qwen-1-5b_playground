#!/usr/bin/env python3
"""
bnb_counter_bindings.py

Simplified Python bindings that extract operation counters from instrumented BitsAndBytes kernels.
No timing overhead - just counts operations for validation.
"""

import ctypes
import os
import sys
from pathlib import Path
from typing import Dict, Optional

class BnBCounterExtractor:
    """Extract operation counters from instrumented BitsAndBytes kernels"""
    
    def __init__(self):
        self.lib = None
        self._load_library()
        
    def _load_library(self):
        """Load the instrumented BitsAndBytes library"""
        try:
            import bitsandbytes
            bnb_path = Path(bitsandbytes.__file__).parent
            
            # Search for CUDA library with instrumentation
            import glob
            cuda_libs = glob.glob(str(bnb_path / "**" / "*cuda*.so"), recursive=True)
            
            print(f"🔍 Found CUDA libraries: {cuda_libs}")
            
            # Try CUDA libraries
            for cuda_path in cuda_libs:
                if os.path.exists(cuda_path):
                    try:
                        print(f"🔄 Trying CUDA library: {cuda_path}")
                        self.lib = ctypes.CDLL(str(cuda_path))
                        
                        # Test if instrumentation functions exist
                        if hasattr(self.lib, 'extract_bnb_counter_data'):
                            print(f"✅ Loaded instrumented CUDA library: {cuda_path}")
                            self._setup_function_prototypes()
                            return
                        else:
                            print(f"⚠️ Library missing instrumentation: {cuda_path}")
                            self.lib = None
                    except OSError as e:
                        print(f"⚠️ Failed to load {cuda_path}: {e}")
            
            print("❌ No instrumented CUDA library found")
            
        except Exception as e:
            print(f"❌ Failed to load library: {e}")
            self.lib = None
    
    def _setup_function_prototypes(self):
        """Set up ctypes function prototypes for counter functions"""
        
        try:
            # Check which functions exist
            functions = [
                'extract_bnb_counter_data',
                'get_nf4_lookup_count', 
                'get_scaling_factor_count',
                'get_memory_access_count',
                'get_kernel_call_count',
                'reset_bnb_counters'
            ]
            
            for func_name in functions:
                if hasattr(self.lib, func_name):
                    print(f"✅ Found function: {func_name}")
                else:
                    print(f"⚠️ Function not found: {func_name}")
            
            # extract_bnb_counter_data() -> void
            if hasattr(self.lib, 'extract_bnb_counter_data'):
                self.lib.extract_bnb_counter_data.argtypes = []
                self.lib.extract_bnb_counter_data.restype = None
            
            # get_*_count(count*) -> void
            for func_name in ['get_nf4_lookup_count', 'get_scaling_factor_count', 
                             'get_memory_access_count', 'get_kernel_call_count']:
                if hasattr(self.lib, func_name):
                    getattr(self.lib, func_name).argtypes = [ctypes.POINTER(ctypes.c_uint)]
                    getattr(self.lib, func_name).restype = None
            
            # reset_bnb_counters() -> void
            if hasattr(self.lib, 'reset_bnb_counters'):
                self.lib.reset_bnb_counters.argtypes = []
                self.lib.reset_bnb_counters.restype = None
                
        except Exception as e:
            print(f"⚠️ Function prototype setup failed: {e}")
    
    def test_instrumentation(self) -> bool:
        """Test if counter instrumentation is working"""
        print("🧪 Testing counter instrumentation...")
        
        if not self.lib:
            print("❌ Library not loaded")
            return False
        
        # Check if key functions exist
        key_functions = ['extract_bnb_counter_data', 'reset_bnb_counters']
        for func in key_functions:
            if not hasattr(self.lib, func):
                print(f"❌ Function {func} not found")
                return False
        
        # Try to reset counters
        try:
            self.lib.reset_bnb_counters()
            print("✅ Successfully called reset_bnb_counters")
            return True
        except Exception as e:
            print(f"❌ Failed to call functions: {e}")
            return False
    
    def extract_counter_data(self):
        """Extract counter data from GPU to CPU"""
        if self.lib and hasattr(self.lib, 'extract_bnb_counter_data'):
            self.lib.extract_bnb_counter_data()
    
    def get_nf4_lookup_count(self) -> int:
        """Get NF4 lookup operation count"""
        if not self.lib or not hasattr(self.lib, 'get_nf4_lookup_count'):
            return 0
        
        try:
            count = ctypes.c_uint()
            self.lib.get_nf4_lookup_count(ctypes.byref(count))
            return count.value
        except Exception:
            return 0
    
    def get_scaling_factor_count(self) -> int:
        """Get scaling factor access count"""
        if not self.lib or not hasattr(self.lib, 'get_scaling_factor_count'):
            return 0
        
        try:
            count = ctypes.c_uint()
            self.lib.get_scaling_factor_count(ctypes.byref(count))
            return count.value
        except Exception:
            return 0
    
    def get_memory_access_count(self) -> int:
        """Get memory access operation count"""
        if not self.lib or not hasattr(self.lib, 'get_memory_access_count'):
            return 0
        
        try:
            count = ctypes.c_uint()
            self.lib.get_memory_access_count(ctypes.byref(count))
            return count.value
        except Exception:
            return 0
    
    def get_kernel_call_count(self) -> int:
        """Get total kernel call count"""
        if not self.lib or not hasattr(self.lib, 'get_kernel_call_count'):
            return 0
        
        try:
            count = ctypes.c_uint()
            self.lib.get_kernel_call_count(ctypes.byref(count))
            return count.value
        except Exception:
            return 0
    
    def reset_counters(self):
        """Reset all operation counters with enhanced verification for large models"""
        if self.lib and hasattr(self.lib, 'reset_bnb_counters'):
            try:
                import torch
                if not torch.cuda.is_available():
                    return False
                
                # Save current device
                current_device = torch.cuda.current_device()
                
                # Method 1: Stream-based reset with memory clearing
                reset_stream = torch.cuda.Stream()
                with torch.cuda.stream(reset_stream):
                    # Clear GPU caches before reset
                    torch.cuda.empty_cache()
                    
                    # Reset counters
                    self.lib.reset_bnb_counters()
                    
                    # Force completion
                    reset_stream.synchronize()
                
                # Method 2: Device-wide synchronization
                torch.cuda.synchronize(device=current_device)
                
                # Method 3: Memory fence for global consistency
                if hasattr(torch.cuda, 'memory_barrier'):
                    torch.cuda.memory_barrier()
                
                # Extra synchronization for large models
                import time
                time.sleep(0.2)  # Increased delay for large models
                
                # Method 4: Double reset (sometimes helps with persistent state)
                self.lib.reset_bnb_counters()
                torch.cuda.synchronize()
                time.sleep(0.1)
                
                # Force data extraction to refresh host-side copies
                self.extract_counter_data()
                torch.cuda.synchronize()
                
                # Verify reset worked
                for verify_attempt in range(3):
                    self.extract_counter_data()
                    
                    nf4_count = self.get_nf4_lookup_count()
                    scaling_count = self.get_scaling_factor_count()
                    memory_count = self.get_memory_access_count()
                    kernel_count = self.get_kernel_call_count()
                    
                    total_remaining = nf4_count + scaling_count + memory_count + kernel_count
                    
                    if total_remaining == 0:
                        return True
                        
                    # If not zero, try one more aggressive sync
                    torch.cuda.synchronize()
                    time.sleep(0.1)
                
                # If we get here, reset failed
                print(f"⚠️ Reset incomplete after enhanced attempts: {total_remaining:,} ops remaining")
                print(f"   NF4: {nf4_count:,}, Scaling: {scaling_count:,}, Memory: {memory_count:,}, Kernels: {kernel_count:,}")
                
                # Last resort: Try to detect if it's a cumulative issue
                if kernel_count > 0 and kernel_count % 864 == 0:
                    print("   🔍 Detected multiple of base kernel count - likely accumulation issue")
                    
                return False
                    
            except Exception as e:
                print(f"❌ Counter reset error: {e}")
                import traceback
                traceback.print_exc()
                return False
        return False

    def force_gpu_memory_clear(self):
        """Force GPU memory clearing - call before reset for large models"""
        try:
            import torch
            if torch.cuda.is_available():
                # Clear all caches
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Additional synchronization
                torch.cuda.synchronize()
                
                return True
        except Exception as e:
            print(f"⚠️ GPU memory clear failed: {e}")
        return False
    
    def diagnose_counter_state(self):
        """Diagnose counter state issues for large models"""
        print("\n🔍 COUNTER STATE DIAGNOSIS")
        print("=" * 40)
        
        try:
            import torch
            
            # GPU state
            if torch.cuda.is_available():
                print(f"CUDA Device: {torch.cuda.get_device_name()}")
                print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
                print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
                
            # Extract current state
            self.extract_counter_data()
            counters = self.get_comprehensive_counter_report()
            
            print(f"\nCurrent Counter Values:")
            for key, value in counters.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:,}")
            
            # Check for patterns
            kernel_count = counters.get('kernel_call_count', 0)
            if kernel_count > 0:
                if kernel_count == 864:
                    print("\n✅ Kernel count matches single batch pattern")
                elif kernel_count % 864 == 0:
                    print(f"\n⚠️ Kernel count is {kernel_count // 864}x base - accumulation detected!")
                else:
                    print(f"\n🤔 Unusual kernel count: {kernel_count}")
                    
            # Test reset capability
            print("\nTesting reset capability...")
            original_count = counters.get('total_operations', 0)
            
            self.lib.reset_bnb_counters()
            torch.cuda.synchronize()
            self.extract_counter_data()
            
            new_counters = self.get_comprehensive_counter_report()
            new_count = new_counters.get('total_operations', 0)
            
            if new_count < original_count:
                print(f"✅ Partial reset achieved: {original_count:,} → {new_count:,}")
            elif new_count == original_count:
                print(f"❌ Reset had no effect: still {new_count:,}")
            else:
                print(f"🤔 Unexpected: count increased {original_count:,} → {new_count:,}")
                
        except Exception as e:
            print(f"❌ Diagnosis failed: {e}")
            import traceback
            traceback.print_exc()

    def diagnose_counter_state(self):
        """Diagnose counter state issues for large models"""
        print("\n🔍 COUNTER STATE DIAGNOSIS")
        print("=" * 40)
        
        try:
            import torch
            
            # GPU state
            if torch.cuda.is_available():
                print(f"CUDA Device: {torch.cuda.get_device_name()}")
                print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
                print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
                
            # Extract current state
            self.extract_counter_data()
            counters = self.get_comprehensive_counter_report()
            
            print(f"\nCurrent Counter Values:")
            for key, value in counters.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:,}")
            
            # Check for patterns
            kernel_count = counters.get('kernel_call_count', 0)
            if kernel_count > 0:
                if kernel_count == 864:
                    print("\n✅ Kernel count matches single batch pattern")
                elif kernel_count % 864 == 0:
                    print(f"\n⚠️ Kernel count is {kernel_count // 864}x base - accumulation detected!")
                else:
                    print(f"\n🤔 Unusual kernel count: {kernel_count}")
                    
            # Test reset capability
            print("\nTesting reset capability...")
            original_count = counters.get('total_operations', 0)
            
            self.lib.reset_bnb_counters()
            torch.cuda.synchronize()
            self.extract_counter_data()
            
            new_counters = self.get_comprehensive_counter_report()
            new_count = new_counters.get('total_operations', 0)
            
            if new_count < original_count:
                print(f"✅ Partial reset achieved: {original_count:,} → {new_count:,}")
            elif new_count == original_count:
                print(f"❌ Reset had no effect: still {new_count:,}")
            else:
                print(f"🤔 Unexpected: count increased {original_count:,} → {new_count:,}")
                
        except Exception as e:
            print(f"❌ Diagnosis failed: {e}")
            import traceback
            traceback.print_exc()
    
    def get_comprehensive_counter_report(self) -> Dict[str, int]:
        """Get comprehensive counter report for all operations"""
        
        # Extract latest counter data
        self.extract_counter_data()
        
        # Get all counts
        return {
            'nf4_lookup_count': self.get_nf4_lookup_count(),
            'scaling_factor_count': self.get_scaling_factor_count(),
            'memory_access_count': self.get_memory_access_count(),
            'kernel_call_count': self.get_kernel_call_count(),
            'total_operations': (self.get_nf4_lookup_count() + 
                               self.get_scaling_factor_count() + 
                               self.get_memory_access_count()),
        }


# Global counter extractor instance
_counter_extractor = None

def get_counter_extractor() -> BnBCounterExtractor:
    """Get global counter extractor instance"""
    global _counter_extractor
    if _counter_extractor is None:
        _counter_extractor = BnBCounterExtractor()
    return _counter_extractor

def test_instrumentation() -> bool:
    """Test if counter instrumentation is working"""
    try:
        extractor = get_counter_extractor()
        return extractor.test_instrumentation()
    except Exception as e:
        print(f"❌ Counter instrumentation test failed: {e}")
        return False

def print_counter_summary():
    """Print a summary of operation counters"""
    
    print("🔍 BITSANDBYTES OPERATION COUNTER SUMMARY")
    print("=" * 50)
    
    try:
        extractor = get_counter_extractor()
        report = extractor.get_comprehensive_counter_report()
        
        print(f"📊 Operation Counts:")
        print(f"  NF4 lookups: {report['nf4_lookup_count']:,}")
        print(f"  Scaling factor accesses: {report['scaling_factor_count']:,}")
        print(f"  Memory accesses: {report['memory_access_count']:,}")
        print(f"  Kernel calls: {report['kernel_call_count']:,}")
        print(f"  Total operations: {report['total_operations']:,}")
        
        if report['kernel_call_count'] > 0:
            print(f"\n📈 Averages per kernel call:")
            print(f"  NF4 lookups/kernel: {report['nf4_lookup_count'] / report['kernel_call_count']:.1f}")
            print(f"  Scaling accesses/kernel: {report['scaling_factor_count'] / report['kernel_call_count']:.1f}")
            print(f"  Memory accesses/kernel: {report['memory_access_count'] / report['kernel_call_count']:.1f}")
        
    except Exception as e:
        print(f"❌ Failed to extract counter data: {e}")

if __name__ == "__main__":
    print("🧪 Testing BitsAndBytes counter extraction...")
    
    if test_instrumentation():
        print("✅ Counter instrumentation test passed!")
        print_counter_summary()
    else:
        print("❌ Counter instrumentation test failed!")
        print("\n💡 Troubleshooting:")
        print("1. Make sure you compiled with BNB_ENABLE_INSTRUMENTATION=1")
        print("2. Check that instrumentation message appeared during compilation")
        print("3. Run some quantized operations first to generate counter data")