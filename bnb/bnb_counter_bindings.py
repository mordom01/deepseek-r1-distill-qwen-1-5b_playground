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
        self.instrumentation_available = False
        self._load_library()
        
    def _load_library(self):
        """Load the instrumented BitsAndBytes library"""
        try:
            import bitsandbytes
            bnb_path = Path(bitsandbytes.__file__).parent
            
            # Search for CUDA library with instrumentation
            import glob
            cuda_libs = glob.glob(str(bnb_path / "**" / "*cuda*.so"), recursive=True)
            
            print(f"Found CUDA libraries: {cuda_libs}")
            
            # Try CUDA libraries
            for cuda_path in cuda_libs:
                if os.path.exists(cuda_path):
                    try:
                        print(f"Trying CUDA library: {cuda_path}")
                        self.lib = ctypes.CDLL(str(cuda_path))
                        
                        # Check if instrumentation functions exist
                        if hasattr(self.lib, 'extract_bnb_counter_data'):
                            print(f"Loaded instrumented CUDA library: {cuda_path}")
                            self._setup_function_prototypes()
                            return
                        else:
                            print(f"Library missing instrumentation: {cuda_path}")
                            self.lib = None
                    except OSError as e:
                        print(f"Failed to load {cuda_path}: {e}")
            
            print("No instrumented CUDA library found")
            
        except Exception as e:
            print(f"Failed to load library: {e}")
            self.lib = None
    
    def _setup_function_prototypes(self):
        """Set up ctypes function prototypes with fallback detection"""
        
        # Check if instrumentation is available
        if not hasattr(self.lib, 'extract_bnb_counter_data'):
            print("BitsAndBytes compiled without instrumentation support")
            self.instrumentation_available = False
            return
            
        self.instrumentation_available = True
        
        try:
            # Check which functions exist
            functions = [
                'extract_bnb_counter_data',
                'get_nf4_lookup_count', 
                'get_scaling_factor_count',
                'get_memory_access_count',
                'get_kernel_call_count',
                'reset_bnb_counters',
                'get_warp_divergence_count',
                'get_cache_line_loads',
                'get_bytes_loaded',
                'get_coalesced_loads',
                'get_scattered_loads'
            ]
            
            for func_name in functions:
                if hasattr(self.lib, func_name):
                    print(f"Found function: {func_name}")
                else:
                    print(f"Function not found: {func_name}")
            
            # extract_bnb_counter_data() -> void
            if hasattr(self.lib, 'extract_bnb_counter_data'):
                self.lib.extract_bnb_counter_data.argtypes = []
                self.lib.extract_bnb_counter_data.restype = None
            
            # Basic counter getters
            for func_name in ['get_nf4_lookup_count', 'get_scaling_factor_count', 
                            'get_memory_access_count', 'get_kernel_call_count',
                            'get_warp_divergence_count', 'get_cache_line_loads',
                            'get_coalesced_loads', 'get_scattered_loads']:
                if hasattr(self.lib, func_name):
                    getattr(self.lib, func_name).argtypes = [ctypes.POINTER(ctypes.c_uint)]
                    getattr(self.lib, func_name).restype = None
            
            # Special case for bytes_loaded (64-bit)
            if hasattr(self.lib, 'get_bytes_loaded'):
                self.lib.get_bytes_loaded.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)]
                self.lib.get_bytes_loaded.restype = None
            
            # reset_bnb_counters() -> bool
            if hasattr(self.lib, 'reset_bnb_counters'):
                self.lib.reset_bnb_counters.argtypes = []
                self.lib.reset_bnb_counters.restype = ctypes.c_bool
                
        except Exception as e:
            print(f"Function prototype setup failed: {e}")
            self.instrumentation_available = False
    
    def test_instrumentation(self) -> bool:
        """Test if counter instrumentation is working"""
        print("Testing counter instrumentation...")
        
        if not self.instrumentation_available:
            print("Instrumentation not available")
            return False
        
        if not self.lib:
            print("Library not loaded")
            return False
        
        # Check if key functions exist
        key_functions = ['extract_bnb_counter_data', 'reset_bnb_counters']
        for func in key_functions:
            if not hasattr(self.lib, func):
                print(f"Function {func} not found")
                return False
        
        # Try to reset counters
        try:
            self.lib.reset_bnb_counters()
            print("Successfully called reset_bnb_counters")
            return True
        except Exception as e:
            print(f"Failed to call functions: {e}")
            return False
    
    def extract_counter_data(self):
        """Extract counter data from GPU to CPU with enhanced error handling"""
        if not self.instrumentation_available:
            return
            
        if self.lib and hasattr(self.lib, 'extract_bnb_counter_data'):
            try:
                self.lib.extract_bnb_counter_data()
                
                # Optional: Verify extraction worked by spot-checking a counter
                test_count = self.get_nf4_lookup_count()
                if test_count < 0:
                    print("Warning: Counter extraction may have failed - negative values detected")
                    
            except Exception as e:
                print(f"Counter extraction failed: {e}")
        else:
            print("extract_bnb_counter_data function not available")
    
    def get_nf4_lookup_count(self) -> int:
        """Get NF4 lookup operation count"""
        if not self.instrumentation_available:
            return 0
            
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
        if not self.instrumentation_available:
            return 0
            
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
        if not self.instrumentation_available:
            return 0
            
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
        if not self.instrumentation_available:
            return 0
            
        if not self.lib or not hasattr(self.lib, 'get_kernel_call_count'):
            return 0
        
        try:
            count = ctypes.c_uint()
            self.lib.get_kernel_call_count(ctypes.byref(count))
            return count.value
        except Exception:
            return 0

    def get_warp_divergence_count(self) -> int:
        """Get warp divergence count - tracks branch divergence in warps"""
        if not self.instrumentation_available:
            return 0
            
        if not self.lib or not hasattr(self.lib, 'get_warp_divergence_count'):
            return 0
        
        try:
            count = ctypes.c_uint()
            self.lib.get_warp_divergence_count(ctypes.byref(count))
            return count.value
        except Exception:
            return 0

    def get_cache_line_loads(self) -> int:
        """Get cache line load count - tracks unique 128-byte memory chunks loaded"""
        if not self.instrumentation_available:
            return 0
            
        if not self.lib or not hasattr(self.lib, 'get_cache_line_loads'):
            return 0
        
        try:
            count = ctypes.c_uint()
            self.lib.get_cache_line_loads(ctypes.byref(count))
            return count.value
        except Exception:
            return 0

    def get_bytes_loaded(self) -> int:
        """Get total bytes loaded from memory"""
        if not self.instrumentation_available:
            return 0
            
        if not self.lib or not hasattr(self.lib, 'get_bytes_loaded'):
            return 0
        
        try:
            count = ctypes.c_ulonglong()
            self.lib.get_bytes_loaded(ctypes.byref(count))
            return count.value
        except Exception:
            return 0

    def get_coalesced_loads(self) -> int:
        """Get coalesced load count - efficient memory access patterns"""
        if not self.instrumentation_available:
            return 0
            
        if not self.lib or not hasattr(self.lib, 'get_coalesced_loads'):
            return 0
        
        try:
            count = ctypes.c_uint()
            self.lib.get_coalesced_loads(ctypes.byref(count))
            return count.value
        except Exception:
            return 0

    def get_scattered_loads(self) -> int:
        """Get scattered load count - inefficient memory access patterns"""
        if not self.instrumentation_available:
            return 0
            
        if not self.lib or not hasattr(self.lib, 'get_scattered_loads'):
            return 0
        
        try:
            count = ctypes.c_uint()
            self.lib.get_scattered_loads(ctypes.byref(count))
            return count.value
        except Exception:
            return 0

    def get_comprehensive_counter_report(self) -> Dict[str, int]:
        """Get comprehensive counter report including advanced metrics"""
        
        if not self.instrumentation_available:
            return {
                'nf4_lookup_count': 0,
                'scaling_factor_count': 0,
                'memory_access_count': 0,
                'kernel_call_count': 0,
                'warp_divergence_count': 0,
                'cache_line_loads': 0,
                'bytes_loaded': 0,
                'coalesced_loads': 0,
                'scattered_loads': 0,
                'total_operations': 0,
                'divergence_rate': 0,
                'coalescing_efficiency': 0,
                'accesses_per_cache_line': 0,
                'instrumentation_available': False
            }
        
        # Extract latest counter data
        self.extract_counter_data()
        
        # Get all basic counts
        report = {
            'nf4_lookup_count': self.get_nf4_lookup_count(),
            'scaling_factor_count': self.get_scaling_factor_count(),
            'memory_access_count': self.get_memory_access_count(),
            'kernel_call_count': self.get_kernel_call_count(),
            'warp_divergence_count': self.get_warp_divergence_count(),
            'cache_line_loads': self.get_cache_line_loads(),
            'bytes_loaded': self.get_bytes_loaded(),
            'coalesced_loads': self.get_coalesced_loads(),
            'scattered_loads': self.get_scattered_loads(),
            'instrumentation_available': True
        }
        
        # Calculate total operations
        total_ops = (report['nf4_lookup_count'] + 
                    report['scaling_factor_count'] + 
                    report['memory_access_count'])
        report['total_operations'] = total_ops
        
        # Calculate divergence rate
        if report['kernel_call_count'] > 0:
            total_warps = report['kernel_call_count'] * (512 / 32)  # Assume 512 threads per kernel
            if total_warps > 0:
                divergence_rate = (report['warp_divergence_count'] / total_warps) * 100
                report['divergence_rate'] = min(divergence_rate, 100.0)  # Cap at 100%
            else:
                report['divergence_rate'] = 0
        else:
            report['divergence_rate'] = 0
        
        # Calculate memory coalescing efficiency
        total_loads = report['coalesced_loads'] + report['scattered_loads']
        if total_loads > 0:
            report['coalescing_efficiency'] = (report['coalesced_loads'] / 
                                            total_loads) * 100
        else:
            report['coalescing_efficiency'] = 0
        
        # Calculate cache efficiency (accesses per cache line)
        if report['cache_line_loads'] > 0:
            report['accesses_per_cache_line'] = (report['scaling_factor_count'] / 
                                                report['cache_line_loads'])
        else:
            report['accesses_per_cache_line'] = 0
        
        return report

    def reset_counters(self) -> bool:
        """Reset all operation counters with enhanced verification for large models"""
        if not self.instrumentation_available:
            print("Counter reset not available - no instrumentation")
            return False
            
        if self.lib and hasattr(self.lib, 'reset_bnb_counters'):
            try:
                import torch
                import time
                if not torch.cuda.is_available():
                    return False
                
                # Save current device
                current_device = torch.cuda.current_device()
                
                # Clear GPU caches before reset
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device=current_device)
                
                # Call C++ reset function
                reset_success = bool(self.lib.reset_bnb_counters())
                
                if not reset_success:
                    print("C++ counter reset failed")
                    return False
                
                # Additional verification
                torch.cuda.synchronize()
                time.sleep(0.1)
                
                # Verify reset worked by checking counters
                self.extract_counter_data()
                
                nf4_count = self.get_nf4_lookup_count()
                scaling_count = self.get_scaling_factor_count()
                memory_count = self.get_memory_access_count()
                kernel_count = self.get_kernel_call_count()
                
                total_remaining = nf4_count + scaling_count + memory_count + kernel_count
                
                if total_remaining == 0:
                    print("Counter reset successful - all counters at zero")
                    return True
                else:
                    print(f"Reset incomplete: {total_remaining:,} ops remaining")
                    print(f"NF4: {nf4_count:,}, Scaling: {scaling_count:,}")
                    print(f"Memory: {memory_count:,}, Kernels: {kernel_count:,}")
                    return False
                    
            except Exception as e:
                print(f"Counter reset error: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print("reset_bnb_counters function not available")
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
            print(f"GPU memory clear failed: {e}")
        return False
    
    def diagnose_counter_state(self):
        """Diagnose counter state issues for large models"""
        print("\nCOUNTER STATE DIAGNOSIS")
        print("=" * 40)
        
        if not self.instrumentation_available:
            print("Instrumentation not available - no counters to diagnose")
            return
        
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
                    print("\nKernel count matches single batch pattern")
                elif kernel_count % 864 == 0:
                    print(f"\nKernel count is {kernel_count // 864}x base - accumulation detected!")
                else:
                    print(f"\nUnusual kernel count: {kernel_count}")
                    
            # Test reset capability
            print("\nTesting reset capability...")
            original_count = counters.get('total_operations', 0)
            
            self.lib.reset_bnb_counters()
            torch.cuda.synchronize()
            self.extract_counter_data()
            
            new_counters = self.get_comprehensive_counter_report()
            new_count = new_counters.get('total_operations', 0)
            
            if new_count < original_count:
                print(f"Partial reset achieved: {original_count:,} -> {new_count:,}")
            elif new_count == original_count:
                print(f"Reset had no effect: still {new_count:,}")
            else:
                print(f"Unexpected: count increased {original_count:,} -> {new_count:,}")
                
        except Exception as e:
            print(f"Diagnosis failed: {e}")
            import traceback
            traceback.print_exc()


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
        print(f"Counter instrumentation test failed: {e}")
        return False

def print_counter_summary():
    """Print a summary of operation counters including advanced metrics"""
    
    print("BITSANDBYTES OPERATION COUNTER SUMMARY")
    print("=" * 50)
    
    try:
        extractor = get_counter_extractor()
        report = extractor.get_comprehensive_counter_report()
        
        if not report.get('instrumentation_available', False):
            print("Instrumentation not available - compiled without BNB_ENABLE_INSTRUMENTATION")
            print("Using PyTorch profiler only mode")
            return
        
        print(f"Basic Operation Counts:")
        print(f"  NF4 lookups: {report['nf4_lookup_count']:,}")
        print(f"  Scaling factor accesses: {report['scaling_factor_count']:,}")
        print(f"  Memory accesses: {report['memory_access_count']:,}")
        print(f"  Kernel calls: {report['kernel_call_count']:,}")
        print(f"  Total operations: {report['total_operations']:,}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Branch divergence rate: {report['divergence_rate']:.1f}%")
        print(f"  Memory coalescing efficiency: {report['coalescing_efficiency']:.1f}%")
        print(f"  Cache line loads: {report['cache_line_loads']:,}")
        print(f"  Accesses per cache line: {report['accesses_per_cache_line']:.1f}")
        print(f"  Total bytes loaded: {report['bytes_loaded']:,} ({report['bytes_loaded']/1e9:.2f} GB)")
        
        if report['kernel_call_count'] > 0:
            print(f"\nAverages per kernel call:")
            print(f"  NF4 lookups/kernel: {report['nf4_lookup_count'] / report['kernel_call_count']:,.1f}")
            print(f"  Scaling accesses/kernel: {report['scaling_factor_count'] / report['kernel_call_count']:,.1f}")
            print(f"  Memory accesses/kernel: {report['memory_access_count'] / report['kernel_call_count']:,.1f}")
            print(f"  Bytes/kernel: {report['bytes_loaded'] / report['kernel_call_count']:,.0f}")
        
        # Identify optimization opportunities
        print(f"\nOptimization Opportunities:")
        if report['divergence_rate'] > 10:
            print(f"  High branch divergence ({report['divergence_rate']:.1f}%) - constant memory will help!")
        if report['coalescing_efficiency'] < 70:
            print(f"  Poor memory coalescing ({report['coalescing_efficiency']:.1f}%) - vectorized access needed!")
        if report['accesses_per_cache_line'] < 4:
            print(f"  Inefficient cache usage ({report['accesses_per_cache_line']:.1f} accesses/line) - shared memory will help!")
        
    except Exception as e:
        print(f"Failed to extract counter data: {e}")

if __name__ == "__main__":
    print("Testing BitsAndBytes counter extraction...")
    
    if test_instrumentation():
        print("Counter instrumentation test passed!")
        print_counter_summary()
    else:
        print("Counter instrumentation test failed!")
        print("\nTroubleshooting:")
        print("1. Make sure you compiled with BNB_ENABLE_INSTRUMENTATION=1")
        print("2. Check that instrumentation message appeared during compilation")
        print("3. Run some quantized operations first to generate counter data")
        print("4. Profiler will fall back to PyTorch-only mode")