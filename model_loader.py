"""
Updated model_loader.py with proper profiling integration
"""

import os
import time
import torch
import logging
from typing import Tuple, Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Enhanced singleton loader for the model and tokenizer with profiling support.
    Ensures the model is only loaded once and reused across calls.
    """

    # Class-level variables
    _model: Optional[AutoModelForCausalLM] = None
    _tokenizer: Optional[AutoTokenizer] = None
    _profiler: Optional[Any] = None
    _profiling_enabled: bool = False

    @classmethod
    def load_model(cls, 
                enable_profiling: bool = False,
                profiling_config: Optional[Dict[str, Any]] = None) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Enhanced load_model method with profiling support
        """
        
        # If model is already loaded and profiling settings haven't changed, return cached
        if (cls._model is not None and cls._tokenizer is not None and 
            getattr(cls, '_profiling_enabled', False) == enable_profiling):
            logger.info("✅ Using cached model from memory")
            return cls._tokenizer, cls._model

        # Load fresh model (your existing code)
        logger.info("🔄 Loading model from disk...")
        start = time.time()

        load_dotenv()
        model_name = os.getenv("MODEL_NAME")

        if not model_name:
            raise EnvironmentError(
                "MODEL_NAME not found in .env or environment variables."
            )

        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer (your existing code)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Quantization configuration (your existing code)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Load model (your existing code)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            quantization_config=quant_config
        )

        # Cache the loaded model and settings
        cls._model, cls._tokenizer = model, tokenizer
        cls._profiling_enabled = enable_profiling

        logger.info("Model and tokenizer loaded and cached.")
        logger.info(f"Model loaded in {round(time.time() - start, 2)}s")

        # Enable profiling if requested
        if enable_profiling:
            cls._enable_profiling(profiling_config)
        
        return cls._tokenizer, cls._model

    @classmethod
    def _enable_profiling(cls, profiling_config: Optional[Dict[str, Any]] = None):
        """Enable profiling on the loaded model"""
        try:
            # Try to import enhanced profiler
            from transformers.utils.profiling import AsyncProfiler, enable_profiling_on_model
            
            # Set up full profiling (no sampling)
            if profiling_config is None:
                profiling_config = {
                    'sample_rate': 1.0,  # Full profiling
                    'adaptive_sampling': False,  # No adaptive sampling  
                    'max_operations_per_layer': float('inf'),  # No limits
                    'use_cuda_events': True
                }

            profiler = AsyncProfiler(
                model_name="qwen2_benchmark",
                sample_rate=profiling_config.get('sample_rate', 1.0),
                adaptive_sampling=profiling_config.get('adaptive_sampling', False),
                max_operations_per_layer=profiling_config.get('max_operations_per_layer', float('inf')),
                use_cuda_events=profiling_config.get('use_cuda_events', True)
            )

            if hasattr(cls._model, 'model'):
                cls._model.model.profiler = profiler
                enable_profiling_on_model(cls._model, profiler)
                logger.info("✅ Enhanced profiling enabled")
                cls._profiling_enabled = True
            else:
                logger.warning("Model structure doesn't support profiling")
                cls._profiling_enabled = False
                
        except ImportError:
            logger.warning("Enhanced profiling not available - check profiling.py location")
            cls._profiling_enabled = False
        except Exception as e:
            logger.error(f"Failed to enable profiling: {e}")
            cls._profiling_enabled = False

    @classmethod
    def load_model_with_profiling(cls, profiling_config: Optional[Dict[str, Any]] = None) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """Load model with enhanced profiling enabled"""
        return cls.load_model(enable_profiling=True, profiling_config=profiling_config)

    @classmethod
    def is_profiling_enabled(cls) -> bool:
        """Check if profiling is enabled"""
        return (cls._profiling_enabled and 
                cls._model is not None and 
                hasattr(cls._model, 'model') and 
                hasattr(cls._model.model, 'profiler') and
                cls._model.model.profiler is not None)

    @classmethod
    def get_profiling_results(cls) -> Optional[Dict[str, Any]]:
        """
        Get profiling results from the loaded model.
        
        Returns:
            Dict with profiling results or None if profiling not enabled
        """
        if not cls.is_profiling_enabled():
            logger.warning("Profiling not enabled or model not loaded")
            return None
        
        try:
            return cls._model.model.profiler.get_aggregated_stats()
        except Exception as e:
            logger.warning(f"Failed to get profiling results: {e}")
            return None

    @classmethod
    def reset_profiling_stats(cls):
        """Reset profiling statistics"""
        if not cls.is_profiling_enabled():
            return
        
        try:
            cls._model.model.profiler.reset_stats()
            logger.info("Profiling statistics reset")
        except Exception as e:
            logger.warning(f"Failed to reset profiling: {e}")

    @classmethod
    def export_profiling_results(cls, output_path: str, format: str = 'json') -> Optional[str]:
        """
        Export profiling results to file.
        
        Args:
            output_path: Path to save results
            format: Export format ('json' or 'csv')
            
        Returns:
            Path to exported file or None if failed
        """
        if not cls.is_profiling_enabled():
            logger.warning("Profiling not enabled or model not loaded")
            return None
        
        try:
            results = cls._model.model.profiler.get_aggregated_stats()
            
            import json
            from pathlib import Path
            
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Profiling results exported to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.warning(f"Failed to export profiling results: {e}")
            return None

    @classmethod
    def configure_profiling(cls, **kwargs):
        """
        Update profiling configuration for the loaded model.
        
        Args:
            **kwargs: Profiling configuration parameters
        """
        if not cls.is_profiling_enabled():
            logger.warning("Profiling not enabled or model not loaded")
            return
        
        try:
            profiler = cls._model.model.profiler
            
            # Update profiling settings
            if 'sample_rate' in kwargs:
                profiler.sample_rate = kwargs['sample_rate']
                logger.info(f"Updated sample rate to {kwargs['sample_rate']}")
            
            if 'adaptive_sampling' in kwargs:
                profiler.adaptive_sampling = kwargs['adaptive_sampling']
                logger.info(f"Updated adaptive sampling to {kwargs['adaptive_sampling']}")
            
            if 'max_operations_per_layer' in kwargs:
                profiler.max_operations_per_layer = kwargs['max_operations_per_layer']
                logger.info(f"Updated max operations per layer to {kwargs['max_operations_per_layer']}")
        
        except Exception as e:
            logger.warning(f"Failed to configure profiling: {e}")

    @classmethod
    def clear_cache(cls):
        """Clear the cached model and tokenizer"""
        cls._model = None
        cls._tokenizer = None
        cls._profiler = None
        cls._profiling_enabled = False
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model cache cleared")


# Simple profiling fallback
def enable_simple_profiling(model, tokenizer) -> bool:
    """Simple profiling enabler as a fallback"""
    try:
        # Create a minimal profiler
        class SimpleProfiler:
            def __init__(self):
                self.stats = {}
                
            def reset_stats(self):
                self.stats = {}
                
            def get_aggregated_stats(self):
                return {
                    'operations': {},
                    'summary': {'total_estimated_time_ms': 0, 'total_operations': 0}
                }
        
        # Attach simple profiler
        if hasattr(model, 'model'):
            model.model.profiler = SimpleProfiler()
            logger.info("✅ Simple profiler attached (fallback mode)")
            return True
        else:
            logger.warning("Cannot attach profiler - model structure not supported")
            return False
            
    except Exception as e:
        logger.error(f"Failed to enable simple profiling: {e}")
        return False


# Convenience functions for backward compatibility and ease of use

def load_model_with_profiling(profiling_config: Optional[Dict[str, Any]] = None) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Convenience function to load model with profiling enabled.
    
    Args:
        profiling_config: Optional profiling configuration
        
    Returns:
        Tuple of tokenizer and model
    """
    return ModelLoader.load_model(enable_profiling=True, profiling_config=profiling_config)


def load_model_without_profiling() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Convenience function to load model without profiling.
    
    Returns:
        Tuple of tokenizer and model
    """
    return ModelLoader.load_model(enable_profiling=False)


def create_profiling_config(sample_rate: float = 1.0,
                           adaptive_sampling: bool = True,
                           max_operations_per_layer: int = 1000) -> Dict[str, Any]:
    """
    Create a profiling configuration dictionary.
    
    Args:
        sample_rate: Base sampling rate (0.0 to 1.0)
        adaptive_sampling: Enable adaptive sampling based on operation importance
        max_operations_per_layer: Maximum operations to profile per layer
        
    Returns:
        Configuration dictionary
    """
    return {
        'sample_rate': sample_rate,
        'adaptive_sampling': adaptive_sampling,
        'max_operations_per_layer': max_operations_per_layer
    }


# Example usage configurations
PROFILING_PRESETS = {
    'development': create_profiling_config(
        sample_rate=1.0,
        adaptive_sampling=True,
        max_operations_per_layer=1000
    ),
    'production_monitoring': create_profiling_config(
        sample_rate=0.1,
        adaptive_sampling=True,
        max_operations_per_layer=100
    ),
    'minimal_overhead': create_profiling_config(
        sample_rate=0.05,
        adaptive_sampling=True,
        max_operations_per_layer=50
    ),
    'detailed_analysis': create_profiling_config(
        sample_rate=1.0,
        adaptive_sampling=False,  # Profile everything for detailed analysis
        max_operations_per_layer=5000
    )
}


def load_model_with_preset(preset_name: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load model with a predefined profiling preset.
    
    Args:
        preset_name: Name of the preset ('development', 'production_monitoring', 
                    'minimal_overhead', 'detailed_analysis')
                    
    Returns:
        Tuple of tokenizer and model
    """
    if preset_name not in PROFILING_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PROFILING_PRESETS.keys())}")
    
    config = PROFILING_PRESETS[preset_name]
    logger.info(f"Loading model with '{preset_name}' profiling preset")
    
    return ModelLoader.load_model(enable_profiling=True, profiling_config=config)


if __name__ == "__main__":
    # Example usage
    print("Testing ModelLoader with profiling...")
    
    # Test 1: Load with profiling
    tokenizer, model = load_model_with_profiling()
    print("✅ Model loaded with profiling")
    
    # Test 2: Get profiling results (should be empty initially)
    results = ModelLoader.get_profiling_results()
    print(f"📊 Initial profiling results: {len(results.get('operations', {})) if results else 0} operations")
    
    # Test 3: Load with preset
    tokenizer2, model2 = load_model_with_preset('minimal_overhead')
    print("✅ Model loaded with minimal_overhead preset")
    
    # Test 4: Clear cache
    ModelLoader.clear_cache()
    print("🗑️ Cache cleared")