"""Integration hooks for AI tools to automatically optimize GPU usage."""
import torch
import functools
import logging
from typing import Callable, Any, Dict
from ..core import GPUGovernor

class AIToolOptimizer:
    """Automatic GPU optimization for AI tools and frameworks."""
    
    def __init__(self):
        self.governor = GPUGovernor()
        self.logger = logging.getLogger(__name__)
        self.optimization_enabled = True
        
    def auto_optimize(self, complexity_hint: str = None):
        """Decorator to automatically optimize GPU for AI tool functions."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.optimization_enabled:
                    return func(*args, **kwargs)
                
                # Analyze function complexity
                complexity = self._analyze_function_complexity(func, args, kwargs, complexity_hint)
                
                # Apply optimization
                self.governor.optimize_for_workload(complexity)
                fp_tier = self.governor.apply_fp_for_workload(complexity)
                
                self.logger.info(f"Auto-optimized {func.__name__} with {fp_tier.upper()} precision")
                
                # Execute with optimization context
                with self.governor.fp_precision_context():
                    result = func(*args, **kwargs)
                
                return result
            return wrapper
        return decorator
    
    def _analyze_function_complexity(self, func: Callable, args: tuple, kwargs: dict, hint: str) -> float:
        """Analyze the complexity of a function call."""
        # Use hint if provided
        if hint:
            return self.governor.analyzer.analyze_complexity(hint)
        
        # Analyze based on function name and arguments
        func_name = func.__name__.lower()
        
        # High complexity indicators
        if any(keyword in func_name for keyword in ['train', 'fine_tune', 'generate_large', 'process_batch']):
            return 0.8
        
        # Medium complexity indicators
        if any(keyword in func_name for keyword in ['infer', 'predict', 'encode', 'decode']):
            return 0.5
        
        # Analyze tensor sizes if present
        for arg in args:
            if isinstance(arg, torch.Tensor):
                size = arg.numel()
                if size > 1000000:  # > 1M elements
                    return 0.7
                elif size > 100000:  # > 100K elements
                    return 0.4
        
        return 0.3  # Default low complexity
    
    def optimize_for_ai_workload(self, workload_type: str, **params):
        """Optimize for specific AI workload types."""
        workload_complexity_map = {
            'text_generation': 0.4,
            'image_generation': 0.7,
            'model_training': 0.9,
            'fine_tuning': 0.8,
            'inference': 0.3,
            'embedding': 0.5,
            'translation': 0.4,
            'summarization': 0.3,
            'question_answering': 0.4,
            'classification': 0.3
        }
        
        complexity = workload_complexity_map.get(workload_type, 0.5)
        
        # Adjust based on parameters
        if 'batch_size' in params and params['batch_size'] > 8:
            complexity += 0.1
        if 'sequence_length' in params and params['sequence_length'] > 512:
            complexity += 0.1
        if 'image_size' in params and params['image_size'] > 512:
            complexity += 0.2
        
        complexity = min(1.0, complexity)
        
        self.governor.optimize_for_workload(complexity)
        return self.governor.apply_fp_for_workload(complexity)
    
    def get_energy_savings_summary(self) -> Dict[str, Any]:
        """Get summary of energy savings from AI workloads."""
        total_saved = sum(self.governor.energy_metrics)
        history = self.governor.get_optimization_history()
        
        return {
            'total_energy_saved_joules': total_saved,
            'total_workloads_optimized': len(self.governor.energy_metrics),
            'average_energy_saved_per_workload': total_saved / max(1, len(self.governor.energy_metrics)),
            'optimization_efficiency': history.get('average_complexity', 0.0),
            'current_precision_tier': self.governor.current_fp_tier
        }

# Global instance for easy access
ai_optimizer = AIToolOptimizer()

# Convenience decorator
def auto_optimize_gpu(complexity_hint: str = None):
    """Decorator to automatically optimize GPU for AI functions."""
    return ai_optimizer.auto_optimize(complexity_hint)
