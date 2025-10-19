"""Precision selection and energy metrics."""
from dataclasses import dataclass
from typing import Tuple, Optional
from contextlib import nullcontext
import torch
import logging

@dataclass
class EnergyMetrics:
    """Energy and resource metrics for a precision tier."""
    fp_tier: str
    power_saved_percent: float
    memory_saved_percent: float
    relative_speed: float
    cumulative_energy_saved: float

class PrecisionManager:
    """Manages FP precision selection and provides execution contexts."""
    
    # Precision tiers and their characteristics
    TIERS = {
        'fp4': {
            'power_saved_percent': 65.0,
            'memory_saved_percent': 75.0,
            'relative_speed': 2.2
        },
        'fp8': {
            'power_saved_percent': 55.0,
            'memory_saved_percent': 60.0,
            'relative_speed': 2.0
        },
        'fp16': {
            'power_saved_percent': 45.0,
            'memory_saved_percent': 50.0,
            'relative_speed': 1.5
        },
        'fp32': {  # Baseline
            'power_saved_percent': 0.0,
            'memory_saved_percent': 0.0,
            'relative_speed': 1.0
        }
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_fp_tier = 'fp32'
        
    def select_precision(self, complexity: float) -> Tuple[str, EnergyMetrics]:
        """Select most efficient precision tier for given complexity.
        
        Args:
            complexity: Float in [0,1] indicating computational complexity
            
        Returns:
            Tuple of (precision_tier, energy_metrics)
        """
        # Map complexity ranges to precision tiers
        if complexity <= 0.05:
            tier = 'fp4'
        elif complexity <= 0.2:
            tier = 'fp8'
        elif complexity <= 0.7:
            tier = 'fp16'
        else:
            tier = 'fp32'
            
        # Get characteristics for selected tier
        chars = self.TIERS[tier]
        metrics = EnergyMetrics(
            fp_tier=tier,
            power_saved_percent=chars['power_saved_percent'],
            memory_saved_percent=chars['memory_saved_percent'],
            relative_speed=chars['relative_speed'],
            cumulative_energy_saved=0.0
        )
        
        return tier, metrics
        
    def get_execution_context(self, tier: Optional[str] = None):
        """Get appropriate execution context for precision tier.
        
        Args:
            tier: Precision tier ('fp4','fp8','fp16','fp32'). If None, uses current.
            
        Returns:
            Context manager for running operations at specified precision.
        """
        if tier is None:
            tier = self.current_fp_tier
            
        if not torch.cuda.is_available():
            return nullcontext()
            
        # FP4/FP8 fall back to FP16 since they need special runtime support
        if tier in ('fp4', 'fp8'):
            self.logger.warning(
                f"Requested {tier} execution, but low-bit precision requires "
                "specialized runtime support. Falling back to fp16 for safety."
            )
            try:
                return torch.amp.autocast('cuda', dtype=torch.float16)
            except Exception:
                return nullcontext()
                
        # Native FP16 support via autocast
        if tier == 'fp16':
            try:
                return torch.amp.autocast('cuda', dtype=torch.float16)
            except Exception:
                return nullcontext()
                
        # FP32 default
        return nullcontext()
        
    def compute_energy_saved(
        self,
        duration: float,
        power_baseline: float,
        power_actual: Optional[float],
        metrics: EnergyMetrics
    ) -> float:
        """Compute energy (Joules) saved during workload execution.
        
        Args:
            duration: Workload duration in seconds
            power_baseline: Baseline power draw (watts)
            power_actual: Actual power measured (watts), if available
            metrics: EnergyMetrics for the precision tier used
            
        Returns:
            Joules saved (estimated or measured)
        """
        # Use actual power if available, else estimate from baseline
        if power_actual is not None:
            power_used = power_actual
        else:
            # Estimate from baseline using saved_percent
            power_used = power_baseline * (1.0 - metrics.power_saved_percent/100.0)
            
        # Compute energy saved
        baseline_joules = power_baseline * duration
        actual_joules = power_used * duration
        joules_saved = max(0.0, baseline_joules - actual_joules)
        
        # Update cumulative savings
        metrics.cumulative_energy_saved += joules_saved
        
        return joules_saved