"""Main GPU governor coordinating complexity analysis, precision selection, and monitoring."""
import time
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import torch
try:
    import pynvml
except ImportError:
    pynvml = None

from .analyzer import ComplexityAnalyzer
from .precision import PrecisionManager, EnergyMetrics
from .telemetry import TelemetryManager


@dataclass
class GPUState:
    """Represents current GPU state metrics"""
    utilization: float
    memory_used: float
    memory_total: float
    temperature: Optional[float]
    power_draw: Optional[float]
    clock_speed: Optional[int]
    performance_state: str


class GPUGovernor:
    """Energy-efficient GPU governor that automatically selects optimal floating-point 
    precision based on prompt complexity analysis. The primary goal is to minimize 
    energy consumption while maintaining acceptable accuracy.
    
    Energy Savings Approach:
    1. Analyze prompt complexity to determine computational requirements
    2. Select most efficient precision that maintains accuracy:
       - FP32: Baseline (100% power use)
       - FP16: ~45-55% power savings
       - BF16: ~40-50% power savings
    3. Track and report energy/resource savings
    """

    def __init__(self, gpu_id: int = 0):
        self.logger = logging.getLogger(__name__)
        self.gpu_id = gpu_id
        
        # Initialize components
        self.analyzer = ComplexityAnalyzer()
        self.precision = PrecisionManager()
        self.telemetry = TelemetryManager(gpu_id)
        
        self.current_fp_tier = 'fp32'
        self.monitoring_history = []
        self.energy_metrics = []  # Track energy savings over time
        self.baseline_power = 100.0  # Assumed baseline power draw for FP32 (watts)

    def _get_gpu_state(self) -> Optional[GPUState]:
        """Get current GPU state through telemetry manager."""
        return self.telemetry.get_gpu_state()

    def _determine_performance_state(self, utilization: float) -> str:
        """Determine performance state based on utilization."""
        return self.telemetry.determine_performance_state(utilization)

    def optimize_for_workload(self, workload_complexity: float):
        """Apply runtime optimizations based on workload complexity."""
        try:
            self.precision.optimize_for_workload(workload_complexity)

            state = self._get_gpu_state()
            if state:
                self.monitoring_history.append({
                    'timestamp': time.time(),
                    'complexity': float(workload_complexity),
                'state': state
            })
            self.logger.info(f"Optimized for complexity={workload_complexity:.2f}")
        except Exception as e:
            self.logger.error(f"Error during optimization: {e}")

    def get_current_metrics(self) -> Dict[str, str]:
        state = self._get_gpu_state()
        if state is None:
            return {}
        return {
            'utilization': f"{state.utilization:.1f}%",
            'memory_used': f"{state.memory_used:.0f}MB / {state.memory_total:.0f}MB",
            'performance_state': state.performance_state,
            'memory_utilization': f"{(state.memory_used/state.memory_total)*100:.1f}%"
        }

    def get_optimization_history(self) -> Dict:
        if not self.monitoring_history:
            return {}
        import numpy as _np
        arr = _np.array([(x['timestamp'], x['complexity']) for x in self.monitoring_history])
        return {
            'timestamps': arr[:,0].tolist(),
            'complexity_values': arr[:,1].tolist(),
            'total_optimizations': len(self.monitoring_history),
            'average_complexity': float(_np.mean(arr[:,1]))
        }

    def cleanup(self):
        try:
            torch.cuda.empty_cache()
            self.logger.info("GPU resources cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    # --- Floating point precision mapping and context ---
    def map_complexity_to_fp(self, complexity: float) -> Tuple[str, EnergyMetrics]:
        """Map complexity score to most energy-efficient FP precision.
        
        Returns:
            Tuple[str, EnergyMetrics]: Selected FP tier and estimated energy savings
        
        Strategy:
        - Low complexity -> FP16 (maximum energy savings)
        - Medium complexity -> FP16 (balanced)
        - High complexity -> FP32 (when accuracy critical)
        """
        # New tiering: FP4 < FP8 < FP16 < FP32
        # Note: FP4/FP8 support requires specialized hardware or custom kernels. Here
        # we prefer FP4/FP8 for the lowest-complexity prompts but implement safe
        # fallbacks (run in FP16) when the runtime does not support them.
        if complexity <= 0.05:
            # Extremely simple: FP4 (best energy savings) - simulated if unsupported
            metrics = EnergyMetrics(
                fp_tier='fp4',
                power_saved_percent=65.0,  # Aggressive estimate
                memory_saved_percent=75.0,  # Small footprint
                relative_speed=2.2,
                cumulative_energy_saved=0.0
            )
            return 'fp4', metrics

        if complexity <= 0.2:
            # Very low complexity: FP8
            metrics = EnergyMetrics(
                fp_tier='fp8',
                power_saved_percent=55.0,
                memory_saved_percent=60.0,
                relative_speed=2.0,
                cumulative_energy_saved=0.0
            )
            return 'fp8', metrics

        if complexity <= 0.7:
            # Let precision manager determine optimal precision
            tier, metrics = self.precision.analyze_complexity(float(complexity))
            return tier, metrics

    def apply_fp_for_workload(self, complexity: float) -> Tuple[str, EnergyMetrics]:
        """Decide and record FP tier for upcoming workload."""
        fp_tier, metrics = self.precision.analyze_complexity(float(complexity))
        self.current_fp_tier = fp_tier
        
        # Update energy savings with current telemetry
        metrics = self.telemetry.update_energy_metrics(metrics)
            
        # Record decision and metrics
        self.monitoring_history.append({
            'timestamp': time.time(),
            'complexity': float(complexity),
            'chosen_fp': fp_tier,
            'power_saved_percent': metrics.power_saved_percent,
            'memory_saved_percent': metrics.memory_saved_percent,
            'energy_saved': metrics.cumulative_energy_saved
        })
        
        self.energy_metrics.append(metrics)
        self.logger.info(
            f"Selected {fp_tier} for complexity={complexity:.2f}, "
            f"estimated power savings: {metrics.power_saved_percent:.1f}%"
        )
        return fp_tier, metrics

    def compute_joules_saved(self, duration_seconds: float, metrics: EnergyMetrics) -> float:
        """Compute estimated energy savings using telemetry data."""
        return self.telemetry.compute_energy_savings(duration_seconds, metrics)

    def fp_precision_context(self):
        """Return a context manager for the current precision tier."""
        return self.precision.get_context(self.current_fp_tier)
    def collect_garbage(self):
        """Release unused GPU memory."""
        self.telemetry.collect_garbage()

    def optimize_for_prompt(self, prompt_text: str) -> Tuple[str, Dict]:
        """Entry point for per-prompt optimization."""
        # Analyze prompt complexity 
        score = self.analyzer.analyze_prompt(prompt_text)
        
        # Get precision tier and energy savings metrics
        fp_tier, metrics = self.apply_fp_for_workload(score)
        
        # Return optimization results
        return fp_tier, {
            'complexity_score': score,
            'precision_tier': fp_tier,
            'power_savings': metrics.power_saved_percent,
            'memory_savings': metrics.memory_saved_percent,
            'speed_multiplier': metrics.relative_speed
        }

    def get_status(self) -> Dict:
        """Get current state and optimization history."""
        current = self._get_gpu_state()
        energy_saved = sum(m.cumulative_energy_saved for m in self.energy_metrics)
        
        return {
            'gpu_state': current,
            'current_fp': self.current_fp_tier,
            'history_len': len(self.monitoring_history),
            'total_energy_saved': energy_saved
        }

    def summarize_energy_savings(self) -> Dict[str, float]:
        """Get cumulative energy savings statistics."""
        if not self.monitoring_history:
            return {
                'avg_power_saved_percent': 0.0,
                'avg_memory_saved_percent': 0.0,
                'total_joules_saved': 0.0
            }

        avg_power = sum(m['power_saved_percent'] for m in self.monitoring_history) / len(self.monitoring_history)
        avg_memory = sum(m['memory_saved_percent'] for m in self.monitoring_history) / len(self.monitoring_history)
        total_energy = sum(m['energy_saved'] for m in self.monitoring_history)

        return {
            'avg_power_saved_percent': avg_power,
            'avg_memory_saved_percent': avg_memory,
            'total_joules_saved': total_energy
        }
