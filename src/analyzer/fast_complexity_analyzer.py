import re
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import logging

@dataclass
class ComplexityScore:
    raw_score: float
    gpu_power_target: int      # Target power limit in watts
    gpu_util_target: int       # Target GPU utilization %
    memory_clock: int          # Memory clock in MHz
    core_clock: int           # Core clock in MHz
    complexity_level: str      # low/medium/high

class FastComplexityAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize pattern matchers for quick analysis
        self.complexity_patterns = {
            'matrix_ops': r'\b(matrix|matrices|multiply|multiplication)\b',
            'ml_ops': r'\b(train|learning|neural|network|deep|CNN|RNN|LSTM)\b',
            'data_size': r'\b(\d+)[xX×](\d+)\b|\b(\d+)\s*(GB|MB|K)\b',
            'iterations': r'\b(loop|iterate|epoch|batch|steps?)\b.*?(\d+)',
            'precision': r'\b(float64|double|float32|float16|half|int8|fp16|fp32)\b'
        }
        
        # GPU power profiles (example values, adjust based on your GPU)
        self.power_profiles = {
            'low': {
                'power_limit': 80,    # 80W
                'gpu_util': 60,       # 60%
                'memory_clock': 4000,  # 4000MHz
                'core_clock': 1200    # 1200MHz
            },
            'medium': {
                'power_limit': 150,   # 150W
                'gpu_util': 85,       # 85%
                'memory_clock': 6000, # 6000MHz
                'core_clock': 1600    # 1600MHz
            },
            'high': {
                'power_limit': 250,   # 250W
                'gpu_util': 100,      # 100%
                'memory_clock': 7500, # 7500MHz
                'core_clock': 1900    # 1900MHz
            }
        }

    def _analyze_data_size(self, prompt: str) -> float:
        """Quickly estimate data size complexity."""
        matches = re.findall(self.complexity_patterns['data_size'], prompt.lower())
        if not matches:
            return 0.3  # Default to low if no size specified
            
        max_size = 0
        for match in matches:
            if match[0] and match[1]:  # Matrix dimensions
                size = int(match[0]) * int(match[1])
                max_size = max(max_size, size)
            elif match[2] and match[3]:  # Memory size
                size_str = match[3].upper()
                size = int(match[2])
                if 'GB' in size_str:
                    size *= 1024 * 1024 * 1024
                elif 'MB' in size_str:
                    size *= 1024 * 1024
                elif 'K' in size_str:
                    size *= 1024
                max_size = max(max_size, size)
        
        # Normalize size score
        if max_size == 0:
            return 0.3
        elif max_size < 1024 * 1024:  # < 1MB
            return 0.3
        elif max_size < 1024 * 1024 * 1024:  # < 1GB
            return 0.6
        else:
            return 0.9

    def _analyze_operation_complexity(self, prompt: str) -> float:
        """Quickly estimate computational complexity."""
        prompt_lower = prompt.lower()
        
        # Count complex operations
        ml_ops = len(re.findall(self.complexity_patterns['ml_ops'], prompt_lower))
        matrix_ops = len(re.findall(self.complexity_patterns['matrix_ops'], prompt_lower))
        
        # Weight the operations
        score = (ml_ops * 0.4 + matrix_ops * 0.3) / 5.0  # Normalize to 0-1
        return min(score, 1.0)

    def _analyze_iterations(self, prompt: str) -> float:
        """Estimate complexity based on iteration counts."""
        matches = re.findall(self.complexity_patterns['iterations'], prompt.lower())
        if not matches:
            return 0.3
            
        max_iterations = 0
        for match in matches:
            if match[1].isdigit():
                max_iterations = max(max_iterations, int(match[1]))
        
        # Normalize iteration score
        if max_iterations < 100:
            return 0.3
        elif max_iterations < 1000:
            return 0.6
        else:
            return 0.9

    def analyze_prompt(self, prompt: str) -> ComplexityScore:
        """
        Quickly analyze prompt complexity and determine GPU power settings.
        This method is optimized for speed and efficiency.
        """
        try:
            # Fast complexity analysis
            data_score = self._analyze_data_size(prompt)
            op_score = self._analyze_operation_complexity(prompt)
            iter_score = self._analyze_iterations(prompt)
            
            # Weighted combination for final score
            final_score = (
                data_score * 0.4 +    # Data size is most important
                op_score * 0.35 +     # Operation complexity next
                iter_score * 0.25     # Iterations least important
            )
            
            # Determine complexity level and power profile
            if final_score < 0.4:
                profile = self.power_profiles['low']
                level = 'low'
            elif final_score < 0.7:
                profile = self.power_profiles['medium']
                level = 'medium'
            else:
                profile = self.power_profiles['high']
                level = 'high'
                
            return ComplexityScore(
                raw_score=final_score,
                gpu_power_target=profile['power_limit'],
                gpu_util_target=profile['gpu_util'],
                memory_clock=profile['memory_clock'],
                core_clock=profile['core_clock'],
                complexity_level=level
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing prompt complexity: {e}")
            # Return safe default values if analysis fails
            return ComplexityScore(
                raw_score=0.3,
                gpu_power_target=80,
                gpu_util_target=60,
                memory_clock=4000,
                core_clock=1200,
                complexity_level='low'
            )