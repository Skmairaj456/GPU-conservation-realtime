"""Complexity analysis for GPU workloads."""
import re
from typing import Dict, Optional

class ComplexityAnalyzer:
    """Analyzes input prompts to estimate computational complexity."""
    
    # Complexity indicators and their weights
    INDICATORS = {
        'matrix_ops': 0.3,  # Matrix operations (NxN)
        'batch_size': 0.2,  # Batch processing
        'iterations': 0.2,  # Loop/repeated operations
        'precision': 0.15,  # Precision requirements
        'memory': 0.15     # Memory intensity
    }
    
    def __init__(self):
        # Precompile regex patterns
        self.matrix_pattern = re.compile(r'(\d+)\s*[xX]\s*(\d+)')
        self.batch_pattern = re.compile(r'batch\s*[_-]?size\s*[=:]?\s*(\d+)', re.I)
        self.iter_pattern = re.compile(r'iterations?\s*[=:]?\s*(\d+)', re.I)
    
    def analyze(self, prompt: str) -> Dict[str, float]:
        """Analyze prompt and return complexity scores per component."""
        scores = {
            'matrix_ops': self._analyze_matrix_ops(prompt),
            'batch_size': self._analyze_batch_size(prompt),
            'iterations': self._analyze_iterations(prompt),
            'precision': self._analyze_precision_needs(prompt),
            'memory': self._analyze_memory_intensity(prompt)
        }
        return scores
    
    def estimate_complexity(self, prompt: str) -> float:
        """Estimate overall complexity score in [0,1]."""
        scores = self.analyze(prompt)
        weighted_sum = sum(
            score * self.INDICATORS[component]
            for component, score in scores.items()
        )
        return min(1.0, max(0.0, weighted_sum))
    
    def _analyze_matrix_ops(self, prompt: str) -> float:
        """Analyze matrix operation complexity."""
        matches = self.matrix_pattern.findall(prompt.lower())
        if not matches:
            return 0.0
        max_size = max(
            int(n) * int(m)
            for n, m in matches
        )
        # Normalize: assume 4096x4096 is max complexity
        return min(1.0, max_size / (4096 * 4096))
    
    def _analyze_batch_size(self, prompt: str) -> float:
        """Analyze batch processing complexity."""
        match = self.batch_pattern.search(prompt)
        if not match:
            return 0.0
        size = int(match.group(1))
        # Normalize: assume batch_size=32 is max complexity
        return min(1.0, size / 32.0)
    
    def _analyze_iterations(self, prompt: str) -> float:
        """Analyze iteration/loop complexity."""
        match = self.iter_pattern.search(prompt)
        if not match:
            return 0.0
        iters = int(match.group(1))
        # Normalize: assume 1000 iterations is max complexity
        return min(1.0, iters / 1000.0)
    
    def _analyze_precision_needs(self, prompt: str) -> float:
        """Analyze precision requirements."""
        prompt = prompt.lower()
        if 'high precision' in prompt or 'fp32' in prompt:
            return 1.0
        if 'mixed precision' in prompt or 'fp16' in prompt:
            return 0.5
        if 'low precision' in prompt or 'int8' in prompt:
            return 0.2
        return 0.5  # Default to medium precision needs
    
    def _analyze_memory_intensity(self, prompt: str) -> float:
        """Analyze memory intensity signals."""
        prompt = prompt.lower()
        score = 0.0
        memory_indicators = {
            'large': 0.8,
            'huge': 1.0,
            'memory intensive': 0.9,
            'small': 0.2,
            'tiny': 0.1
        }
        for indicator, value in memory_indicators.items():
            if indicator in prompt:
                score = max(score, value)
        return score or 0.5  # Default to medium if no signals