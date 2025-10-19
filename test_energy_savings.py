
def demonstrate_energy_savings():
    governor = GPUGovernor()
    print("\nEnergy Conservation Demo")
    print("======================")
    
    # Test cases with different complexity levels
    test_cases = [
        ("Simple matrix multiply (2x2)", 0.1),
        ("Medium batch processing", 0.5),
        ("Complex transformer inference", 0.9)
    ]
    
    total_energy_saved = 0.0
    baseline_energy = 0.0
    
    for desc, complexity in test_cases:
        print(f"\nTesting: {desc}")
        print(f"Complexity Score: {complexity:.2f}")
        
        # Let governor select precision (returns tier and EnergyMetrics)
        fp_result = governor.apply_fp_for_workload(complexity)
        if isinstance(fp_result, tuple):
            fp_tier, energy_metrics = fp_result
        else:
            fp_tier = fp_result
            energy_metrics = None
        
        # Create sample workload (matrix multiplication)
        size = 1024 if complexity < 0.5 else 2048
        with governor.fp_precision_context():
            A = torch.randn(size, size, device='cuda')
            B = torch.randn(size, size, device='cuda')
            
            start = time.time()
            C = torch.matmul(A, B)
            torch.cuda.synchronize()
            duration = time.time() - start
        
        # Get runtime metrics
        runtime_metrics = governor.get_current_metrics()
        history = governor.monitoring_history[-1]  # Get last record

        print(f"Selected Precision: {fp_tier}")
        print(f"GPU Utilization: {runtime_metrics.get('utilization', 'n/a')}")
        print(f"Memory Usage: {runtime_metrics.get('memory_used', 'n/a')}")
        print(f"Estimated Power Savings: {history.get('power_saved_percent', 0.0):.1f}%")
import torch
import time
import pytest
from src.core.gpu_controller import GPUGovernor

@pytest.mark.parametrize("desc,complexity,expected_fp", [
    ("Simple matrix multiply (2x2)", 0.1, "fp8"),
    ("Medium batch processing", 0.5, "fp16"),
    ("Complex transformer inference", 0.9, "fp32"),
])
def test_energy_savings_fp_selection(desc, complexity, expected_fp):
    governor = GPUGovernor()
    fp_tier, energy_metrics = governor.apply_fp_for_workload(complexity)
    assert fp_tier == expected_fp, f"Expected {expected_fp}, got {fp_tier} for {desc}"
    assert energy_metrics.power_saved_percent >= 0.0
    # Simulate workload
    size = 1024 if complexity < 0.5 else 2048
    with governor.fp_precision_context():
        A = torch.randn(size, size, device='cuda')
        B = torch.randn(size, size, device='cuda')
        start = time.time()
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        duration = time.time() - start
    runtime_metrics = governor.get_current_metrics()
    assert 'utilization' in runtime_metrics
    assert 'memory_used' in runtime_metrics