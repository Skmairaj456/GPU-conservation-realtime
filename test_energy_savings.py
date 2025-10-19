import torch
from src.core.gpu_controller import GPUGovernor
import time

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
        print(f"Memory Savings: {history.get('memory_saved_percent', 0.0):.1f}%")
        print(f"Execution Time: {duration*1000:.1f}ms")

        # Compute Joules saved during this workload (if energy_metrics available)
        if energy_metrics is not None:
            joules = governor.compute_joules_saved(duration, energy_metrics)
        else:
            joules = 0.0
        total_energy_saved += joules
        baseline_energy += governor.baseline_power * duration
        # Cleanup
        del A, B, C
        torch.cuda.empty_cache()
        time.sleep(1)  # Let GPU cool between tests
    
    # Summary
    print("\nFinal Energy Analysis")
    print("===================")
    print(f"Baseline Energy (FP32): {baseline_energy:.1f} Joules")
    print(f"Total Energy Saved: {total_energy_saved:.1f} Joules")
    print(f"Energy Reduction: {(total_energy_saved/baseline_energy)*100:.1f}%")
    
if __name__ == "__main__":
    demonstrate_energy_savings()