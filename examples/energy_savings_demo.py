"""Energy savings demonstration using the GPU governor."""
import torch
import time
from src.core import GPUGovernor
from src.utils import format_joules, format_watts

def demonstrate_energy_savings():
    """Run workloads at different complexity levels and measure energy savings."""
    governor = GPUGovernor()
    print("\nGPU Energy Conservation Demo")
    print("==========================")
    
    test_cases = [
        ("Simple matrix multiply (low complexity)", "matrix 512x512 batch_size=1"),
        ("Medium batch processing", "matrix 1024x1024 batch_size=8 iterations=10"),
        ("Complex transformer (high precision)", "large transformer model batch_size=16 high precision")
    ]
    
    total_energy_saved = 0.0
    total_duration = 0.0
    
    for desc, prompt in test_cases:
        print(f"\nTest Case: {desc}")
        print(f"Prompt: {prompt}")
        
        # Let governor analyze and select precision
        fp_tier, energy_metrics = governor.optimize_for_prompt(prompt)
        
        # Run test workload
        size = 1024 if "simple" in desc.lower() else 2048
        with governor.get_execution_context():
            A = torch.randn(size, size, device='cuda')
            B = torch.randn(size, size, device='cuda')
            
            start = time.time()
            C = torch.matmul(A, B)
            torch.cuda.synchronize()
            duration = time.time() - start
        
        # Get metrics and compute energy saved
        runtime_metrics = governor.get_current_metrics()
        joules_saved = governor.compute_energy_saved(duration, energy_metrics)
        
        # Update totals
        total_energy_saved += joules_saved
        total_duration += duration
        
        # Report results
        print("\nResults:")
        print(f"Selected Precision: {fp_tier}")
        print(f"GPU Utilization: {runtime_metrics.get('utilization', 'n/a')}")
        print(f"Memory Usage: {runtime_metrics.get('memory_used', 'n/a')}")
        if 'power' in runtime_metrics:
            print(f"Power Draw: {runtime_metrics['power']}")
        print(f"Execution Time: {duration*1000:.1f}ms")
        print(f"Energy Saved: {format_joules(joules_saved)}")
        
        # Cleanup
        del A, B, C
        torch.cuda.empty_cache()
        time.sleep(1)  # Let GPU cool between tests
    
    # Final summary
    print("\nFinal Energy Analysis")
    print("====================")
    print(f"Total Runtime: {total_duration:.1f}s")
    print(f"Total Energy Saved: {format_joules(total_energy_saved)}")
    history = governor.get_optimization_history()
    print(f"Average Complexity: {history.get('average_complexity', 0.0):.2f}")
    
if __name__ == "__main__":
    demonstrate_energy_savings()