import torch
import time
import logging
from src.core.gpu_controller import GPUGovernor
from src.core.complexity_analyzer import estimate_complexity
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_prompt_complexity(prompt: str) -> float:
    """
    Analyze the complexity of a task prompt.
    Returns a value between 0 and 1.
    """
    # Keywords indicating complexity
    high_complexity = ['train', 'deep', 'neural', '4k', 'high-resolution', 
                      'batch', 'parallel', 'real-time']
    medium_complexity = ['process', 'analyze', 'compute', 'matrix', 'transform']
    
    prompt_lower = prompt.lower()
    
    # Count complexity indicators
    high_count = sum(1 for word in high_complexity if word in prompt_lower)
    medium_count = sum(1 for word in medium_complexity if word in prompt_lower)
    
    # Extract numeric values (e.g., matrix sizes, iterations)
    import re
    numbers = [int(n) for n in re.findall(r'\d+', prompt)]
    size_factor = max(numbers) / 10000 if numbers else 0.3
    
    # Calculate complexity score
    complexity = (high_count * 0.2 + medium_count * 0.1 + size_factor * 0.5)
    return min(max(complexity, 0.1), 1.0)

def run_gpu_task(size: int, iterations: int):
    """Run a GPU task with specified parameters."""
    # Create test matrices
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    
    for i in range(iterations):
        # Matrix multiplication
        c = torch.matmul(a, b)
        
        # Additional operations
        c = torch.nn.functional.relu(c)
        if size > 1000:
            # FFT in fp16 may fail for non-power-of-two sizes on some GPUs.
            # Try performing FFT normally; on failure, run FFT in float32 with autocast disabled.
            try:
                c = torch.fft.fft2(c)
            except RuntimeError as e:
                msg = str(e).lower()
                if 'cufft' in msg or 'cufft' in msg or 'power of two' in msg:
                    # Fallback: perform FFT in float32 precision
                    from torch.cuda.amp import autocast
                    with autocast(enabled=False):
                        c = torch.fft.fft2(c.float())
                else:
                    raise
        
        # Force GPU synchronization
        torch.cuda.synchronize()
        
        if i % 10 == 0:
            print(f"Completed iteration {i}/{iterations}")
    
    del a, b, c
    torch.cuda.empty_cache()

def demonstrate_gpu_governor():
    """Demonstrate GPU Governor with different workloads."""
    governor = GPUGovernor()
    
    test_cases = [
        ("Process a small 500x500 matrix", 500, 50),
        ("Analyze 1000x1000 dataset with neural network", 1000, 100),
        ("Train deep learning model on 4K images", 2000, 200)
    ]
    
    print("\nGPU Governor Demonstration")
    print("=========================")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("This demo will show real-time GPU optimization for different workloads.")
    
    for prompt, size, iterations in test_cases:
        input(f"\nPress Enter to run: '{prompt}'...")

        # Analyze complexity and optimize GPU
        complexity = estimate_complexity(prompt)
        print(f"\nAnalyzed Complexity: {complexity:.2f}")

        # Apply GPU optimization and choose FP tier
        governor.optimize_for_workload(complexity)
        chosen_fp = governor.apply_fp_for_workload(complexity)
        print(f"Chosen FP tier: {chosen_fp}")

        # Show initial metrics
        print("\nInitial GPU State:")
        for metric, value in governor.get_current_metrics().items():
            print(f"{metric}: {value}")

        # Run the task
        print(f"\nRunning task...")
        start_time = time.time()
        # Run workload under precision context
        with governor.fp_precision_context():
            run_gpu_task(size, max(1, iterations//10))
        duration = time.time() - start_time

        # Show final metrics
        print(f"\nTask completed in {duration:.2f} seconds")
        print("\nFinal GPU State:")
        for metric, value in governor.get_current_metrics().items():
            print(f"{metric}: {value}")

        # Add some cooling time between tests
        time.sleep(2)
        governor.cleanup()
        
    # Show optimization history
    history = governor.get_optimization_history()
    print("\nOptimization History:")
    print(f"Total optimizations: {history['total_optimizations']}")
    print(f"Average complexity: {history['average_complexity']:.2f}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your GPU setup.")
        sys.exit(1)
        
    demonstrate_gpu_governor()