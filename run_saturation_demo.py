import subprocess
import sys
import time
from pathlib import Path

def main():
    # Start performance collector in background
    perf_output = "saturation_metrics.csv"
    print(f"Starting performance collection to {perf_output}...")
    collector_proc = subprocess.Popen([
        sys.executable,
        "perf_collector.py",
        "--output", perf_output,
        "--interval", "0.5"  # Sample every 500ms for granular data
    ])
    
    try:
        # Let collector start up
        time.sleep(1)
        
        # Run heavy test (longer duration for good data)
        print("Running GPU saturation test...")
        heavy_proc = subprocess.run([
            sys.executable,
            "heavy_batched_test.py",
            "--iterations", "100",  # Longer run for better metrics
            "--compile", "auto"  # Will try compile if Triton available
        ], check=True)
        
        # Let collector capture cooldown
        time.sleep(2)
        
    finally:
        # Ensure collector stops
        collector_proc.terminate()
        collector_proc.wait()
    
    print(f"\nTest complete! Performance data saved to {perf_output}")
    print("You can plot this CSV file to show GPU utilization over time.")
    
if __name__ == "__main__":
    main()