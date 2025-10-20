# NVIDIA GPU Governor — Real-time GPU Optimization Demo

This project demonstrates a real-time GPU governor that analyzes task complexity, selects an appropriate floating-point precision tier (fp32/fp16), applies safe runtime optimizations, and runs workloads under the chosen precision while monitoring GPU metrics.

## Features

- **Intelligent Precision Selection**: Automatically chooses between FP32 and FP16 based on task complexity analysis
- **Real-time GPU Monitoring**: Tracks GPU utilization, power consumption, and performance metrics
- **Adaptive Optimization**: Applies runtime optimizations based on workload characteristics
- **Comprehensive Telemetry**: Collects detailed performance data for analysis

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic GPU Governor Demo

Run the main governor demo that analyzes prompts and applies floating-point precision:

```bash
python test_gpu_governor.py
```

### Heavy Workload Testing

Run a batched matrix multiplication test to saturate the GPU:

```bash
python heavy_batched_test.py
```

### Performance Monitoring

Collect performance logs while running workloads:

```bash
python perf_collector.py --duration 60 --out log.csv
```

### Energy Savings Demo

Demonstrate energy savings with different precision modes:

```bash
python examples/energy_savings_demo.py
```

## Project Structure

```
src/
├── core/           # Core GPU control and analysis modules
├── analyzer/       # Complexity analysis algorithms
└── utils/          # Utility functions and metrics
```

## How It Works

1. **Task Analysis**: The system analyzes incoming tasks to determine their computational complexity
2. **Precision Selection**: Based on complexity, it selects the optimal floating-point precision (FP32/FP16)
3. **Runtime Optimization**: Applies appropriate optimizations for the selected precision
4. **Monitoring**: Continuously monitors GPU metrics and performance
5. **Adaptive Adjustment**: Makes real-time adjustments based on performance feedback

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- PyTorch with CUDA
- nvidia-ml-py (for GPU monitoring)