"""Utility functions for energy and performance metrics."""
from typing import List, Dict, Any
import time
import csv
from pathlib import Path

def compute_running_average(values: List[float], window: int = 5) -> List[float]:
    """Compute running average over a window."""
    if not values:
        return []
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start:i + 1]
        result.append(sum(window_vals) / len(window_vals))
    return result

def log_metrics_to_csv(
    metrics: List[Dict[str, Any]],
    output_file: str,
    timestamp_key: str = 'timestamp'
) -> None:
    """Log metrics to CSV file.
    
    Args:
        metrics: List of metric dictionaries
        output_file: Path to output CSV
        timestamp_key: Key for timestamp in metrics
    """
    if not metrics:
        return
        
    # Ensure parent directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all unique keys as columns
    columns = set()
    for entry in metrics:
        columns.update(entry.keys())
    columns = sorted(list(columns))
    
    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for entry in metrics:
            writer.writerow(entry)

def format_joules(joules: float, precision: int = 1) -> str:
    """Format energy value with appropriate unit prefix."""
    if joules < 1:
        return f"{joules*1000:.{precision}f}mJ"
    if joules < 1000:
        return f"{joules:.{precision}f}J"
    return f"{joules/1000:.{precision}f}kJ"

def format_watts(watts: float, precision: int = 1) -> str:
    """Format power value with appropriate unit prefix."""
    if watts < 1:
        return f"{watts*1000:.{precision}f}mW"
    if watts < 1000:
        return f"{watts:.{precision}f}W"
    return f"{watts/1000:.{precision}f}kW"