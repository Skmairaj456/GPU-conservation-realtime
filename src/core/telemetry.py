"""GPU telemetry and power monitoring."""
import logging
from dataclasses import dataclass
from typing import Optional, Dict

try:
    import pynvml
except ImportError:
    pynvml = None

@dataclass
class GPUTelemetry:
    """GPU telemetry data point."""
    timestamp: float
    utilization: float
    memory_used: float
    memory_total: float
    power_draw: Optional[float]
    temperature: Optional[float]
    clock_speed: Optional[int]

class TelemetryManager:
    """Manages GPU telemetry collection via NVML."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._nvml = pynvml
        self._nvml_initialized = False
        self._nvml_handle = None
        self._init_nvml()
    
    def _init_nvml(self):
        """Initialize NVML for power/temp monitoring."""
        if not self._nvml or self._nvml_initialized:
            return
        try:
            self._nvml.nvmlInit()
            self._nvml_handle = self._nvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_initialized = True
            self.logger.info("NVML initialized for power telemetry")
        except Exception as e:
            self.logger.warning(f"NVML init failed: {e}; power telemetry disabled")
            self._nvml = None
    
    def get_telemetry(self) -> Optional[GPUTelemetry]:
        """Get current GPU telemetry."""
        import torch
        import time
        
        if not torch.cuda.is_available():
            return None
            
        try:
            # Basic metrics via PyTorch
            memory_used = torch.cuda.memory_allocated() / 1024**2
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            
            # Try utilization (PyTorch >=2.2)
            try:
                util = torch.cuda.utilization()
            except Exception:
                util = 0.0
                
            telemetry = GPUTelemetry(
                timestamp=time.time(),
                utilization=float(util),
                memory_used=float(memory_used),
                memory_total=float(memory_total),
                power_draw=None,
                temperature=None,
                clock_speed=None
            )
            
            # Enrich with NVML data if available
            if self._nvml and self._nvml_initialized and self._nvml_handle:
                try:
                    # Update memory from NVML (more accurate)
                    mem = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    telemetry.memory_used = float(mem.used) / 1024**2
                    telemetry.memory_total = float(mem.total) / 1024**2
                    
                    # Get power (milliwatts)
                    try:
                        pwr_mw = self._nvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                        telemetry.power_draw = float(pwr_mw) / 1000.0
                    except Exception:
                        pass
                        
                    # Get temperature (Celsius)
                    try:
                        temp = self._nvml.nvmlDeviceGetTemperature(
                            self._nvml_handle,
                            self._nvml.NVML_TEMPERATURE_GPU
                        )
                        telemetry.temperature = float(temp)
                    except Exception:
                        pass
                        
                    # Get clock speed (MHz)
                    try:
                        clock = self._nvml.nvmlDeviceGetClockInfo(
                            self._nvml_handle,
                            self._nvml.NVML_CLOCK_GRAPHICS
                        )
                        telemetry.clock_speed = int(clock)
                    except Exception:
                        pass
                        
                except Exception as e:
                    self.logger.debug(f"NVML data collection partial failure: {e}")
                    
            return telemetry
            
        except Exception as e:
            self.logger.error(f"Failed to collect GPU telemetry: {e}")
            return None
    
    def get_formatted_metrics(self) -> Dict[str, str]:
        """Get current metrics formatted for display."""
        telemetry = self.get_telemetry()
        if not telemetry:
            return {}
            
        metrics = {
            'utilization': f"{telemetry.utilization:.1f}%",
            'memory_used': f"{telemetry.memory_used:.0f}MB / {telemetry.memory_total:.0f}MB",
            'memory_utilization': f"{(telemetry.memory_used/telemetry.memory_total)*100:.1f}%"
        }
        
        if telemetry.power_draw is not None:
            metrics['power'] = f"{telemetry.power_draw:.1f}W"
        if telemetry.temperature is not None:
            metrics['temperature'] = f"{telemetry.temperature:.1f}°C"
        if telemetry.clock_speed is not None:
            metrics['clock'] = f"{telemetry.clock_speed}MHz"
            
        return metrics