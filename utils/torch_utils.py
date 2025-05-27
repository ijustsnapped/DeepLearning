# your_project_name/utils/torch_utils.py
import torch

def get_device(device_str: str | None = None) -> torch.device:
    """Gets the torch device."""
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CudaTimer:
    """Context manager for timing CUDA operations."""
    def __init__(self, device: torch.device):
        self.device = device
        if self.device.type == 'cuda':
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        self.elapsed_time_ms = 0

    def __enter__(self):
        if self.device.type == 'cuda':
            self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device.type == 'cuda':
            self.end_event.record()
            torch.cuda.synchronize() # Wait for the events to complete
            self.elapsed_time_ms = self.start_event.elapsed_time(self.end_event)

    def get_elapsed_time_ms(self) -> float:
        return self.elapsed_time_ms

def reset_cuda_peak_memory_stats(device: torch.device):
    """Resets CUDA peak memory statistics if on a CUDA device."""
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

def empty_cuda_cache(device: torch.device):
    """Empties CUDA cache if on a CUDA device."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()