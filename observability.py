import psutil
import threading
import time
import torch
from contextlib import contextmanager


@contextmanager
def resource_monitor(interval=1.0):
    # interval is in seconds
    """Monitor system resources during model loading"""
    monitoring = True
    resources = {'peak_cpu': 0, 'peak_memory': 0, 'peak_gpu_memory': 0}
    
    def monitor():
        while monitoring:
            resources['peak_cpu'] = max(resources['peak_cpu'], psutil.cpu_percent())
            resources['peak_memory'] = max(resources['peak_memory'], psutil.virtual_memory().percent)
            
            if torch.cuda.is_available():
                try:
                    gpu_mem = torch.cuda.memory_allocated() / 1024**3
                    resources['peak_gpu_memory'] = max(resources['peak_gpu_memory'], gpu_mem)
                except:
                    pass
            
            time.sleep(interval)
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    
    try:
        yield resources
    finally:
        monitoring = False
        monitor_thread.join(timeout=1.0)