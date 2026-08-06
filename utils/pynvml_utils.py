import pynvml
pynvml.nvmlInit()

def is_free_processes(id: int, memory_thresh=1024) -> bool:
    from functools import reduce
    handle = pynvml.nvmlDeviceGetHandleByIndex(id)
    compute_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    graphics_processes = pynvml.nvmlDeviceGetGraphicsRunningProcesses(
        handle)
    used_memory_process = list(filter(
        lambda x: x.usedGpuMemory is not None, compute_processes + graphics_processes))
    if len(used_memory_process) < 1:
        return True
    elif len(used_memory_process) == 1:
        used_memory = used_memory_process[0].usedGpuMemory / 1024**2  # MB
    else:
        used_memory = reduce(lambda x, y: x.usedGpuMemory + 
                             y.usedGpuMemory, used_memory_process) / 1024**2
    print(f"gpu id {id}, used_memory {used_memory} MB")
    return used_memory < memory_thresh

