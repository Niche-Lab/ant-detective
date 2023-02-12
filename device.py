import torch
import gc
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import time

def free_gpu_cache():
    # pause for 3 seconds
    time.sleep(3)

    print("Initial GPU Usage")
    gpu_usage()                             

    gc.collect()
    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

