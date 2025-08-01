# utils/profiling.py

import torch
import time
import logging
import contextlib

@contextlib.contextmanager
def profile_section(name: str):
    logging.info(f"[PROFILE] Starting '{name}'...")
    start_time = time.time()
    yield
    duration = time.time() - start_time
    logging.info(f"[PROFILE] '{name}' took {duration:.4f} seconds.")

def log_gpu_memory():
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        logging.info(f"[GPU] Memory Allocated: {mem_allocated:.2f} MB")
        logging.info(f"[GPU] Memory Reserved:  {mem_reserved:.2f} MB")
    else:
        logging.info("[GPU] CUDA not available.")
