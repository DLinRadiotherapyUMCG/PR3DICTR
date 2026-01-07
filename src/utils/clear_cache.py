import torch
import gc

def clear_cache():
    """
    Clears the GPU cache to free up memory.
    """
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()