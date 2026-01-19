import re
from pathlib import Path

def ensure_dir(path: str | Path):
    """Ensure that a directory exists."""
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_text(text: str) -> str:
    """Normalize text for use in filenames and paths."""
    normalized = re.sub(r'[^a-zA-Z0-9]+', '_', text)
    normalized = normalized.strip('_')
    return normalized.lower()


def clear_cuda_cache():
    """Clear CUDA cache to prevent OOM errors."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()