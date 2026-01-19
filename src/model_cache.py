from datetime import datetime
from typing import List, Dict, Optional
from huggingface_hub import scan_cache_dir


def list_cached_models(
    sort_by: str = "size",
    reverse: bool = True,
    model_filter: Optional[str] = None
) -> List[Dict[str, any]]:
    """List HuggingFace models cached locally.

    Args:
        sort_by: Sort key - "size", "name", "accessed", or "modified"
        reverse: Sort in descending order (default True)
        model_filter: Optional substring to filter model names

    Returns:
        List of dicts with model info: name, size, last_accessed, last_modified
    """
    try:
        cache_info = scan_cache_dir()
    except Exception as e:
        print(f"Error scanning cache: {e}")
        return []

    models = []

    for repo in cache_info.repos:
        # Only include models (not datasets or spaces)
        if repo.repo_type != "model":
            continue

        # Apply filter if provided
        if model_filter and model_filter.lower() not in repo.repo_id.lower():
            continue

        # Convert bytes to GB
        size_gb = repo.size_on_disk / (1024**3)

        # Convert timestamps to readable format
        last_accessed = datetime.fromtimestamp(repo.last_accessed)
        last_modified = datetime.fromtimestamp(repo.last_modified)

        models.append({
            "name": repo.repo_id,
            "size_bytes": repo.size_on_disk,
            "size_gb": size_gb,
            "last_accessed": last_accessed,
            "last_modified": last_modified,
            "nb_files": repo.nb_files,
        })

    # Sort models
    sort_keys = {
        "size": "size_bytes",
        "name": "name",
        "accessed": "last_accessed",
        "modified": "last_modified",
    }

    sort_key = sort_keys.get(sort_by, "size_bytes")
    models.sort(key=lambda x: x[sort_key], reverse=reverse)

    return models


def print_cached_models(
    sort_by: str = "size",
    reverse: bool = True,
    model_filter: Optional[str] = None
) -> None:
    """Print cached HuggingFace models in a formatted table.

    Args:
        sort_by: Sort key - "size", "name", "accessed", or "modified"
        reverse: Sort in descending order (default True)
        model_filter: Optional substring to filter model names
    """
    models = list_cached_models(sort_by=sort_by, reverse=reverse, model_filter=model_filter)

    if not models:
        print("No cached models found.")
        return

    total_size = sum(m["size_gb"] for m in models)

    print(f"\nCached HuggingFace Models ({len(models)} models, {total_size:.2f} GB total)")
    print("─" * 80)

    for model in models:
        name = model["name"]
        size = f"{model['size_gb']:.2f} GB"
        date = model["last_accessed"].strftime("%Y-%m-%d")

        # Truncate long names
        if len(name) > 50:
            name = name[:47] + "..."

        print(f"{name:<50} {size:>10}    {date}")

    print()


def get_cache_size() -> float:
    """Get total size of HuggingFace cache in GB.

    Returns:
        Total cache size in gigabytes
    """
    try:
        cache_info = scan_cache_dir()
        return cache_info.size_on_disk / (1024**3)
    except Exception as e:
        print(f"Error scanning cache: {e}")
        return 0.0
