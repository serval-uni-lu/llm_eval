import re
from pathlib import Path
from itertools import islice
from operator import itemgetter
from typing import Any, Sequence


def ensure_dir(path: str | Path):
    """Ensure that a directory exists."""
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_text(text: str) -> str:
    """Normalize text for use in filenames and paths."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    normalized = normalized.strip("_")
    return normalized.lower()


def clear_cuda_cache():
    """Clear CUDA cache to prevent OOM errors."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def get_items(
    obj: Sequence[Any],
    *indices: int,
    batch: bool = False,
) -> tuple[Any, ...] | tuple[tuple[Any, ...], ...]:
    """
    Parameters
    ----------
    obj : sequence
        Source sequence
    *indices : int
        Indices to extract
    batch : bool
        Controls the shape of the returned data.

        If False (default):
            Returns a flat tuple containing the selected elements,
            preserving their original order.

        If True:
            Interprets each selected element as a row and returns
            a tuple of columns (i.e. the transpose of the rows).

    Examples
    --------
    >>> data = [(1, "a"), (2, "b"), (3, "c")]

    # get_subsequence behavior
    >>> get_items(data, 0, 2)
    (1, 3)

    >>> get_items(data, 1)
    (2,)

    # get_batch behavior
    >>> get_items(data, 0, 2, batch=True)
    ((1, 3), ('a', 'c'))

    >>> get_items(data, 1, batch=True)
    ((2,), ('b',))
    """

    if not indices:
        return ()

    items = itemgetter(*indices)(obj)

    # normalize itemgetter behavior
    if len(indices) == 1:
        items = (items,)

    if not batch:
        return items

    # batch mode: transpose rows -> columns
    if not isinstance(items[0], tuple):
        items = (items,)

    return tuple(zip(*items))
