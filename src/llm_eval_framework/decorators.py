import time
from typing import Callable
from functools import wraps


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each attempt

    Example:
        @retry(max_attempts=3, delay=1.0, backoff=2.0)
        def unstable_function():
            # May fail, will retry up to 3 times
            pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        print(
                            f"    Attempt {attempt}/{max_attempts} failed: {str(e)[:50]}..."
                            f" Retrying in {current_delay:.1f}s"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        print(f"    All {max_attempts} attempts failed")

            # Re-raise the last exception after all retries
            raise last_exception

        return wrapper

    return decorator


def retry_batches(retries: int):
    """
    Decorator that adds retry logic to a batch-processing function.
    The decorated function must accept: (llm_inputs, batch_size)
    And must return: (results_dict, failed_indices_list)
    """
    from .utils import get_items

    def decorator(func):
        def wrapper(llm_inputs: list[dict], batch_size: int):
            total = len(llm_inputs)
            indices = list(range(total))
            final_results = {}

            for _ in range(retries):
                if not indices:
                    break

                results, failed = func(
                    get_items(llm_inputs, *indices, batch=False), batch_size
                )

                # map subset-index → original-index
                for sub_idx, res in results.items():
                    final_results[indices[sub_idx]] = res

                # remap remaining failures back to original indices
                indices = [indices[sub_idx] for sub_idx in failed]

            # at the end of all the retries, put enriched error msg in final_results
            for sub_idx, error_msg in failed.items():
                final_results[indices[sub_idx]] = error_msg

            # produce result list in original order
            return [final_results.get(i, None) for i in range(total)]

        return wrapper

    return decorator
