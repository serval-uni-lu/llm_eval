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
                        print(f"    Attempt {attempt}/{max_attempts} failed: {str(e)[:50]}... Retrying in {current_delay:.1f}s")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        print(f"    All {max_attempts} attempts failed")

            # Re-raise the last exception after all retries
            raise last_exception

        return wrapper
    return decorator
