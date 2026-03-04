"""
app/utils/retry.py
Exponential backoff retry decorator for async functions.
Configurable per-exception-type retries with jitter.
"""
import asyncio
import functools
import random
from typing import Any, Callable, Type

import structlog

log = structlog.get_logger(__name__)


def async_retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Decorator for async functions: retry with exponential backoff + jitter.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        exponential_base: Multiplier for each successive delay
        jitter: Add random jitter to prevent thundering herd
        exceptions: Tuple of exception types to retry on
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt == max_retries:
                        log.error(
                            "retry_exhausted",
                            function=func.__name__,
                            attempt=attempt,
                            error=str(exc),
                        )
                        raise

                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    if jitter:
                        delay *= (0.5 + random.random())  # ±50% jitter

                    log.warning(
                        "retry_attempt",
                        function=func.__name__,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay_seconds=round(delay, 2),
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)

            raise last_exception  # type: ignore[misc]

        return wrapper
    return decorator
