from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")


def batch_iter(items: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    bucket: list[T] = []
    for item in items:
        bucket.append(item)
        if len(bucket) == batch_size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket
