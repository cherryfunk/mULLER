"""
HKT-aware wrappers for standard Python types.

This module provides thin wrappers around built-in Python types that make them
compatible with the `returns` library's Higher-Kinded Types (HKT) system.
These wrappers inherit from both the original type and `SupportsKind1`,
providing zero runtime overhead while satisfying mypy's type checking.

Example:
    >>> from muller import List
    >>> sort = List([1, 2, 3, 4, 5, 6])  # HKT-compatible list
"""

from __future__ import annotations

from typing import TypeVar

from returns.primitives.hkt import SupportsKind1

_T = TypeVar("_T")


class List(list[_T], SupportsKind1["List", _T]):  # type: ignore[type-arg]
    """
    HKT-aware list wrapper for use with the returns library.

    This is a thin wrapper around Python's built-in `list` that implements
    `SupportsKind1`, making it compatible with `Kind1[List, T]` type annotations.

    Inherits all behavior from `list` with zero runtime overhead.

    Example:
        >>> sort = List(range(1, 7))  # [1, 2, 3, 4, 5, 6]
        >>> sort.append(7)
        >>> len(sort)
        7
    """

    pass
