"""Shared helper to make experiment result dictionaries JSON-serialisable.

Custom implementation: a thin recursive converter, not covered by any
researched library.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays/bools inside dicts, lists
    and tuples into native Python types accepted by json.dump."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    return obj
