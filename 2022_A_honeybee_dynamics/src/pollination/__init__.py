"""授粉供需与果园蜂箱配比。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.pollination.orchard_20acres import run_orchard_optimization as run_orchard_optimization

__all__ = ["run_orchard_optimization"]


def __getattr__(name: str) -> Any:
    if name == "run_orchard_optimization":
        from src.pollination.orchard_20acres import run_orchard_optimization

        return run_orchard_optimization
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
