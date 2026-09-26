"""一键：classifier → importance → 出图与 LaTeX 片段。

    PYTHONPATH=. python -m src.ml
"""

from __future__ import annotations

from src.ml.classifier import main as run_classifier
from src.ml.importance import main as run_importance
from src.ml.plot_and_report import main as run_plot_and_report


def main() -> int:
    """顺序执行三条命令对应的入口。"""
    code = run_classifier()
    if code != 0:
        return code
    code = run_importance()
    if code != 0:
        return code
    return run_plot_and_report()


if __name__ == "__main__":
    raise SystemExit(main())
