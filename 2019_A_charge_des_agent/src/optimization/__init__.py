"""Multi-objective sizing and stress tests for airport charging DES."""

__all__ = ["run_pareto_sizing_pipeline"]


def __getattr__(name: str):
    if name == "run_pareto_sizing_pipeline":
        from .pareto_sizing import run_pareto_sizing_pipeline

        return run_pareto_sizing_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
