"""LangGraph DES agents: self-reflection and task planning."""

__all__ = [
    "build_reflection_graph",
    "run_reflection_agent",
    "run_planning_agent",
    "plan_subtasks",
]


def __getattr__(name: str):
    """Lazy export to avoid runpy double-import warnings."""
    if name in {"build_reflection_graph", "run_reflection_agent"}:
        from .reflection_agent import build_reflection_graph, run_reflection_agent

        return {
            "build_reflection_graph": build_reflection_graph,
            "run_reflection_agent": run_reflection_agent,
        }[name]
    if name in {"run_planning_agent", "plan_subtasks"}:
        from .planning_agent import plan_subtasks, run_planning_agent

        return {
            "run_planning_agent": run_planning_agent,
            "plan_subtasks": plan_subtasks,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
