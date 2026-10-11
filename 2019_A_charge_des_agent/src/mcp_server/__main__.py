"""
Cursor / Claude MCP stdio entrypoint.

Launch:
  PYTHONPATH=src python -m mcp_server

Do not print to stdout before the MCP handshake.
"""

from __future__ import annotations

from .server import mcp


def main() -> None:
    """Start FastMCP with the default stdio transport."""
    mcp.run()


if __name__ == "__main__":
    main()
