"""Executable MCP server entrypoint for the personal context memory system."""

from __future__ import annotations

import os
from pathlib import Path

from adapters.mcp import create_mcp_server, load_services_from_env


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    services = load_services_from_env(root_dir)
    try:
        server = create_mcp_server(services)
        transport = os.getenv("MCP_TRANSPORT", "stdio")
        server.run(transport=transport)
    finally:
        if services.graph_store is not None:
            services.graph_store.close()


if __name__ == "__main__":
    main()
