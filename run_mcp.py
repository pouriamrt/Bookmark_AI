"""Entry point for the Bookmark AI MCP server (stdio transport)."""

from bookmark_app.mcp_server import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
