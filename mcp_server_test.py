#MCP testing
# math_server.py
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

mcp = FastMCP("Maths")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a - b

if __name__ == "__main__":
    print("MCP server is running using stdio transport ...")
    mcp.run(transport="stdio")