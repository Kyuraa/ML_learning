from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse
from starlette.requests import Request
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

import logging
import json

# Initialize the MCP server
mcp = FastMCP("MathServer")

async def handle_messages(request: Request):
    try:
        data = await request.json()
        logging.info(f"Received: {data}")

        if data.get("action") == "introspect":
            response = {
                "type": "response",
                "data": {
                    "tools": [
                        {
                            "name": "add",
                            "description": "Add two numbers",
                            "parameters": {"a": {"type": "int"}, "b": {"type": "int"}},
                        },
                        {
                            "name": "subtract",
                            "description": "Subtract two numbers",
                            "parameters": {"a": {"type": "int"}, "b": {"type": "int"}},
                        },
                        {
                            "name": "multiply",
                            "description": "Multiply two numbers",
                            "parameters": {"a": {"type": "int"}, "b": {"type": "int"}},
                        },
                    ]
                }
            }

        elif data.get("action") == "execute":
            tool_name = data.get("tool")
            params = data.get("data", {})

            if tool_name == "add":
                result = add(**params)
            elif tool_name == "subtract":
                result = subtract(**params)
            elif tool_name == "multiply":
                result = multiply(**params)
            elif tool_name == "reverse_text":
                result = reverse_text(**params)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            response = {"type": "response", "data": result}

        else:
            raise ValueError(f"Unknown action: {data.get('action')}")

        return JSONResponse(response)

    except Exception as e:
        logging.error(f"handle_messages error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# MCP Tools
@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b

@mcp.tool()
def subtract(a: int, b: int) -> int:
    return a - b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    return a * b

@mcp.tool()
def reverse_text(text: str) -> str:
        return text[::-1]

# Starlette setup
middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
]

routes = [
    Mount("/mcp", routes=[
        Route("/messages", endpoint=handle_messages, methods=["POST"]),
    ]),
    Route("/messages", endpoint=handle_messages, methods=["POST"]),
]

app = Starlette(routes=routes, middleware=middleware)

# Entry point
if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    print("✅ Math MCP server running at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")