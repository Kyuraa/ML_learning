import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Server configuration
server_params = StdioServerParameters(
    command="python",
    args=["mcp_server_test.py"],
    env=None,
)
# Example call
async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("##### Available tools:", tools)
            result = await session.call_tool("add", arguments={"a": 7, "b": 13})
            print("##### Result of addition (7 + 13):", result)
            result = await session.call_tool("multiply", arguments={"a": 12, "b": 3})
            print("##### Result of multiply(12, 3):", result)

if __name__ == "__main__":
    asyncio.run(run())

