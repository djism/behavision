import asyncio
import json
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters
from ollama import AsyncClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL = "qwen3:4b"


def mcp_tools_to_ollama_tools(mcp_tools):
    """
    Convert tools discovered from the MCP server into
    Ollama function-tool definitions.

    The MCP server remains the source of truth.
    We do not hard-code Behavision tool schemas here.
    """

    ollama_tools = []

    for tool in mcp_tools:
        raw = tool.model_dump(
            by_alias=True
        )

        parameters = (
            raw.get("inputSchema")
            or raw.get("input_schema")
            or {
                "type": "object",
                "properties": {},
            }
        )

        ollama_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": (
                        tool.description
                        or f"Behavision tool: {tool.name}"
                    ),
                    "parameters": parameters,
                },
            }
        )

    return ollama_tools


def mcp_result_to_dict(result):
    """
    Convert MCP result into JSON-compatible data
    that we can return to the local LLM.
    """

    if result.structured_content is not None:
        return result.structured_content

    text_parts = []

    for block in result.content:
        if hasattr(block, "text"):
            text_parts.append(
                block.text
            )

    return {
        "text": "\n".join(text_parts)
    }


async def run_agent(
    question: str,
    mcp_client: Client,
    ollama_tools: list,
):
    """
    Local agent loop:

    1. Give user question + MCP tools to Ollama.
    2. Ollama decides which tool(s) to call.
    3. Execute those tools through MCP.
    4. Return tool results to Ollama.
    5. Repeat until Ollama produces a final answer.
    """

    ollama_client = AsyncClient()

    messages = [
        {
            "role": "system",
            "content": (
                "You are the Behavision experiment assistant. "
                "Use the available tools whenever the user asks for "
                "actual run-specific data, motion statistics, tracking "
                "quality, or QC information. "
                "Never invent experiment values or units. "
                "If units are not explicitly provided by the tool, "
                "describe values exactly as reported. "
                "Behavision motion metrics may be in normalized arena "
                "coordinates rather than physical units. "
                "Do not claim that values are normal, expected, scientifically valid, "
                "or high quality unless the tool output explicitly defines those criteria. "
                "When discussing tracking quality, distinguish between "
                "'passed the available QC checks' and 'proven correct tracking'. "
                "If multiple tools are needed, call all relevant tools. "
                "After receiving tool results, provide a concise answer "
                "based only on those results."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    for _ in range(6):

        response = await ollama_client.chat(
            model=MODEL,
            messages=messages,
            tools=ollama_tools,
        )

        assistant_message = response.message

        # Keep the assistant message in conversation history.
        messages.append(
            assistant_message.model_dump()
        )

        tool_calls = (
            assistant_message.tool_calls
            or []
        )

        # -----------------------------------
        # No tool calls: final answer
        # -----------------------------------

        if not tool_calls:

            answer = (
                assistant_message.content
                or "No answer returned."
            )

            print(
                f"\nBehavision > {answer}\n"
            )

            return

        # -----------------------------------
        # Execute all requested MCP tools
        # -----------------------------------

        for tool_call in tool_calls:

            tool_name = tool_call.function.name

            arguments = (
                tool_call.function.arguments
                or {}
            )

            print(
                f"\n[MCP] LLM selected tool: "
                f"{tool_name}"
            )

            print(
                f"[MCP] Arguments: "
                f"{arguments}"
            )

            try:

                result = await mcp_client.call_tool(
                    tool_name,
                    arguments,
                )

                tool_data = mcp_result_to_dict(
                    result
                )

            except Exception as exc:

                tool_data = {
                    "success": False,
                    "error": str(exc),
                }

            print(
                "[MCP] Tool completed"
            )

            # Return tool output to Ollama.
            messages.append(
                {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": json.dumps(
                        tool_data
                    ),
                }
            )

    print(
        "\nBehavision > "
        "Stopped because the tool-call limit was reached.\n"
    )


async def main():

    print(
        "\nBehavision Local AI Assistant"
    )

    print(
        f"Model: {MODEL}"
    )

    print(
        "Runs entirely through local Ollama + MCP."
    )

    print(
        "Type 'exit' to quit.\n"
    )

    # ---------------------------------------
    # Configure local Behavision MCP server
    # ---------------------------------------

    server = StdioServerParameters(
        command=sys.executable,

        args=[
            "-m",
            "behavision.mcp.server",
        ],

        cwd=str(PROJECT_ROOT),
    )

    # ---------------------------------------
    # Start MCP session
    # ---------------------------------------

    async with Client(server) as mcp_client:

        # -----------------------------------
        # MCP tool discovery
        # -----------------------------------

        tools_result = await mcp_client.list_tools()

        mcp_tools = tools_result.tools

        print(
            "Connected to Behavision MCP server."
        )

        print(
            "\nDiscovered MCP tools:"
        )

        for tool in mcp_tools:
            print(
                f"  - {tool.name}"
            )

        # Convert MCP schemas into Ollama tools.
        ollama_tools = mcp_tools_to_ollama_tools(
            mcp_tools
        )

        print(
            "\nLocal LLM can now choose "
            "among these MCP tools.\n"
        )

        # -----------------------------------
        # Interactive chat loop
        # -----------------------------------

        while True:

            question = input(
                "You > "
            ).strip()

            if question.lower() in {
                "exit",
                "quit",
                "q",
            }:
                break

            if not question:
                continue

            try:

                await run_agent(
                    question,
                    mcp_client,
                    ollama_tools,
                )

            except Exception as exc:

                print(
                    f"\nError: {exc}\n"
                )


if __name__ == "__main__":
    asyncio.run(
        main()
    )