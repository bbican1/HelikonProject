import json

from pipelines.gen_pipeline.tools.base_tool import BaseHelikonTool
from pipelines.gen_pipeline.tools.ragtool import RagTool

TOOLS = [
    RagTool
]

def list_tools() -> str:
    return json.dumps([
        {
            "name": tool.model_fields["name"].default,
            "description": tool.model_fields.name["description"].default
        } for tool in TOOLS
    ],
    ensure_ascii=False)

def get_tool(name: str) -> BaseHelikonTool:
    for tool in TOOLS:
        if tool.model_fields["name"].default == name:
            return tool()
    raise ValueError(f"Unknown Tool: {name}")

def get_all_tools() -> list[BaseHelikonTool]:
    return [tool() for tool in TOOLS]

def get_tools(names: list[str]) -> list[BaseHelikonTool]:
    tools = []
    for name in names:
        tools.append(get_tool(name))
    return tools