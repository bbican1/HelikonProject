from abc import ABC

from langchain_core.tools import BaseTool


class BaseHelikonTool(BaseTool, ABC):
    name: str
    description: str