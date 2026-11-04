from abc import ABC, abstractmethod

from llm import get_llm
from pipelines.gen_pipeline.tools.tool_registry import get_tools


class BaseAgent(ABC):
    name: str
    description: str
    tool_names: list[str]

    @abstractmethod
    def invoke(self, state):
        pass

    # kann kein Attribut sein damit das Modell zur Laufzeit geändert werden kann
    def build_llm_with_tools(self, model):
        return get_llm(model).bind_tools(get_tools(self.tool_names))