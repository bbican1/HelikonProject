import textwrap
from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_llm
from pipelines.gen_pipeline.pipeline_state import PipelineState
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

    def basic_response(self, state: PipelineState, sys_prompt: str):
        llm = self.build_llm_with_tools(state["model"])
        response = llm.invoke([SystemMessage(textwrap.dedent(sys_prompt))] + [HumanMessage(state["user_prompt"])] + state["messages"])
        return {"messages": [response]}