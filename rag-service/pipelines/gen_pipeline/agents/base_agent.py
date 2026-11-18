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
    sys_prompt: str

    @abstractmethod
    def invoke(self, state):
        pass

    # kann kein Attribut sein damit das Modell zur Laufzeit geändert werden kann
    def build_llm_with_tools(self, model):
        return get_llm(model).bind_tools(get_tools(self.tool_names))

    def basic_response(self, state: PipelineState):
        if not state["retry_count"] == 0 and not state["approved"]:
            return self.basic_regenerate(state)
        llm = self.build_llm_with_tools(state["model"])
        response = llm.invoke(self.build_message_history(state))
        return {"messages": [response]}

    def basic_regenerate(self, state: PipelineState):
        regen_prompt = f"""
            Der Validator hat deine Antwort abgelehnt.
            
            Feedback:
            {state["validator_feedback"]}
            
            Verbessere deine letzte Antwort und beachte das Feedback.
            
            Verwende bereits vorhandene Tool-Ergebnisse aus der Historie.
            
            Entschuldige dich nicht für Fehler und weise auch sonst nicht darauf hin,
            dass erneut generiert werden musste. Der User soll das nicht erfahren.
        """
            # Führe keine erneuten Tool-Aufrufe durch, außer wenn die benötigte Information
            # nicht in der bisherigen Historie vorhanden ist.

        llm = get_llm(state["model"])   # TODO toolcalling geht nicht weil er sonst unendlich viele macht
        response = llm.invoke(self.build_message_history(state) + [HumanMessage(textwrap.dedent(regen_prompt))])
        return {"messages":[response]}

    def build_message_history(self, state: PipelineState):
        return [SystemMessage(textwrap.dedent(self.sys_prompt))] + [HumanMessage(state["user_prompt"])] + state["messages"]