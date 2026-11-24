import textwrap

from langchain_core.messages import SystemMessage, AIMessage
from pydantic.v1 import BaseModel

from config import settings
from llm import get_llm
from pipelines.gen_pipeline.agents.base_agent import BaseAgent
from pipelines.gen_pipeline.gen_pipeline_state import GenPipelineState


class PromptcheckAgent(BaseAgent):
    name = "promptcheck_agent"
    description = "Checkt den Prompt auf verbotene Inhalte"
    prompt = """
        Stelle fest ob der Prompt verbotene Inhalte enthält.
        
        Wenn der Prompt verbotene Inhalte enthält soll er abgelehnt werden.
        Wenn du ablehnst schreibe die Begründung als reason dazu.
    """

    # TODO Security Team
    def invoke(self, state: GenPipelineState):
        llm = get_llm(settings.gen_pipeline_promptcheck_model)
        structured_llm = llm.with_structured_output(
            PromptcheckResult
        )
        response = structured_llm.invoke(
            [SystemMessage(textwrap.dedent(self.prompt))] +
            state["messages"]
        )
        state["prompt_ok"] = response.ok
        state["prompt_reason"] = response.reason
        if not response.ok:
            state["messages"].append(AIMessage(response.reason))
        return state

class PromptcheckResult(BaseModel):
    ok: bool
    reason: str