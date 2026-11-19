import json
import textwrap

from pydantic import BaseModel

from config import Settings, settings
from llm import get_llm
from pipelines.gen_pipeline.agents.agent_registry import list_agents
from pipelines.gen_pipeline.agents.base_agent import BaseAgent
from pipelines.gen_pipeline.gen_pipeline_state import GenPipelineState
from routes.models import list_models


class PlannerAgent(BaseAgent):
    name = "planner_agent"
    description = "Plant den zu nutzenden Agenten"

    def invoke(self, state: GenPipelineState):
        prompt = textwrap.dedent(f"""
            Du bist ein Planungsagent für ein KI-System.
            
            Deine Aufgaben:
            1. Wähle den passenden Agenten für die Benutzeranfrage.
            2. Wähle ein geeignetes Sprachmodell.
            
            Verfügbare Agenten:
            {list_agents()}
            
            Benutzeranfrage:
            {state["messages"][0].content}
            
            {list_models()}
            
            Antworte ausschließlich als JSON:
            {{
              "agent": "name_des_agenten"
            }}
        """)

        llm = get_llm(settings.gen_pipeline_planner_model)
        structured_llm = llm.with_structured_output(
            PlannerResult
        )
        response = structured_llm.invoke(prompt)
        state["selected_agent"] = response.agent
        if not state["model"] or state["model"]=="":
            state["model"] = response.model
        return state

class PlannerResult(BaseModel):
    agent: str
    model: str  # TODO nur welche mit toolcalling