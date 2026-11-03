import json
import textwrap

from pydantic import BaseModel

from llm import get_llm
from pipelines.gen_pipeline.agents.agent_registry import list_agents


class PlannerAgent:

    def invoke(self, state):

        prompt = textwrap.dedent(f"""
              Du bist ein Planungsagent für ein KI-System.

              Deine Aufgabe ist es, anhand der Benutzeranfrage
              den geeignetsten Agenten auszuwählen.

              Verfügbare Agenten:
              {list_agents()}

              Benutzeranfrage:
              {state["user_prompt"]}

              Antworte ausschließlich als JSON:
              {{
                  "agent": "name_des_agenten"
              }}
        """)

        llm = get_llm(state["model"])
        structured_llm = llm.with_structured_output(
            PlannerResult
        )
        response = structured_llm.invoke(prompt)
        state["selected_agent"] = response.agent
        return state

class PlannerResult(BaseModel):
    agent: str