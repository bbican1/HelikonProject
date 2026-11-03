import json

from pydantic import BaseModel

from llm import get_llm
from pipelines.gen_pipeline.agents.explainer import ExplainerAgent


class PlannerAgent:

    def invoke(self, state):
        agents = [
            ExplainerAgent()
        ]
        available_agents = "\n".join(
            [
                f"""
                        Name: {agent.name}
                        Beschreibung: {agent.description}
                        """
                for agent in agents
            ]
        )

        prompt = f"""
              Du bist ein Planungsagent für ein KI-System.

              Deine Aufgabe ist es, anhand der Benutzeranfrage
              den geeignetsten Agenten auszuwählen.

              Verfügbare Agenten:
              {available_agents}

              Benutzeranfrage:
              {state["user_prompt"]}

              Antworte ausschließlich als JSON:
              {{
                  "agent": "name_des_agenten"
              }}
        """

        llm = get_llm(state["model"])
        structured_llm = llm.with_structured_output(
            PlannerResult
        )
        response = structured_llm.invoke(prompt)
        state["selected_agent"] = response.agent
        return state

class PlannerResult(BaseModel):
    agent: str