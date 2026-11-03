from ..agents.base_agent import BaseAgent
from llm import get_llm


class BewertungsAgent(BaseAgent):
    name = "bewertungs_agent"
    description = "Bewertet eine Aufgabe"

    def invoke(self, state):
        prompt = f"""
            Du bist ein Bewertungsagent. Bewerte folgende Aufgabe eines Users:
            {state["user_prompt"]}
        """

        response = get_llm(state["model"]).invoke(prompt)
        state["answer"] = response.content

        return state