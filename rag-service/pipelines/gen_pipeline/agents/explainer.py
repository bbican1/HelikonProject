from ..agents.base import BaseAgent
from llm import get_llm


class ExplainerAgent(BaseAgent):
    name = "explainer"
    description = "Erklärt ein Thema"

    def invoke(self, state):
        prompt = f"""
            Du bist ein Erkläragent. Erkläre dem User die folgende Frage verständlch:
            {state["user_prompt"]}
        """

        response = get_llm(state["model"]).invoke(prompt)
        state["answer"] = response.content

        return state