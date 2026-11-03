from ..agents.base_agent import BaseAgent
from llm import get_llm


class ErklaerAgent(BaseAgent):
    name = "erklaer_agent"
    description = "Erklärt ein Thema"

    def invoke(self, state):
        prompt = f"""
            Du bist ein Erkläragent. Erkläre dem User die folgende Frage verständlch:
            {state["user_prompt"]}
        """

        response = get_llm(state["model"]).invoke(prompt)
        state["answer"] = response.content

        return state