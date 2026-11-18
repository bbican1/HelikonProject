import textwrap

from langchain_core.messages import SystemMessage
from pydantic import BaseModel

from config import settings
from llm import get_llm
from pipelines.gen_pipeline.agents.base_agent import BaseAgent


class ValidatorAgent(BaseAgent):
    name = "validator_agent"
    description = "Validiert die Ausgaben anderer Agenten"
    prompt = """
        Du bist ein Validator-Agent für ein KI-System.
        
        Deine Aufgabe ist es die Ausgabe eines andern Agenten zu
        verifizieren. Dabei ist speziell auf zwei Dinge zu achten:
        
        1. Wurde das vorgegebene Format eingehalten
        2. Ist der Inhalt korrekt und nicht halluziniert.
        
        Zusätzlich erhälst du auch die Historie der generierung inkl. tool-calls.
        
        Du sollst entscheiden, ob eine Ausgabe so wie sie ist an den User
        weitergegeben werden kann, oder ob eine Überarbeitung notwendig ist.
        
        Antworte ausschließlich als JSON:
        {{
            "approved": true ODER false,
            "feedback": "Begründung nur wenn nicht approved"
        }}
    """

    def invoke(self, state):
        llm = get_llm(settings.gen_pipeline_validator_model)
        structured_llm = llm.with_structured_output(
            ValidatorResult
        )
        response = structured_llm.invoke(
            # TODO ganze history
            [SystemMessage(textwrap.dedent(self.prompt))] +
            state["messages"]
        )
        state["approved"] = response.approved
        state["validator_feedback"] = response.feedback
        return state

class ValidatorResult(BaseModel):
    approved: bool
    feedback: str