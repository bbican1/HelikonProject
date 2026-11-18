import textwrap

from ..agents.base_agent import BaseAgent
from llm import get_llm


class BewertungsAgent(BaseAgent):
    name = "bewertungs_agent"
    description = "Bewertet eine Aufgabe"
    tool_names = ["rag_tool"]
    prompt = """
                Du bist ein Bewertungsagent.
                WICHTIG:
                Bevor du antwortest, prüfe ob du Informationen aus
                den hochgeladenen Dokumenten benötigst.

                Wenn die Frage Bezug auf ein PDF, Skript oder Lernmaterial nimmt,
                musst du zuerst das Tool rag_tool verwenden.
                Bewerte folgenden User-Input nachvollziehbar:
            """

    def invoke(self, state):
        return self.basic_response(state, self.prompt)