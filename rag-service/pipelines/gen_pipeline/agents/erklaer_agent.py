import textwrap

from langchain_core.messages import SystemMessage, HumanMessage

from llm import get_llm
from ..agents.base_agent import BaseAgent
from ..tools.ragtool import RagTool


class ErklaerAgent(BaseAgent):
    name = "erklaer_agent"
    description = "Erklärt ein Thema"
    tool_names = ["rag_tool"]
    sys_prompt = """
            Du bist ein Erkläragent.
            WICHTIG:
            Bevor du antwortest, prüfe ob du Informationen aus
            den hochgeladenen Dokumenten benötigst.
            
            Wenn die Frage Bezug auf ein PDF, Skript oder Lernmaterial nimmt,
            musst du zuerst das Tool rag_tool verwenden.
            Erkläre dem User die folgende Frage verständlch:
        """


    def invoke(self, state):
        return self.basic_response(state)