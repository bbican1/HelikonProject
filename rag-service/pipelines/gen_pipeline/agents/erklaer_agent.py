import textwrap

from langchain_core.messages import SystemMessage, HumanMessage

from llm import get_llm
from ..agents.base_agent import BaseAgent
from ..tools.ragtool import RagTool


class ErklaerAgent(BaseAgent):
    name = "erklaer_agent"
    description = "Erklärt ein Thema"
    tool_names = ["rag_tool"]


    def invoke(self, state):
        llm = self.build_llm_with_tools(state["model"])
        prompt = SystemMessage(textwrap.dedent(f"""
            Du bist ein Erkläragent.
            WICHTIG:
            Bevor du antwortest, prüfe ob du Informationen aus
            den hochgeladenen Dokumenten benötigst.
            
            Wenn die Frage Bezug auf ein PDF, Skript oder Lernmaterial nimmt,
            musst du zuerst das Tool rag_tool verwenden.
            Erkläre dem User die folgende Frage verständlch:
        """))

        response = llm.invoke([prompt] + [HumanMessage(state["user_prompt"])] + state["messages"])
        print(response.tool_calls)

        return {
            "messages": [response]
        }