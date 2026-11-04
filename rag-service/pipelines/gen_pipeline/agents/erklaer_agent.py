import textwrap

from ..agents.base_agent import BaseAgent
from llm import get_llm
from ..tools.ragtool import RagTool


class ErklaerAgent(BaseAgent):
    name = "erklaer_agent"
    description = "Erklärt ein Thema"


    def invoke(self, state):
        tools = [RagTool(state)]
        llm = get_llm(state["model"]).bind_tools(tools)

        prompt = textwrap.dedent(f"""
            Du bist ein Erkläragent.
            WICHTIG:
            Bevor du antwortest, prüfe ob du Informationen aus
            den hochgeladenen Dokumenten benötigst.
            
            Wenn die Frage Bezug auf ein PDF, Skript oder Lernmaterial nimmt,
            musst du zuerst das Tool rag_tool verwenden.
            Erkläre dem User die folgende Frage verständlch:
            {state["user_prompt"]}
        """)

        response = llm.invoke(prompt)
        print(response.tool_calls)

        if response.tool_calls:
            tool_call = response.tool_calls[0]

            tool = tools[0]

            result = tool.invoke(
                tool_call["args"]
            )

            print("RAG Ergebnis:")
            print(result)

        state["answer"] = response.content

        return state