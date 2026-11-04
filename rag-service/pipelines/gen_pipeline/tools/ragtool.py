import textwrap

from langchain_core.tools import BaseTool

from generation import _build_filter, _source_label, _build_source
from pipelines.gen_pipeline.state import PipelineState
from vector_store import get_vectorstore

class RagTool(BaseTool):

    name:str = "rag_tool"
    description: str = textwrap.dedent("""
    Durchsucht hochgeladene Unterrichtsdokumente.

    Verwende dieses Tool, wenn Informationen aus PDFs oder
    hochgeladenen Lernmaterialien benötigt werden.

    search_term:
    - enthält den konkreten Begriff oder die Fragestellung,
    - nach der gesucht werden soll.

    Beispiel:
    {name: "rag_tool",args:{search_term: "funktionale Programmiersprachen"}}
    
    in search_term darf NIEMALS type: String stehen sondern NUR der search_term
    
    
    """)

    def __init__(self, state: PipelineState):
        super().__init__()
        self._state = state

    def _run(self, search_term):
        vs = get_vectorstore()
        print(self._state)
        results_with_score = vs.similarity_search_with_score(
            search_term,
            k=5,    # TODO
            #filter=_build_filter(self._state["user_id"], self._state["subject"]), TODO fix filter or wait for weaviate
        )

        context_parts = []
        sources = []
        for i, (doc, dist) in enumerate(results_with_score):
            ref_num = i + 1
            label = _source_label(doc, ref_num)
            context_parts.append(f"{label}:\n{doc.page_content}")
            sources.append(_build_source(doc, dist, ref_number=ref_num))

        print(context_parts)
        return context_parts