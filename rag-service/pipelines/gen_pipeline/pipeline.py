from pipelines.gen_pipeline.planner import PlannerAgent
from pipelines.gen_pipeline.agents.erklaer_agent import ErklaerAgent
from pipelines.gen_pipeline.agents.bewertungs_agent import BewertungsAgent
from pipelines.gen_pipeline.state import PipelineState

from langgraph.graph import StateGraph, START, END

def route_agent(state):
    return state["selected_agent"]

planner = PlannerAgent()
erklaer_agent = ErklaerAgent()
bewertungs_agent = BewertungsAgent()

builder = StateGraph(PipelineState)

builder.add_node("planner", planner.invoke)

builder.add_conditional_edges("planner", route_agent)

builder.add_node("erklaer_agent", erklaer_agent.invoke)
builder.add_node("bewertungs_agent", bewertungs_agent.invoke)
builder.add_edge(START, "planner")
builder.add_edge("erklaer_agent", END)

graph = builder.compile()

