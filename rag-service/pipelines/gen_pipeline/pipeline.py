from pipelines.gen_pipeline.planner import PlannerAgent
from pipelines.gen_pipeline.agents.explainer import ExplainerAgent
from pipelines.gen_pipeline.state import PipelineState

from langgraph.graph import StateGraph, START, END

def route_agent(state):
    return state["selected_agent"]

planner = PlannerAgent()
explainer = ExplainerAgent()

builder = StateGraph(PipelineState)

builder.add_node("planner", planner.invoke)

builder.add_conditional_edges("planner", route_agent)

builder.add_node("explainer", explainer.invoke)
builder.add_edge(START, "planner")
builder.add_edge("explainer", END)

graph = builder.compile()

