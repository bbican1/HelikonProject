from typing import Literal

from langchain_core.messages import ToolMessage

from llm import get_llm
from pipelines.gen_pipeline.agents.agent_registry import get_all_agents
from pipelines.gen_pipeline.planner import PlannerAgent
from pipelines.gen_pipeline.state import PipelineState

from langgraph.graph import StateGraph, START, END

from pipelines.gen_pipeline.tools.tool_registry import get_tool


def route_agent(state):
    return state["selected_agent"]


def tool_node(state: PipelineState):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = get_tool(tool_call["name"])
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: PipelineState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

planner = PlannerAgent()

builder = StateGraph(PipelineState)

builder.add_node("planner", planner.invoke)

builder.add_conditional_edges("planner", route_agent)
builder.add_node("tool_node", tool_node)

for agent in get_all_agents():
    builder.add_node(agent.name, agent.invoke)
    #builder.add_conditional_edges(agent.name, tools_condition)
    builder.add_conditional_edges(agent.name, should_continue,["tool_node", END])
builder.add_edge("tool_node", "erklaer_agent")


builder.add_edge(START, "planner")
builder.add_edge("erklaer_agent", END)

graph = builder.compile()



