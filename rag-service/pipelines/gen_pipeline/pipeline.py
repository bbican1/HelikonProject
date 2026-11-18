from typing import Literal

from langchain_core.messages import ToolMessage

from pipelines.gen_pipeline.agents.agent_registry import get_all_agents
from pipelines.gen_pipeline.planner import PlannerAgent
from pipelines.gen_pipeline.state import PipelineState

from langgraph.graph import StateGraph, START, END

from pipelines.gen_pipeline.tools.tool_registry import get_tool


def tool_node(state: PipelineState):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = get_tool(tool_call["name"])
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: PipelineState) -> Literal["tool_node", "finalize"]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)#
    return "finalize"

def finalize_answer(state: PipelineState):
    """Sets the last message as Answer"""
    last_message = state["messages"][-1]

    return {
        "answer": last_message.content
    }

planner = PlannerAgent()

builder = StateGraph(PipelineState)

builder.add_edge(START, "planner")
builder.add_node("planner", planner.invoke)

builder.add_conditional_edges("planner", lambda state: state["selected_agent"])
builder.add_node("tool_node", tool_node)
builder.add_conditional_edges("tool_node", lambda state: state["selected_agent"])

for agent in get_all_agents():
    builder.add_node(agent.name, agent.invoke)
    builder.add_conditional_edges(agent.name, should_continue,["tool_node", "finalize"])

builder.add_node("finalize", finalize_answer)
builder.add_edge("finalize", END)

graph = builder.compile()



