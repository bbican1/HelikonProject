import json

from pipelines.gen_pipeline.agents.base_agent import BaseAgent
from pipelines.gen_pipeline.agents.bewertungs_agent import BewertungsAgent
from pipelines.gen_pipeline.agents.erklaer_agent import ErklaerAgent

AGENTS = [
    ErklaerAgent,
    BewertungsAgent
]

def list_agents() -> str:
    return json.dumps([
        {
            "name": agent.name,
            "description": agent.description
        } for agent in AGENTS
    ],
    ensure_ascii=False)

def get_agent(name: str) -> BaseAgent:
    for agent in AGENTS:
        if agent.name == name:
            return agent()
    raise ValueError(f"Unknown agent: {name}")

def get_all_agents() -> list[BaseAgent]:
    return [agent() for agent in AGENTS]