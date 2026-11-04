from typing import TypedDict

class PipelineState(TypedDict):
    model: str
    user_prompt: str
    selected_agent: str
    answer: str
    validator_feedback: str
    approved: bool
    retry_count: int
    user_id: str    # TODO check
    subject: str