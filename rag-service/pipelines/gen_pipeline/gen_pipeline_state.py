from langgraph.graph import MessagesState


class GenPipelineState(MessagesState):
    model: str
    selected_agent: str
    answer: str
    validator_feedback: str
    approved: bool
    retry_count: int
    user_id: str    # TODO check
    subject: str
    prompt_ok: bool
    prompt_reason: str