from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    # Field(...) adds validation rules and documentation in one place
    question: str = Field(min_length=3, max_length=1000,
                          examples=["How many vacation days do employees get?"])
    k: int | None = Field(default=None, ge=1, le=10,
                          description="How many chunks to retrieve (default from settings)")


class SourceOut(BaseModel):
    source: str
    page: int
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
