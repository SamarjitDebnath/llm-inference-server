from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The input text for the model")
    
    max_tokens: int | None = Field(
        default=None, 
        ge=1, 
        le=2048, 
        description="Must be between 1 and 2048"
    )
    
    temperature: float | None = Field(
        default=None, 
        ge=0.0, 
        le=2.0, 
        description="Standard LLM temperature range is usually 0.0 to 2.0"
    )

class BatchGenerateRequest(BaseModel):
    requests: list[GenerateRequest] = Field(..., min_length=1, description="A list of generation requests to batch")

class BatchGenerateResponse(BaseModel):
    outputs: list[str] = Field(..., description="Decoded outputs for each item in the batch")
    batch_size: int = Field(..., description="Number of requests processed in the batch")
    queue_latency_ms: float | None = Field(
        default=None,
        description="Average queue latency in milliseconds for this batch; null if no queue latency data is available"
    )
    token_throughput_per_sec: float | None = Field(
        default=None,
        description="Token throughput measured in tokens per second; null if no throughput data is available"
    )

class HealthResponse(BaseModel):
    status: str = Field(..., description="The health status of the server")
