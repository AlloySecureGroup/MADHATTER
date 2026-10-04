from typing import Literal
from pydantic import BaseModel, Field

from .model_registry import DEFAULT_MODEL_ID


class LoadRequest(BaseModel):
    model_id: str = Field(default=DEFAULT_MODEL_ID, min_length=1)


class AttackRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=12000)
    mode: Literal["embedding_pgd", "token_hotflip"] = "embedding_pgd"
    epsilon: float = Field(default=0.03, gt=0.0, le=2.0)
    step_size: float = Field(default=0.0075, gt=0.0, le=1.0)
    steps: int = Field(default=12, ge=1, le=100)
    norm: Literal["linf", "l2"] = "linf"
    objective: Literal["next_token_kl", "hidden_cosine", "combined"] = "combined"
    hidden_weight: float = Field(default=1.0, ge=0.0, le=100.0)
    top_k: int = Field(default=8, ge=3, le=30)
    random_start: bool = True
    max_token_changes: int = Field(default=3, ge=1, le=20)
    candidate_top_k: int = Field(default=8, ge=2, le=64)
    editable_last_n: int = Field(default=64, ge=1, le=512)


class TrainExample(BaseModel):
    prompt: str = Field(min_length=1, max_length=12000)
    response: str = Field(min_length=1, max_length=12000)


class TrainRequest(BaseModel):
    training_goal: Literal["robust", "sensitive"] = "sensitive"
    attack_mode: Literal["embedding_pgd", "token_hotflip"] = "embedding_pgd"
    examples: list[TrainExample] = Field(min_length=1, max_length=500)
    epochs: int = Field(default=1, ge=1, le=20)
    learning_rate: float = Field(default=2e-4, gt=0.0, le=0.1)
    epsilon: float = Field(default=0.02, gt=0.0, le=2.0)
    step_size: float = Field(default=0.005, gt=0.0, le=1.0)
    attack_steps: int = Field(default=4, ge=1, le=30)
    norm: Literal["linf", "l2"] = "linf"
    robust_lambda: float = Field(default=0.5, ge=0.0, le=100.0)
    lora_rank: int = Field(default=8, ge=1, le=128)
    lora_alpha: int = Field(default=16, ge=1, le=512)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    max_length: int = Field(default=512, ge=32, le=4096)
    max_token_changes: int = Field(default=2, ge=1, le=12)
    candidate_top_k: int = Field(default=6, ge=2, le=32)
    editable_last_n: int = Field(default=64, ge=1, le=512)
    seed: int = 7
