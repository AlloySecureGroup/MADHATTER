import pytest
from pydantic import ValidationError

from app.schemas import AttackRequest, TrainExample, TrainRequest


def test_discrete_attack_schema():
    req = AttackRequest(prompt="hello", mode="token_hotflip", max_token_changes=2)
    assert req.mode == "token_hotflip"
    assert req.max_token_changes == 2


def test_attack_rejects_empty_prompt():
    with pytest.raises(ValidationError):
        AttackRequest(prompt="")


def test_robust_training_schema():
    req = TrainRequest(
        training_goal="robust",
        attack_mode="token_hotflip",
        examples=[TrainExample(prompt="p", response="r")],
    )
    assert req.training_goal == "robust"
