"""Extend the framework's typed policy configuration without changing its schema."""
from dataclasses import dataclass, field
from verl.workers.config.actor import PolicyLossConfig


@dataclass
class MathEPolicyLossConfig(PolicyLossConfig):
    math_e: dict = field(default_factory=dict)
