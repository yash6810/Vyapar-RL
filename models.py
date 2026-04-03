"""
models.py — Pydantic Action, Observation, and State models for Vyapar-RL.
"""
from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation, State


class GSTAction(Action):
    """Action taken by the agent in the GST environment."""
    task_id: str                          # "task1", "task2", or "task3"
    action_type: str = "submit_answer"    # always "submit_answer" for now
    answer: str                           # JSON string of the agent's answer


class GSTObservation(Observation):
    """What the agent sees after each step."""
    task_id: str
    task_name: str
    difficulty: str                       # "easy", "medium", "hard"
    description: str
    task_data: Dict[str, Any]             # task-specific payload (transactions, etc.)
    instructions: str
    feedback: str = ""                    # populated after step 2+
    score: float = 0.0                    # last step's reward
    step_number: int = 1
    max_steps: int = 3


class GSTState(State):
    """Episode metadata."""
    episode_id: str
    step_count: int = 0
    current_task_id: str = "task1"
    current_task_index: int = 0           # 0, 1, 2
    completed_tasks: List[str] = []
    total_score: float = 0.0
    is_complete: bool = False
