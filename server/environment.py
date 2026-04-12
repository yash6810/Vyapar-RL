"""
server/environment.py — Core OpenEnv Environment implementation.
"""

import json
import random
import uuid
from pathlib import Path
from typing import Any

from openenv.core.env_server import Environment

from models import GSTAction, GSTObservation, GSTState
from server.graders import grade_task1, grade_task2, grade_task3, compute_reward


DATA_DIR = Path(__file__).parent.parent / "data"

TASK_CONFIGS = [
    {
        "task_id": "task1",
        "name": "Transaction GST Classifier",
        "difficulty": "easy",
        "max_steps": 3,
        "data_file": "task1_easy.json",
    },
    {
        "task_id": "task2",
        "name": "Quarterly GST Liability Calculator",
        "difficulty": "medium",
        "max_steps": 4,
        "data_file": "task2_medium.json",
    },
    {
        "task_id": "task3",
        "name": "GSTR-1 vs GSTR-2A Reconciliation",
        "difficulty": "hard",
        "max_steps": 5,
        "data_file": "task3_hard.json",
    },
]

INSTRUCTIONS = {
    "task1": 'Return a JSON object mapping transaction IDs to GST slabs. Example: {"1": 18, "2": 5, "3": 0}. Valid slabs: 0, 5, 12, 18, 28.',
    "task2": "Return a JSON with keys: total_sales_value, total_purchase_value, cgst_payable, sgst_payable, igst_payable, total_itc, net_gst_liability. All values in INR.",
    "task3": 'Return a JSON with "mismatches" (list of objects with invoice_no, mismatch_type, gstr1_tax_amount, gstr2a_tax_amount, itc_at_risk) and "total_itc_at_risk". mismatch_type must be one of: amount_mismatch, missing_in_gstr2a, extra_in_gstr2a.',
}


class GSTEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self._current_task_index = 0
        self._current_episode_data = None
        self._current_golden = None
        self._last_feedback = ""
        self._task_datasets = self._load_all_data()
        self._prng = random.Random()
        # Pick an initial episode so _current_golden is never None
        episode = self._prng.choice(self._task_datasets["task1"])
        self._current_episode_data = episode
        self._current_golden = episode.get("golden_answer", {})
        self._state = GSTState(episode_id=str(uuid.uuid4()), step_count=0)

    def _load_all_data(self) -> dict:
        datasets = {}
        for cfg in TASK_CONFIGS:
            path = DATA_DIR / cfg["data_file"]
            with open(path, "r") as f:
                datasets[cfg["task_id"]] = json.load(f)
        return datasets

    def _get_task_config(self) -> dict:
        return TASK_CONFIGS[self._current_task_index]

    def _get_task_data_key(self) -> str:
        return self._get_task_config()["task_id"]

    def _pick_episode(self) -> dict:
        task_id = self._get_task_data_key()
        episodes = self._task_datasets[task_id]
        return self._prng.choice(episodes)

    def reset(
        self, seed=None, episode_id=None, task_index=None, **kwargs
    ) -> GSTObservation:
        if seed is not None:
            self._prng = random.Random(seed)
        else:
            self._prng = random.Random()

        # Allow caller to specify which task to start with (0, 1, or 2)
        if task_index is not None and 0 <= task_index <= 2:
            self._current_task_index = task_index
        else:
            self._current_task_index = 0

        episode = self._pick_episode()
        self._current_episode_data = episode
        self._current_golden = episode.get("golden_answer", {})
        self._last_feedback = ""

        cfg = self._get_task_config()
        self._state = GSTState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_task_id=cfg["task_id"],
            current_task_index=self._current_task_index,
            completed_tasks=[],
            total_score=0.0,
            is_complete=False,
        )

        return self._build_observation(reward=0.0, done=False)

    def step(self, action, **kwargs) -> GSTObservation:
        self._state.step_count += 1
        cfg = self._get_task_config()
        task_id = cfg["task_id"]
        max_steps = cfg["max_steps"]

        # Handle both GSTAction objects and dicts from GenericEnvClient
        if isinstance(action, dict):
            answer_str = action.get("answer", "{}")
        else:
            answer_str = action.answer

        # Grade the action
        task_score, feedback = self._grade_action(answer_str, task_id)
        reward = compute_reward(task_score, self._state.step_count, is_valid=True)
        self._last_feedback = feedback
        self._state.total_score = round(self._state.total_score + reward, 4)

        # Check if this task episode is done
        task_done = (task_score >= 0.95) or (self._state.step_count >= max_steps)
        all_done = False

        if task_done:
            self._state.completed_tasks.append(task_id)
            # Move to next task if there is one
            if self._current_task_index < len(TASK_CONFIGS) - 1:
                self._current_task_index += 1
                episode = self._pick_episode()
                self._current_episode_data = episode
                self._current_golden = episode.get("golden_answer", {})
                self._last_feedback = f"Task {task_id} complete (score={task_score}). Moving to next task."
                self._state.step_count = 0
                self._state.current_task_id = TASK_CONFIGS[self._current_task_index][
                    "task_id"
                ]
                self._state.current_task_index = self._current_task_index
            else:
                all_done = True
                self._state.is_complete = True

        return self._build_observation(reward=reward, done=all_done)

    def _grade_action(self, answer_str: str, task_id: str) -> tuple[float, str]:
        if task_id == "task1":
            return grade_task1(answer_str, self._current_golden)
        elif task_id == "task2":
            return grade_task2(answer_str, self._current_golden)
        elif task_id == "task3":
            return grade_task3(answer_str, self._current_golden)
        return 0.0, "Unknown task"

    def _build_observation(self, reward: float, done: bool) -> GSTObservation:
        cfg = self._get_task_config()
        task_id = cfg["task_id"]

        # Build task_data payload (remove golden_answer before sending to agent)
        task_data = {}
        if self._current_episode_data:
            task_data = {
                k: v
                for k, v in self._current_episode_data.items()
                if k != "golden_answer"
            }

        return GSTObservation(
            task_id=task_id,
            task_name=cfg["name"],
            difficulty=cfg["difficulty"],
            description=f"GST compliance task: {cfg['name']}",
            task_data=task_data,
            instructions=INSTRUCTIONS[task_id],
            feedback=self._last_feedback,
            score=reward,
            step_number=self._state.step_count,
            max_steps=cfg["max_steps"],
            reward=reward,
            done=done,
            metadata={
                "task_id": task_id,
                "difficulty": cfg["difficulty"],
                "step": self._state.step_count,
                "total_score": self._state.total_score,
            },
        )

    @property
    def state(self) -> GSTState:
        return self._state
