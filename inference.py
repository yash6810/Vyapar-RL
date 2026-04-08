"""
inference.py — Baseline inference script for Vyapar-RL.
MUST be in root directory. MUST be named exactly inference.py.
Uses OpenAI client library pointed at HuggingFace Router.
Runtime: < 20 minutes on 2 vCPU / 8GB RAM (no GPU needed).

STDOUT FORMAT (mandatory):
  [START] task=<task_name> env=Vyapar-RL model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<float> rewards=<r1,r2,...,rn>
"""

import json
import os
import sys
import re
import time
import urllib.request
import urllib.error
from typing import List, Optional, Dict
from dataclasses import dataclass, field

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
FALLBACK_MODELS = [
    MODEL_NAME,
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-2b-it",
]
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK = "Vyapar-RL"
SUCCESS_SCORE_THRESHOLD = 0.3


@dataclass
class HTTPEnvResult:
    """Simple result object mimicking GenericEnvClient response."""
    observation: dict = field(default_factory=dict)
    reward: float = 0.0
    done: bool = False


class HTTPEnvClient:
    """
    HTTP-based environment client that uses POST requests instead of WebSocket.
    This avoids WebSocket connection issues on HuggingFace Spaces.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _post(self, endpoint: str, data: dict) -> dict:
        """Make a POST request and return parsed JSON response."""
        url = f"{self.base_url}/{endpoint}"
        payload = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def reset(self, **kwargs) -> HTTPEnvResult:
        """POST /reset and return observation."""
        resp = self._post("reset", kwargs)
        obs = resp.get("observation", resp)
        return HTTPEnvResult(
            observation=obs,
            reward=obs.get("reward", 0.0),
            done=obs.get("done", False),
        )

    def step(self, action: dict) -> HTTPEnvResult:
        """POST /step with action and return observation."""
        resp = self._post("step", {"action": action})
        obs = resp.get("observation", resp)
        return HTTPEnvResult(
            observation=obs,
            reward=obs.get("reward", obs.get("score", 0.0)),
            done=obs.get("done", False),
        )

if not API_KEY:
    print("ERROR: HF_TOKEN environment variable not set.")
    print("Get your token at: https://huggingface.co/settings/tokens")
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are an expert Indian GST compliance assistant with deep knowledge of:
- GST slab rates (0%, 5%, 12%, 18%, 28%) and which goods/services fall in each
- Quarterly GST liability computation (CGST, SGST, IGST, ITC)
- GSTR-1 vs GSTR-2A reconciliation and mismatch detection

Before answering, write a <thought>...</thought> block analyzing the mathematical operations required. Then, output your final answer as ONLY a valid JSON object.
Be precise with numbers. Indian tax laws are strict about accuracy."""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def call_llm(messages: List[Dict[str, str]]) -> str:
    last_error = Exception("All models failed")
    for model in FALLBACK_MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1500,
                temperature=0.1,
            )
            raw_content = response.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            if match:
                raw_content = match.group(0).strip()
            return raw_content
        except Exception as e:
            last_error = e
            print(f"Model {model} failed: {e}", flush=True)
            continue
    raise last_error


def build_prompt(obs_data: dict) -> str:
    task_name = obs_data.get("task_name", "")
    task_data = obs_data.get("task_data", {})
    instruct = obs_data.get("instructions", "")
    feedback = obs_data.get("feedback", "")

    prompt = f"""TASK: {task_name}

TASK DATA:
{json.dumps(task_data, indent=2)}

INSTRUCTIONS:
{instruct}
"""
    if feedback:
        prompt += (
            f"\nPREVIOUS FEEDBACK (use this to improve your answer):\n{feedback}\n"
        )

    prompt += "\nRespond with your <thought> block followed by ONLY the JSON object, nothing else."
    return prompt


def run_task(env_client, task_name: str, task_index: int) -> List[float]:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env_client.reset(task_index=task_index)
        obs = result.observation if hasattr(result, "observation") else result
    except Exception as e:
        print(f"ERROR: Failed to reset environment: {e}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return []

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        for step in range(1, 11):
            try:
                if isinstance(obs, dict):
                    full_obs = obs
                else:
                    full_obs = {
                        "task_id": getattr(obs, "task_id", ""),
                        "task_name": getattr(obs, "task_name", ""),
                        "task_data": getattr(obs, "task_data", {}),
                        "instructions": getattr(obs, "instructions", ""),
                        "feedback": getattr(obs, "feedback", ""),
                    }

                prompt = build_prompt(full_obs)
                messages.append({"role": "user", "content": prompt})

                try:
                    answer = call_llm(messages)
                    messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    print(f"API ERROR: {e}", flush=True)
                    answer = "{}"
                    messages.append({"role": "assistant", "content": answer})

                action = {
                    "task_id": full_obs.get("task_id", f"task{task_index + 1}"),
                    "action_type": "submit_answer",
                    "answer": answer,
                }

                result = env_client.step(action)
                obs = result.observation if hasattr(result, "observation") else result
            except Exception as e:
                print(f"STEP ERROR at step {step}: {type(e).__name__}: {e}", flush=True)
                import traceback

                traceback.print_exc()
                break

            if hasattr(result, "reward"):
                reward = result.reward or 0.0
            elif isinstance(obs, dict):
                reward = obs.get("score", obs.get("reward", 0.0)) or 0.0
            else:
                reward = 0.0

            if hasattr(result, "done"):
                done = result.done
            elif isinstance(obs, dict):
                done = obs.get("done", False)
            else:
                done = False

            error = None
            rewards.append(reward)
            steps_taken = step

            action_str = answer.replace("\n", " ")[:200]
            log_step(
                step=step, action=action_str, reward=reward, done=done, error=error
            )

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"TASK ERROR: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return rewards


def wait_for_env(base_url: str, timeout: int = 180, interval: int = 5) -> bool:
    health_url = base_url.rstrip("/")
    deadline = time.time() + timeout
    print(f"Waiting for environment at {health_url} ...", flush=True)
    while time.time() < deadline:
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                print(f"Environment ready! (Status: {resp.status})", flush=True)
                return True
        except urllib.error.HTTPError as e:
            print(f"Environment ready! (Status: {e.code})", flush=True)
            return True
        except urllib.error.URLError:
            pass
        except Exception:
            pass
        time.sleep(interval)
    print(
        f"WARNING: Environment did not respond within {timeout}s, attempting connection anyway.",
        flush=True,
    )
    return True


def main():
    wait_for_env(ENV_URL)

    all_rewards = []

    task_configs = [
        ("Transaction-GST-Classifier", 0),
        ("Quarterly-GST-Liability-Calculator", 1),
        ("GSTR1-vs-GSTR2A-Reconciliation", 2),
    ]

    max_retries = 25
    retry_delay = 12
    for attempt in range(1, max_retries + 1):
        try:
            env = HTTPEnvClient(base_url=ENV_URL)
            for task_name, task_index in task_configs:
                task_rewards = run_task(env, task_name, task_index)
                all_rewards.extend(task_rewards)
            break
        except (KeyboardInterrupt, SystemExit):
            print("Exiting...", flush=True)
            sys.exit(0)
        except Exception as e:
            import traceback

            print(
                f"Connection attempt {attempt}/{max_retries} failed: {type(e).__name__}: {e}",
                flush=True,
            )
            traceback.print_exc()
            if attempt < max_retries:
                print(f"Retrying in {retry_delay}s...", flush=True)
                time.sleep(retry_delay)
            else:
                print(
                    f"ERROR: All {max_retries} connection attempts failed. Exiting gracefully.",
                    flush=True,
                )
                overall_avg = 0.0
                result = {
                    "overall_avg": overall_avg,
                    "all_rewards": [],
                }
                print("\nJSON_RESULT:", json.dumps(result))
                sys.exit(0)

    overall_avg = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    result = {
        "overall_avg": round(overall_avg, 4),
        "all_rewards": [round(r, 4) for r in all_rewards],
    }
    print("\nJSON_RESULT:", json.dumps(result))


if __name__ == "__main__":
    main()
