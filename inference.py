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
from typing import List, Optional, Dict

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:8000")
BENCHMARK    = "Vyapar-RL"
SUCCESS_SCORE_THRESHOLD = 0.3

if not API_KEY:
    print("ERROR: HF_TOKEN environment variable not set.")
    print("Get your token at: https://huggingface.co/settings/tokens")
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ─── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Indian GST compliance assistant with deep knowledge of:
- GST slab rates (0%, 5%, 12%, 18%, 28%) and which goods/services fall in each
- Quarterly GST liability computation (CGST, SGST, IGST, ITC)
- GSTR-1 vs GSTR-2A reconciliation and mismatch detection

Before answering, write a <thought>...</thought> block analyzing the mathematical operations required. Then, output your final answer as ONLY a valid JSON object.
Be precise with numbers. Indian tax laws are strict about accuracy."""


# ─── Mandatory stdout logging ────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─── LLM helpers ─────────────────────────────────────────────────────────────
def call_llm(messages: List[Dict[str, str]]) -> str:
    """Call the LLM using a stateful messages trajectory and safely extract JSON."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=1500,
        temperature=0.1,
    )
    raw_content = response.choices[0].message.content.strip()
    match = re.search(r'\{.*\}', raw_content, re.DOTALL)
    if match:
        raw_content = match.group(0).strip()
    return raw_content


def build_prompt(obs_data: dict) -> str:
    """Build a clear prompt from the observation dict."""
    task_name = obs_data.get("task_name", "")
    task_data = obs_data.get("task_data", {})
    instruct  = obs_data.get("instructions", "")
    feedback  = obs_data.get("feedback", "")

    prompt = f"""TASK: {task_name}

TASK DATA:
{json.dumps(task_data, indent=2)}

INSTRUCTIONS:
{instruct}
"""
    if feedback:
        prompt += f"\nPREVIOUS FEEDBACK (use this to improve your answer):\n{feedback}\n"

    prompt += "\nRespond with your <thought> block followed by ONLY the JSON object, nothing else."
    return prompt


# ─── Task runner ──────────────────────────────────────────────────────────────
def run_task(env_client, task_name: str, task_index: int) -> List[float]:
    """Run a single task, emitting [START]/[STEP]/[END] logs."""
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    # GenericEnvClient returns result objects with .observation (dict), .reward, .done
    result = env_client.reset(task_index=task_index)
    obs = result.observation if hasattr(result, 'observation') else result
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # Initialize memory trajectory
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        for step in range(1, 11):  # Max 10 steps per task
            # obs is a dict when using GenericEnvClient
            if isinstance(obs, dict):
                full_obs = obs
            else:
                full_obs = {
                    "task_id":      getattr(obs, "task_id",      ""),
                    "task_name":    getattr(obs, "task_name",    ""),
                    "task_data":    getattr(obs, "task_data",    {}),
                    "instructions": getattr(obs, "instructions", ""),
                    "feedback":     getattr(obs, "feedback",     ""),
                }

            prompt = build_prompt(full_obs)
            messages.append({"role": "user", "content": prompt})

            try:
                answer = call_llm(messages)
                messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                print(f"API ERROR: {e}")
                answer = "{}"
                # Append fallback to keep trajectory aligned
                messages.append({"role": "assistant", "content": answer}) 

            # GenericEnvClient sends actions as dicts
            action = {
                "task_id": full_obs.get("task_id", f"task{task_index + 1}"),
                "action_type": "submit_answer",
                "answer": answer,
            }

            result = env_client.step(action)
            obs = result.observation if hasattr(result, 'observation') else result

            if hasattr(result, 'reward'):
                reward = result.reward or 0.0
            elif isinstance(obs, dict):
                reward = obs.get("score", obs.get("reward", 0.0)) or 0.0
            else:
                reward = 0.0

            if hasattr(result, 'done'):
                done = result.done
            elif isinstance(obs, dict):
                done = obs.get("done", False)
            else:
                done = False

            error = None
            rewards.append(reward)
            steps_taken = step

            # Sanitize action string for single-line log output
            action_str = answer.replace("\n", " ")[:200]
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return rewards


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    from openenv import GenericEnvClient

    all_rewards = []

    task_configs = [
        ("Transaction-GST-Classifier",         0),
        ("Quarterly-GST-Liability-Calculator",  1),
        ("GSTR1-vs-GSTR2A-Reconciliation",      2),
    ]

    with GenericEnvClient(base_url=ENV_URL).sync() as env:
        for task_name, task_index in task_configs:
            task_rewards = run_task(env, task_name, task_index)
            all_rewards.extend(task_rewards)

    # Machine-readable summary
    overall_avg = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    result = {
        "overall_avg": round(overall_avg, 4),
        "all_rewards": [round(r, 4) for r in all_rewards],
    }
    print("\nJSON_RESULT:", json.dumps(result))


if __name__ == "__main__":
    main()
