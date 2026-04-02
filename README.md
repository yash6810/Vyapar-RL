# vyapar-gst-env

**An OpenEnv reinforcement learning environment for Indian GST compliance.**

Built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) by Meta & Hugging Face.
Derived from [Vyapar](https://github.com/yash6810/Vyapar) — a production WhatsApp AI
assistant for Indian SMEs. Team: Code Nexus (Yash Upadhyay + Krishna Garg).

---

## 1. Environment Description & Motivation

India has 14 million GST-registered businesses. Every quarter, they must:
1. Classify every expense/sale into the correct GST slab (0–28%)
2. Compute CGST, SGST, IGST payable and claim Input Tax Credit
3. Reconcile their GSTR-1 (filed sales) against GSTR-2A (supplier-declared purchases)

This environment trains AI agents to perform these tasks at expert level.
It was derived from Vyapar, a real WhatsApp assistant used by Indian SMEs for
GST compliance tracking. The logic, data schemas, and GST rules are grounded in
production use, not synthetic toy examples.

---

## 2. Action Space

```python
class GSTAction(Action):
    task_id: str           # "task1", "task2", or "task3"
    action_type: str       # always "submit_answer"
    answer: str            # JSON string containing the agent's answer
```

The `answer` field is always a JSON string. Format varies by task (see Task Descriptions).

---

## 3. Observation Space

```python
class GSTObservation(Observation):
    task_id: str
    task_name: str
    difficulty: str        # "easy", "medium", or "hard"
    description: str
    task_data: dict        # task-specific payload (transactions, GSTR data, etc.)
    instructions: str      # exact format the agent must return
    feedback: str          # populated after step 2+ with what went wrong
    score: float           # reward from last step [0.0, 1.0]
    step_number: int
    max_steps: int
```

---

## 4. Task Descriptions

### Task 1 — Transaction GST Classifier (Easy)
- **Input:** 10 business transactions with descriptions and amounts
- **Output:** JSON mapping transaction IDs to correct GST slab (0/5/12/18/28)
- **Grader:** Exact slab match per transaction. `score = correct / total`
- **Max steps:** 3 | **Reward diversity:** Always unique (partial matches)

### Task 2 — Quarterly GST Liability Calculator (Medium)
- **Input:** 12–15 transactions (sales + purchases) with taxable values and slabs
- **Output:** JSON with 7 computed fields: total_sales_value, total_purchase_value,
  cgst_payable, sgst_payable, igst_payable, total_itc, net_gst_liability
- **Grader:** Proportional error per field (within 1% = full credit)
- **Max steps:** 4 | **Reward diversity:** Math-based, always unique float

### Task 3 — GSTR-1 vs GSTR-2A Reconciliation (Hard)
- **Input:** Two lists of invoice records (GSTR-1 filed, GSTR-2A received)
- **Output:** JSON with all mismatches (by type) and total ITC at risk
- **Grader:** Detection (40%) + type classification (30%) + ITC computation (30%)
- **Max steps:** 5 | **Reward diversity:** Multi-component weighted score

---

## 5. Setup & Usage

### Local development
```bash
# Install OpenEnv
pip install openenv-core

# Clone and install
git clone https://github.com/yash6810/vyapar-gst-env
cd vyapar-gst-env
pip install -e .

# Run server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Test
python -c "
from client import GSTEnvClient
from models import GSTAction
import json

with GSTEnvClient(base_url='http://localhost:8000').sync() as env:
    obs = env.reset()
    print('Task:', obs.task_name)
    print('Instructions:', obs.instructions)
    action = GSTAction(task_id='task1', action_type='submit_answer',
                       answer=json.dumps({'1': 18, '2': 18}))
    obs = env.step(action)
    print('Reward:', obs.reward)
"
```

### Run baseline inference
```bash
export HF_TOKEN=your_huggingface_token
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
export ENV_URL=https://yash6810-vyapar-gst-env.hf.space

python inference.py
```

### Docker
```bash
docker build -t vyapar-gst-env .
docker run -p 8000:8000 vyapar-gst-env
```

---

## 6. Baseline Scores

Run with `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace Router:

| Task | Difficulty | Avg Reward | Notes |
|------|------------|------------|-------|
| Task 1 | Easy | ~ 0.36 | Struggles with zero-rated slab classification |
| Task 2 | Medium | ~ 0.02 | High error rate in math and ITC offsetting |
| Task 3 | Hard | ~ 0.01 | Completely failed GSTR-2A reconciliation logic |
| **Overall** | — | **0.0788** | JSON_RESULT: {"overall_avg": 0.0788} |

> As expected, base reasoning models perform poorly on strict Indian GST computational challenges without specialized tool-use or RAG.
