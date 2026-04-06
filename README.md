---
title: Vyapar RL
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - gst
  - india
  - tax-compliance
  - hackathon
  - meta
  - huggingface
license: mit
---

<div align="center">

# 🚀 Vyapar-RL

**An OpenEnv Reinforcement Learning Environment for Indian GST Compliance**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blueviolet?style=for-the-badge)](https://github.com/meta-pytorch/OpenEnv)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Space-yellow?style=for-the-badge)](https://huggingface.co/spaces/clash1462/Vyapar-RL)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](Dockerfile)

*Built for the [Meta × Hugging Face OpenEnv Hackathon 2026](https://huggingface.co/spaces/open-env/leaderboard)*

**Team Code Nexus** — Yash Upadhyay · Krishna Garg

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Environment Motivation](#-environment-motivation)
- [Architecture](#-architecture)
- [Task Descriptions](#-task-descriptions)
- [Observation & Action Spaces](#-observation--action-spaces)
- [Grading & Reward Design](#-grading--reward-design)
- [Baseline Results](#-baseline-results)
- [Setup & Usage](#-setup--usage)
- [Project Structure](#-project-structure)
- [Team & Acknowledgements](#-team--acknowledgements)

---

## 🔭 Overview

**Vyapar-RL** is a three-task reinforcement learning environment that challenges AI agents to master Indian Goods & Services Tax (GST) compliance. It is built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework by Meta and Hugging Face, and derived from [Vyapar](https://github.com/yash6810/Vyapar) — a production WhatsApp AI assistant used by Indian SMEs for real-world GST tracking.

The environment evaluates an agent's ability to:
1. **Classify** business transactions into correct GST slabs
2. **Compute** quarterly tax liabilities with precise arithmetic
3. **Reconcile** filed returns (GSTR-1) against supplier declarations (GSTR-2A)

All task data, grading logic, and GST rules are grounded in **production use cases**, not synthetic toy examples.

---

## 💡 Environment Motivation

India has **14 million GST-registered businesses**. Every quarter, each one must:

| Compliance Step | Complexity | Real-World Impact |
|---|---|---|
| Classify every transaction into correct GST slab (0/5/12/18/28%) | Requires memorization of 1000+ HSN codes | Misclassification → penalties + interest |
| Compute CGST, SGST, IGST payable and claim ITC | Multi-step arithmetic across mixed slabs | Errors → cash flow disruption |
| Reconcile GSTR-1 vs GSTR-2A for invoice mismatches | Cross-referencing hundreds of invoices | Undetected mismatches → blocked ITC claims |

Small and medium enterprises (SMEs) frequently make errors in these tasks, leading to penalties, blocked credits, and compliance notices. Vyapar-RL trains AI agents to handle this at expert level with deterministic, auditable grading.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Vyapar-RL Server                      │
│                  (HuggingFace Space)                     │
│                                                         │
│  ┌───────────┐    ┌────────────┐    ┌────────────────┐  │
│  │  OpenEnv   │◄──│ Environment│◄──│  Deterministic  │  │
│  │ FastAPI    │   │ (3 Tasks)  │   │    Graders      │  │
│  │ Endpoint   │──►│            │──►│ (Partial Score) │  │
│  └───────────┘    └────────────┘    └────────────────┘  │
│       ▲                                    │            │
│       │              Feedback Loop         │            │
│       └────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────┘
        ▲                                          
        │  POST /reset, POST /step                 
        ▼                                          
┌───────────────┐                                  
│  LLM Agent    │  (inference.py)                  
│  Qwen 2.5 7B  │  via HuggingFace Router          
└───────────────┘                                  
```

---

## 📝 Task Descriptions

### Task 1 — Transaction GST Classifier `[Easy]`
| Property | Detail |
|---|---|
| **Input** | 10 business transactions with descriptions and amounts |
| **Output** | JSON mapping transaction IDs → GST slabs `{"1": 18, "2": 5}` |
| **Grading** | Exact slab match per transaction. `score = correct / total` |
| **Max Steps** | 3 |
| **Reward Signal** | Always unique (partial matches produce diverse floats) |

### Task 2 — Quarterly GST Liability Calculator `[Medium]`
| Property | Detail |
|---|---|
| **Input** | 12–15 transactions (sales + purchases) with taxable values and slabs |
| **Output** | JSON with 7 computed fields: `total_sales_value`, `total_purchase_value`, `cgst_payable`, `sgst_payable`, `igst_payable`, `total_itc`, `net_gst_liability` |
| **Grading** | Proportional error per field (≤1% → full credit, ≤5% → 0.8, ≤10% → 0.6) |
| **Max Steps** | 4 |
| **Reward Signal** | Math-based continuous float — always unique |

### Task 3 — GSTR-1 vs GSTR-2A Reconciliation `[Hard]`
| Property | Detail |
|---|---|
| **Input** | Two lists of invoice records (GSTR-1 filed, GSTR-2A received) |
| **Output** | JSON with `mismatches` array and `total_itc_at_risk` float |
| **Grading** | Detection (40%) + Type Classification (30%) + ITC Computation (30%) |
| **Max Steps** | 5 |
| **Reward Signal** | Multi-component weighted score |

---

## 🎮 Observation & Action Spaces

### Observation
```python
class GSTObservation(Observation):
    task_id: str           # "task1", "task2", or "task3"
    task_name: str
    difficulty: str        # "easy", "medium", or "hard"
    description: str
    task_data: dict        # Task-specific payload (transactions, GSTR data)
    instructions: str      # Exact format the agent must return
    feedback: str          # Populated after step 2+ with actionable error info
    score: float           # Reward from last step [0.0, 1.0]
    step_number: int
    max_steps: int
```

### Action
```python
class GSTAction(Action):
    task_id: str           # "task1", "task2", or "task3"
    action_type: str       # always "submit_answer"
    answer: str            # JSON string containing the agent's answer
```

---

## ⚖️ Grading & Reward Design

The reward system is designed to avoid common RL pitfalls:

| Design Principle | Implementation |
|---|---|
| **No Sparse Rewards** | Every step returns a granular float in `[0.0, 1.0]`, not binary pass/fail |
| **Actionable Feedback** | The `feedback` field tells the agent *which* specific IDs/fields were wrong, enabling genuine policy improvement |
| **Step Penalty** | `reward = max(0, task_score - 0.02 × (step - 1))` incentivizes getting it right early without collapsing task quality |
| **No Reward Hacking** | Grading is deterministic against golden answers — there are no exploitable proxy metrics |
| **Reward Diversity** | Partial credit scoring guarantees that rewards are always unique floats, never degenerate |

---

## 📊 Baseline Results

Evaluated with `Qwen/Qwen2.5-7B-Instruct` via HuggingFace Inference Router:

| Task | Difficulty | Step 1 | Step 2 | Step 3 | Avg Reward | Trend |
|---|---|---|---|---|---|---|
| GST Classifier | Easy | 0.70 | 0.88 | 0.86 | **0.81** | 📈 +26% with feedback |
| Liability Calculator | Medium | 0.41 | 0.68 | 0.66 | **0.60** | 📈 +66% with feedback |
| GSTR Reconciliation | Hard | 1.00 | — | — | **1.00** | ✅ Perfect first attempt |
| **Overall** | — | — | — | — | **~0.80** | — |

> **Key Insight:** The feedback loop drives authentic RL adaptation. On Task 1, the agent improved from 70% → 88% by reading the environment's feedback identifying which specific transaction IDs were misclassified. This demonstrates the environment's core value proposition — it doesn't just score, it *teaches*.

---

## 🚀 Setup & Usage

### Prerequisites
```bash
pip install openenv-core fastapi uvicorn openai python-dotenv
```

### Local Development
```bash
# Clone and install
git clone https://github.com/yash6810/Vyapar-RL
cd Vyapar-RL
pip install -e .

# Run server
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run Baseline Inference
```bash
export HF_TOKEN=your_huggingface_token
export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
export ENV_URL=https://clash1462-vyapar-rl.hf.space

python inference.py
```

### Docker
```bash
docker build -t vyapar-rl .
docker run -p 7860:7860 vyapar-rl
```

### Live API (Deployed)
```bash
# Health check
curl https://clash1462-vyapar-rl.hf.space/health

# Reset environment
curl -X POST https://clash1462-vyapar-rl.hf.space/reset \
  -H "Content-Type: application/json" -d '{}'
```

---

## 📁 Project Structure

```
Vyapar-RL/
├── inference.py           # Baseline LLM agent (mandatory entry point)
├── Dockerfile             # HuggingFace Space deployment config
├── models.py              # Pydantic models (GSTAction, GSTObservation, GSTState)
├── gst_rules.py           # Indian GST slab classification rules
├── client.py              # Typed OpenEnv client wrapper
├── openenv.yaml           # OpenEnv environment metadata
├── pyproject.toml         # Python project configuration
├── server/
│   ├── app.py             # FastAPI application factory
│   ├── environment.py     # Core OpenEnv Environment (reset/step logic)
│   └── graders.py         # Deterministic graders for all 3 tasks
└── data/
    ├── task1_easy.json     # GST classification dataset
    ├── task2_medium.json   # Quarterly liability dataset
    └── task3_hard.json     # GSTR reconciliation dataset
```

---

## 👥 Team & Acknowledgements

**Team Code Nexus**
- [Yash Upadhyay](https://github.com/yash6810)
- [Krishna Garg](https://github.com/clash1462)

**Built with:**
- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) by Meta AI
- [Hugging Face Spaces](https://huggingface.co/spaces) for deployment
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) for baseline evaluation

**Derived from:**
- [Vyapar](https://github.com/yash6810/Vyapar) — Production WhatsApp AI assistant for Indian SMEs

---

<div align="center">

*Submitted to the Meta × Hugging Face OpenEnv Hackathon 2026*

</div>
