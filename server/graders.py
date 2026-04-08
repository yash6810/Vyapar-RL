"""
server/graders.py — Deterministic graders for all 3 tasks.
Each grader returns a float in [0.0, 1.0].
"""

import json
from typing import Any, Dict


def safe_parse_json(answer_str: str) -> tuple[dict | None, bool]:
    """Parse agent's JSON string. Returns (parsed_dict, is_valid)."""
    try:
        parsed = json.loads(answer_str)
        if not isinstance(parsed, dict):
            return None, False
        return parsed, True
    except (json.JSONDecodeError, TypeError):
        return None, False


def grade_task1(
    agent_answer_str: str, golden_answer: Dict[str, int]
) -> tuple[float, str]:
    """
    Task 1 grader: GST slab classification.
    Returns (score, feedback_string).
    """
    parsed, is_valid = safe_parse_json(agent_answer_str)
    if not is_valid or not parsed:
        return 0.0, 'Invalid JSON. Return format: {"1": 18, "2": 5, ...}'

    wrong_ids = []
    correct = 0
    for k, expected_slab in golden_answer.items():
        try:
            agent_slab = parsed.get(str(k))
            if agent_slab is not None and int(agent_slab) == expected_slab:
                correct += 1
            else:
                wrong_ids.append(str(k))
        except (ValueError, TypeError):
            wrong_ids.append(str(k))

    score = round(correct / len(golden_answer), 4) if golden_answer else 0.0
    if wrong_ids:
        feedback = f"Wrong classification for transaction IDs: {', '.join(wrong_ids)}. Check GST slab boundaries."
    else:
        feedback = "Perfect! All transactions correctly classified."
    return score, feedback


def grade_task2(
    agent_answer_str: str, golden_answer: Dict[str, float]
) -> tuple[float, str]:
    """
    Task 2 grader: Quarterly GST liability computation.
    Returns (score, feedback_string).
    """
    parsed, is_valid = safe_parse_json(agent_answer_str)
    if not is_valid or not parsed:
        return (
            0.0,
            'Invalid JSON. Return format: {"total_sales_value": ..., "cgst_payable": ..., etc.}',
        )

    fields = [
        "total_sales_value",
        "total_purchase_value",
        "cgst_payable",
        "sgst_payable",
        "igst_payable",
        "total_itc",
        "net_gst_liability",
    ]
    scores = []
    wrong_fields = []

    for field in fields:
        if field not in golden_answer:
            continue
        expected = float(golden_answer[field])

        try:
            val = parsed.get(field)
            got = float(val) if val is not None else -999.0
        except (ValueError, TypeError):
            got = -999.0

        if got == -999.0:
            scores.append(0.0)
            wrong_fields.append(f"{field} (missing or invalid)")
            continue

        if expected == 0:
            field_score = 1.0 if got == 0 else 0.0
        else:
            error_pct = abs(got - expected) / abs(expected)
            if error_pct <= 0.01:
                field_score = 1.0
            elif error_pct <= 0.05:
                field_score = 0.8
            elif error_pct <= 0.10:
                field_score = 0.6
            elif error_pct <= 0.20:
                field_score = 0.3
            else:
                field_score = 0.0
                wrong_fields.append(f"{field} (got {got}, expected {expected})")

        scores.append(field_score)

    score = round(sum(scores) / len(scores), 4) if scores else 0.0
    if wrong_fields:
        feedback = f"Errors in: {'; '.join(wrong_fields[:3])}. Recheck GST computation formula."
    else:
        feedback = "All fields computed correctly!"
    return score, feedback


def grade_task3(
    agent_answer_str: str, golden_answer: Dict[str, Any]
) -> tuple[float, str]:
    """
    Task 3 grader: GSTR-1 vs GSTR-2A reconciliation.
    Returns (score, feedback_string).
    """
    parsed, is_valid = safe_parse_json(agent_answer_str)
    if not is_valid or not parsed or "mismatches" not in parsed:
        return (
            0.0,
            "Invalid JSON. Must include 'mismatches' list and 'total_itc_at_risk'.",
        )

    try:
        mismatches = parsed.get("mismatches", [])
        if not isinstance(mismatches, list):
            mismatches = []
        agent_mismatches = {
            m["invoice_no"]: m
            for m in mismatches
            if isinstance(m, dict) and "invoice_no" in m
        }
    except Exception:
        agent_mismatches = {}

    try:
        golden_mismatches = {
            m["invoice_no"]: m for m in golden_answer.get("mismatches", [])
        }
    except Exception:
        golden_mismatches = {}

    if not golden_mismatches:
        return 1.0, "No mismatches expected and none found!"

    # Score 1: Detection (40%) — did agent find the right invoice numbers?
    found = set(agent_mismatches.keys()) & set(golden_mismatches.keys())
    detection_score = len(found) / len(golden_mismatches)

    # Score 2: Classification (30%) — did agent label mismatch_type correctly?
    type_correct = sum(
        1
        for inv in found
        if agent_mismatches[inv].get("mismatch_type")
        == golden_mismatches[inv]["mismatch_type"]
    )
    type_score = type_correct / len(golden_mismatches)

    # Score 3: ITC computation (30%) — is total_itc_at_risk correct?
    expected_itc = float(golden_answer.get("total_itc_at_risk", 0))
    try:
        val = parsed.get("total_itc_at_risk", 0)
        agent_itc = float(val) if val is not None else 0.0
    except (ValueError, TypeError):
        agent_itc = 0.0

    if expected_itc == 0:
        itc_score = 1.0 if agent_itc == 0 else 0.0
    else:
        itc_error = abs(agent_itc - expected_itc) / expected_itc
        itc_score = max(0.0, 1.0 - itc_error)

    final = round(0.40 * detection_score + 0.30 * type_score + 0.30 * itc_score, 4)

    missed = set(golden_mismatches.keys()) - found
    feedback_parts = []
    if missed:
        feedback_parts.append(f"Missed invoices: {', '.join(list(missed)[:3])}")
    if itc_score < 0.9:
        feedback_parts.append(f"ITC at risk: got {agent_itc}, expected {expected_itc}")
    feedback = (
        "; ".join(feedback_parts) if feedback_parts else "Excellent reconciliation!"
    )
    return final, feedback


def compute_reward(task_score: float, step_number: int, is_valid: bool) -> float:
    """Wrap task score with step penalty for reward diversity."""
    if not is_valid:
        return 0.0
    base = max(0.05, task_score)
    step_penalty = 0.02 * (step_number - 1)
    return round(max(0.0, min(1.0, base - step_penalty)), 4)
