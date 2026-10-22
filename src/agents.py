"""Phase 4: the AutoGen agents and the experiment loop.

PLANNER (Claude Sonnet): reads the data profile + leaderboard (from MLflow),
    decides the next experiment, and explains why. Has NO tools: it only thinks.
EXPERIMENTER (Claude Haiku): receives the plan and calls train_and_evaluate.
    It is the only agent that can train models.

Loop:  planner -> experimenter -> result logged to MLflow -> repeat
       until the budget is used up.
Check: after the loop, the planner picks ONE run, and the code (not the LLM)
       calls final_test() once. The agents never get access to the test set.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Callable

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import ModelInfo
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from dotenv import load_dotenv

from report import write_report
from tools import MODEL_SPECS, ExperimentSession, as_json

load_dotenv()

PLANNER_MODEL = os.getenv("PLANNER_MODEL", "claude-sonnet-5-5")
EXPERIMENTER_MODEL = os.getenv("EXPERIMENTER_MODEL", "claude-haiku-5-5")

# USD per million tokens (input, output). Used only to show an estimated cost.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50),
          "claude-opus-5-5": (4.0, 20.0)}

PLANNER_SYSTEM = """You are the PLANNER in an AutoML team. Goal: find the model with the
highest VALIDATION AUC-ROC on a binary classification dataset, within a fixed budget.

Each turn you see the data profile, the leaderboard (read from MLflow), the last result,
and the budget left. You choose exactly ONE next experiment.

Strategy:
- Early: try each model family at least once (logistic_regression, random_forest,
  lightgbm, mlp) so you know where the signal is.
- Middle: focus on the best family. Change 1-3 settings at a time so you learn what helps.
- Watch overfit_gap (train AUC minus val AUC). A big gap means the model memorizes:
  regularize (fewer leaves, more min_child_samples / min_samples_leaf, lower learning rate).
- You may try a smaller feature subset if you think some features are noise.
- Never repeat an experiment that is already on the leaderboard.
- Last 1-2 experiments: refine the best config.

Reply in this exact format and nothing else:
```json
{"model_type": "...", "params": {...}, "features": null, "reasoning": "1-2 sentences"}
```
Use "features": null for all features, or a list of column names for a subset.
Only use the allowed settings listed for each model."""

EXPERIMENTER_SYSTEM = """You are the EXPERIMENTER in an AutoML team. The planner sends a
JSON plan. Call the train_and_evaluate tool exactly once, copying model_type, params and
features exactly as given, and put the planner's reasoning in notes. Do not change the plan.
Do not add commentary."""

FINAL_SYSTEM = """You are the PLANNER. The experiment budget is used up. Choose the ONE
experiment to send to the final test. Usually the highest validation AUC, but you may
prefer a nearly-as-good model with a much smaller overfit gap (more likely to generalize).
Reply in this exact format and nothing else:
```json
{"experiment_number": 0, "reason": "1-2 sentences"}
```"""


_THINKING_CACHE: dict[str, dict | None] = {}


def _thinking_off(model: str) -> dict | None:
    """Ask the Models API how this model turns up-front thinking off.
    Newer models use "between_tools"; older ones use "disabled"."""
    if model in _THINKING_CACHE:
        return _THINKING_CACHE[model]
    setting: dict | None = {"type": "between_tools"}
    try:
        import anthropic
        info = anthropic.Anthropic(timeout=10, max_retries=1).get(f"/v1/models/{model}", cast_to=object)
        types = (info.get("capabilities") or {}).get("thinking", {}).get("types", {})
        if types.get("between_tools", {}).get("supported"):
            setting = {"type": "between_tools"}
        elif types.get("disabled", {}).get("supported"):
            setting = {"type": "disabled"}
        elif types:
            setting = None   # thinking can't be turned off: send nothing
    except Exception:
        pass             # API lookup failed: keep the default
    _THINKING_CACHE[model] = setting
    return setting


def _client(model: str) -> AnthropicChatCompletionClient:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not found. Put it in the .env file.")
    thinking = _thinking_off(model)
    return AnthropicChatCompletionClient(
        model=model,
        max_tokens=1024,
        **({"thinking": thinking} if thinking else {}),
        model_info=ModelInfo(vision=False, function_calling=True, json_output=False,
                             family="unknown", structured_output=False),
    )


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _allowed_settings() -> str:
    return "\n".join(f"- {k}: {sorted(v['allowed'])}" for k, v in MODEL_SPECS.items())


def _compact_leaderboard(rows: list[dict]) -> list[dict]:
    return [{"exp": r["experiment_number"], "model": r["model_type"],
             "val_auc": r["val_auc"], "train_auc": r["train_auc"],
             "overfit_gap": round((r["train_auc"] or 0) - (r["val_auc"] or 0), 4),
             "n_features": r["n_features"], "params": r["params"]} for r in rows]


class Usage:
    """Adds up tokens per model so we can show the API cost."""
    def __init__(self):
        self.tokens: dict[str, list[int]] = {}

    def add(self, messages, model_by_agent: dict[str, str]):
        for m in messages:
            u = getattr(m, "models_usage", None)
            if u:
                t = self.tokens.setdefault(model_by_agent.get(m.source, "?"), [0, 0])
                t[0] += u.prompt_tokens
                t[1] += u.completion_tokens

    def cost(self) -> float:
        return round(sum(i / 1e6 * PRICES.get(m, (0, 0))[0] + o / 1e6 * PRICES.get(m, (0, 0))[1]
                         for m, (i, o) in self.tokens.items()), 4)


async def run_automl(session: ExperimentSession,
                     on_event: Callable[[str, dict], None] | None = None) -> dict:
    """Run the full agent loop. on_event(kind, data) lets the UI show live progress."""
    emit = on_event or (lambda kind, data: None)
    usage = Usage()
    models = {"planner": PLANNER_MODEL, "experimenter": EXPERIMENTER_MODEL}
    last_result: dict[str, Any] = {}

    # ---- the experimenter's only tool ----
    async def train_and_evaluate(model_type: str, params: dict[str, Any] | None = None,
                                 features: list[str] | None = None, notes: str = "") -> str:
        """Train one model on the training set and score it on the validation set.
        model_type: logistic_regression | random_forest | lightgbm | mlp.
        params: model settings to override (only allowed ones). features: None = all."""
        r = await asyncio.to_thread(session.train_and_evaluate, model_type, params,
                                    features, notes)
        last_result.clear()
        last_result.update(r)
        return as_json(r)

    planner = AssistantAgent("planner", model_client=_client(PLANNER_MODEL),
                             system_message=PLANNER_SYSTEM)
    experimenter = AssistantAgent("experimenter", model_client=_client(EXPERIMENTER_MODEL),
                                  tools=[train_and_evaluate], system_message=EXPERIMENTER_SYSTEM,
                                  reflect_on_tool_use=False)
    team = RoundRobinGroupChat([planner, experimenter], max_turns=2)

    profile = session.profile_data()
    emit("start", {"profile": profile, "budget": session.budget,
                   "planner_model": PLANNER_MODEL, "experimenter_model": EXPERIMENTER_MODEL})

    steps, failures = [], 0
    while session.used < session.budget and failures < 4:
        board = _compact_leaderboard(session.get_leaderboard(top_n=10))
        task = (f"DATA PROFILE:\n{as_json(profile)}\n\n"
                f"ALLOWED SETTINGS:\n{_allowed_settings()}\n\n"
                f"LEADERBOARD (from MLflow, best first):\n{as_json(board) if board else 'empty'}\n\n"
                f"LAST RESULT:\n{as_json(last_result) if last_result else 'none yet'}\n\n"
                f"BUDGET LEFT: {session.budget - session.used} of {session.budget}\n"
                f"Choose the next experiment.")
        used_before = session.used
        last_result.clear()

        await team.reset()   # fresh conversation each round; memory lives in MLflow
        result = await team.run(task=task)
        usage.add(result.messages, models)

        plan_msg = next((m for m in result.messages if m.source == "planner"), None)
        plan = _parse_json(getattr(plan_msg, "content", "")) or {}

        if session.used == used_before:      # nothing was trained this round
            failures += 1
            err = last_result.get("error", "experimenter did not call the tool")
            last_result["error"] = err
            emit("error", {"plan": plan, "error": err})
            continue

        failures = 0
        step = {"plan": plan, "result": dict(last_result)}
        steps.append(step)
        emit("experiment", {**step, "leaderboard": session.get_leaderboard(top_n=20),
                            "cost_usd": usage.cost()})

    # ---- the check: pick one run, test ONCE ----
    board = session.get_leaderboard(top_n=20)
    if not board:
        raise RuntimeError("No successful experiments; cannot run the final test.")
    chooser = AssistantAgent("planner", model_client=_client(PLANNER_MODEL),
                             system_message=FINAL_SYSTEM)
    res = await chooser.run(task="LEADERBOARD:\n" + as_json(_compact_leaderboard(board)))
    usage.add(res.messages, models)
    choice = _parse_json(res.messages[-1].content) or {}

    by_number = {r["experiment_number"]: r["run_id"] for r in board}
    run_id = by_number.get(choice.get("experiment_number"), board[0]["run_id"])
    if run_id == board[0]["run_id"] and choice.get("experiment_number") not in by_number:
        choice = {"experiment_number": board[0]["experiment_number"],
                  "reason": "Fallback: highest validation AUC."}
    emit("final_choice", choice)

    final = await asyncio.to_thread(session.final_test, run_id)
    summary = {"final": final, "choice": choice, "steps": steps, "leaderboard": board,
               "tokens": usage.tokens, "cost_usd": usage.cost(),
               "experiment_name": session.name, "profile": profile,
               "models": {"planner": PLANNER_MODEL, "experimenter": EXPERIMENTER_MODEL}}
    summary["report_path"] = str(await asyncio.to_thread(write_report, summary))
    emit("done", summary)
    return summary
