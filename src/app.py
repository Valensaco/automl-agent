"""Streamlit app for the AutoML Agent.

    streamlit run src/app.py        (run from the project folder)

Opens on a saved run that replays automatically, so visitors see the agents
work without an API key. New runs: locally your .env key is used; on a public
deploy set APP_PASSWORD in Streamlit secrets (visitors then use their own key).
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agents import EXPERIMENTER_MODEL, PLANNER_MODEL, run_automl
from report import PRETTY, REPORTS, build_markdown
from tools import ExperimentSession

load_dotenv()
st.set_page_config(page_title="AutoML Agent", page_icon=":material/smart_toy:",
                   layout="wide")

# Fixed color per model type (validated categorical palette, slots 1-4).
SHORT = {"logistic_regression": "Logistic regression", "random_forest": "Random forest",
         "lightgbm": "LightGBM", "mlp": "Neural net (MLP)"}
MODEL_COLORS = {"Logistic regression": "#2a78d6", "Random forest": "#eb6834",
                "LightGBM": "#1baf7a", "Neural net (MLP)": "#eda100"}
INK, MUTED, GRID = "#11171C", "#5F6B76", "#E8ECF0"
MAX_BUDGET = 15
REPLAY_DELAY = 0.5

st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1400px;}
h1 {font-weight: 650; letter-spacing: -0.02em; margin-bottom: 0 !important;}
.subtitle {color: #5F6B76; font-size: 1.02rem; margin: 0.2rem 0 0.6rem 0;}
.pill {display: inline-block; padding: 2px 10px; margin: 0 6px 6px 0; border-radius: 999px;
       border: 1px solid #DDE3E8; background: #FFFFFF; font-size: 0.8rem; color: #11171C;
       text-decoration: none;}
.section {font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em;
          text-transform: uppercase; color: #5F6B76; margin: 0.4rem 0 0.4rem 0;}
[data-testid="stChatMessage"] {background: #FFFFFF; border: 1px solid #E3E8EC;
                               border-radius: 8px; padding: 0.6rem 0.8rem;}
[data-testid="stMetricValue"] {font-size: 1.55rem; font-weight: 600;}
/* during a replay or live run, hide leftovers from the previous screen */
[class*="st-key-view-"] [data-stale="true"] {display: none;}
[class*="st-key-view-"] [data-testid="stChatMessage"]:has([data-stale="true"]):not(:has([data-stale="false"])) {display: none;}
</style>
""", unsafe_allow_html=True)


def secret(name: str) -> str | None:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name)


OWNER_KEY = secret("ANTHROPIC_API_KEY")
APP_PASSWORD = secret("APP_PASSWORD")
GITHUB_URL = secret("GITHUB_URL")
MLFLOW_URL = secret("MLFLOW_URL")


# ======================================================================= data
def load_saved_runs() -> dict[str, dict]:
    runs = {}
    for p in REPORTS.glob("*.json"):
        try:
            runs[p.stem] = json.loads(p.read_text())
        except Exception:
            pass
    # featured first: most experiments, then newest
    order = sorted(runs, key=lambda k: (len(runs[k].get("steps", [])), k), reverse=True)
    return {k: runs[k] for k in order}


def steps_frame(steps: list[dict]) -> pd.DataFrame:
    rows, best = [], 0.0
    for s in steps:
        r = s["result"]
        auc = r["val_metrics"]["auc"]
        best = max(best, auc)
        rows.append({"#": r["experiment_number"],
                     "Model": SHORT.get(r["model_type"], r["model_type"]),
                     "Val AUC": auc, "Best so far": best,
                     "Overfit gap": r["overfit_gap"],
                     "Reasoning": s["plan"].get("reasoning", "")})
    return pd.DataFrame(rows)


# ======================================================================= charts
def _style(chart):
    return (chart.configure_view(strokeWidth=0)
            .configure_axis(gridColor=GRID, domainColor=GRID, tickColor=GRID,
                            labelColor=MUTED, titleColor=MUTED, labelFontSize=11,
                            titleFontSize=11, titleFontWeight="normal")
            .configure_legend(labelColor=INK, labelFontSize=11, symbolSize=90)
            .properties(background="transparent"))


def progress_chart(df: pd.DataFrame, budget: int | None = None):
    lo = max(0.5, df["Val AUC"].min() - 0.01)
    hi = min(1.0, df["Val AUC"].max() + 0.005)
    x = alt.X("#:Q", title="Experiment", axis=alt.Axis(tickMinStep=1, format="d"),
              scale=alt.Scale(domain=[0.5, (budget or int(df["#"].max())) + 0.5]))
    yscale = alt.Scale(domain=[lo, hi])
    best = alt.Chart(df).mark_line(strokeWidth=2, color="#8A949C", interpolate="step-after",
                                   strokeDash=[4, 3]).encode(
        x=x, y=alt.Y("Best so far:Q", scale=yscale, title="Validation AUC"),
        tooltip=[alt.Tooltip("Best so far:Q", format=".4f")])
    dots = alt.Chart(df).mark_circle(size=120, opacity=1, stroke="white",
                                     strokeWidth=2).encode(
        x=x, y=alt.Y("Val AUC:Q", scale=yscale),
        color=alt.Color("Model:N", scale=alt.Scale(domain=list(MODEL_COLORS),
                                                   range=list(MODEL_COLORS.values())),
                        legend=alt.Legend(orient="top", title=None, direction="horizontal",
                                          labelLimit=0, columns=2)),
        tooltip=[alt.Tooltip("#:Q", title="Experiment"), "Model",
                 alt.Tooltip("Val AUC:Q", format=".4f"),
                 alt.Tooltip("Overfit gap:Q", format="+.3f")])
    return _style((best + dots).properties(height=260))


def importance_chart(rows: list[dict]):
    df = pd.DataFrame(rows)
    return _style(alt.Chart(df).mark_bar(color="#0194E2", cornerRadiusEnd=4, height=16).encode(
        x=alt.X("auc_drop:Q", title="AUC drop when the feature is shuffled"),
        y=alt.Y("feature:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("feature:N", title="Feature"),
                 alt.Tooltip("auc_drop:Q", title="AUC drop", format=".4f")],
    ).properties(height=30 * len(df)))


def leaderboard_table(df: pd.DataFrame):
    board = (df.drop(columns=["Reasoning", "Best so far"])
             .sort_values("Val AUC", ascending=False).reset_index(drop=True))
    board.insert(0, "Rank", range(1, len(board) + 1))
    return (board.style
            .apply(lambda r: ["background-color: #E6F4FC; font-weight: 600"
                              if r["Rank"] == 1 else "" for _ in r], axis=1)
            .format({"Val AUC": "{:.4f}", "Overfit gap": "{:+.3f}"}))


# ======================================================================= run view
class RunView:
    """Conversation on the left; stats, chart and leaderboard on the right.
    Used for live runs, replays, and showing a finished run."""

    def __init__(self, budget: int):
        self.budget = budget
        self.steps: list[dict] = []
        self.stats = st.empty()
        left, right = st.columns([5, 4], gap="large")
        with left:
            st.markdown('<div class="section">Agent conversation</div>',
                        unsafe_allow_html=True)
            self.chat = st.container(height=620, border=False)
        with right:
            st.markdown('<div class="section">Leaderboard (read from MLflow)</div>',
                        unsafe_allow_html=True)
            self.chart = st.empty()
            self.table = st.empty()

    def intro(self, profile: dict | None, planner: str, experimenter: str):
        with self.chat:
            with st.chat_message("user", avatar=":material/dataset:"):
                if profile:
                    st.markdown(f"**Task:** predict `{profile['target']}` · "
                                f"{profile['rows_train']:,} training rows · "
                                f"{profile['n_features']} features · "
                                f"budget **{self.budget}** experiments")
                else:
                    st.markdown(f"**Task:** budget **{self.budget}** experiments")
                st.caption(f"Planner: {planner} · Experimenter: {experimenter}")

    def add_step(self, step: dict, cost: float | None = None):
        self.steps.append(step)
        r, p = step["result"], step["plan"]
        with self.chat:
            with st.chat_message("planner", avatar=":material/psychology:"):
                st.markdown(f"**Planner** · experiment {r['experiment_number']}: "
                            f"**{PRETTY.get(r['model_type'], r['model_type'])}**")
                st.write(p.get("reasoning", ""))
            with st.chat_message("experimenter", avatar=":material/science:"):
                st.markdown(f"**Experimenter** · val AUC **{r['val_metrics']['auc']:.4f}** · "
                            f"overfit gap {r['overfit_gap']:+.3f} · "
                            f"trained in {r['fit_seconds']}s")
        self.refresh(cost)

    def refresh(self, cost: float | None = None):
        df = steps_frame(self.steps)
        best = df.loc[df["Val AUC"].idxmax()]
        with self.stats.container():
            c = st.columns(4)
            c[0].metric("Experiments", f"{len(df)} / {self.budget}", border=True)
            c[1].metric("Best val AUC", f"{best['Val AUC']:.4f}", border=True)
            c[2].metric("Best model", best["Model"], border=True)
            c[3].metric("API cost", f"${cost:.3f}" if cost is not None else "–", border=True)
        self.chart.altair_chart(progress_chart(df, self.budget), width="stretch")
        self.table.dataframe(leaderboard_table(df), hide_index=True, width="stretch",
                             height=min(36 * (len(df) + 1) + 4, 330))

    def note(self, kind: str, text: str):
        with self.chat:
            with st.chat_message("system", avatar=":material/flag:"):
                (st.success if kind == "ok" else st.warning)(text)

    def test_opened(self, s: dict):
        f = s["final"]
        with self.chat:
            with st.chat_message("system", avatar=":material/lock_open:"):
                st.markdown(f"**Test set opened once:** test AUC "
                            f"**{f['test_metrics']['auc']:.4f}** (validation "
                            f"{f['val_auc']:.4f}). See the **Results** tab.")


def show_run(s: dict, animate: bool):
    view = RunView(s["final"]["experiments_used"])
    models = s.get("models", {})
    view.intro(s.get("profile"), models.get("planner", PLANNER_MODEL),
               models.get("experimenter", EXPERIMENTER_MODEL))
    for step in s["steps"]:
        view.add_step(step, s["cost_usd"])
        if animate:
            time.sleep(REPLAY_DELAY)
    view.note("ok", f"**Final pick: experiment {s['choice'].get('experiment_number')}.** "
                    f"{s['choice'].get('reason', '')}")
    view.test_opened(s)


def run_live(session: ExperimentSession, api_key: str | None):
    view = RunView(session.budget)

    def on_event(kind: str, d: dict):
        if kind == "start":
            view.intro(d["profile"], d["planner_model"], d["experimenter_model"])
        elif kind == "experiment":
            view.add_step({"plan": d["plan"], "result": d["result"]}, d["cost_usd"])
        elif kind == "error":
            view.note("warn", f"Skipped a bad plan: {d['error']}")
        elif kind == "final_choice":
            view.note("ok", f"**Final pick: experiment {d.get('experiment_number')}.** "
                            f"{d.get('reason', '')}")

    with st.spinner("Agents are working..."):
        try:
            summary = asyncio.run(run_automl(session, on_event=on_event, api_key=api_key))
        except Exception as e:
            msg = str(e)
            if "authentication" in msg.lower() or "401" in msg:
                msg = "The API key was rejected. Check that it is correct and has credit."
            st.error(msg)
            return None
    view.test_opened(summary)
    return summary


# ======================================================================= results
def show_results(s: dict):
    f = s["final"]
    t, d, cm = f["test_metrics"], f["test_metrics_default_0_5"], f["confusion_matrix"]
    defaulters = cm["true_positive"] + cm["false_negative"]
    flagged = cm["true_positive"] + cm["false_positive"]

    st.markdown(f"### {PRETTY.get(f['model_type'], f['model_type'])} · "
                f"experiment {s['choice'].get('experiment_number')}")
    st.caption(f"Chosen by the planner: {s['choice'].get('reason', '')}")

    c = st.columns(5)
    c[0].metric("Test AUC", f"{t['auc']:.3f}", f"{-f['val_to_test_auc_drop']:+.3f} vs validation",
                delta_color="off", border=True)
    c[1].metric("Defaulters caught", f"{t['recall']:.0%}",
                f"{(t['recall'] - d['recall']) * 100:+.0f} pts vs 0.50", border=True)
    c[2].metric("Precision", f"{t['precision']:.0%}", border=True)
    c[3].metric("Decision cutoff", f"{f['threshold']:.2f}", "tuned on validation",
                delta_color="off", border=True)
    c[4].metric("API cost", f"${s['cost_usd']:.3f}", f"{f['experiments_used']} experiments",
                delta_color="off", border=True)

    left, right = st.columns([1, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown(f'<div class="section">On the test set '
                        f'({f["test_rows"]:,} unseen rows)</div>', unsafe_allow_html=True)
            st.markdown(
                f"- Caught **{cm['true_positive']:,} of {defaulters:,}** customers who defaulted\n"
                f"- Flagged **{flagged:,}** as risky; **{cm['true_positive']:,}** of them "
                f"really defaulted\n"
                f"- The default 0.50 cutoff would have caught only **{d['recall']:.0%}**")
            st.dataframe(pd.DataFrame(
                {"Predicted: no default": [cm["true_negative"], cm["false_negative"]],
                 "Predicted: default": [cm["false_positive"], cm["true_positive"]]},
                index=["Actual: no default", "Actual: default"]), width="stretch")
    with right:
        with st.container(border=True):
            st.markdown('<div class="section">Most important features</div>',
                        unsafe_allow_html=True)
            st.altair_chart(importance_chart(f["feature_importance"]), width="stretch")

    with st.container(border=True):
        st.markdown('<div class="section">Default vs tuned cutoff</div>',
                    unsafe_allow_html=True)
        names = {"auc": "AUC-ROC", "pr_auc": "PR-AUC", "recall": "Recall",
                 "precision": "Precision", "f1": "F1"}
        st.dataframe(pd.DataFrame({"Cutoff 0.50": {names[k]: d[k] for k in names},
                                   f"Cutoff {f['threshold']:.2f}": {names[k]: t[k] for k in names}}
                                  ).style.format("{:.4f}"), width="stretch")

    with st.expander("Final model settings"):
        st.json(f["params"])
    st.download_button("Download full report", build_markdown(s),
                       file_name=f"{s['experiment_name']}.md", icon=":material/download:")


# ======================================================================= how it works
ARCH = """
digraph {
  rankdir=TB; bgcolor="transparent"; nodesep=0.5; ranksep=0.4;
  node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="#C9D2D9",
        fontname="Helvetica", fontsize=14, fontcolor="#11171C", margin="0.22,0.12"];
  edge [color="#8A949C", fontname="Helvetica", fontsize=11, fontcolor="#5F6B76"];
  data [label="Dataset\\nsplit 60/20/20"];
  planner [label="Planner agent\\n(Claude Sonnet)", fillcolor="#E6F4FC", color="#0194E2"];
  exp [label="Experimenter agent\\n(Claude Haiku)", fillcolor="#E6F4FC", color="#0194E2"];
  tool [label="train_and_evaluate\\nscikit-learn / LightGBM"];
  mlflow [label="MLflow\\nexperiment log"];
  final [label="final_test()\\ncalled once by code"];
  test [label="Test set\\n(locked)", style="rounded,dashed", fillcolor="#FFFFFF"];
  data -> planner [label="profile"];
  planner -> exp [label="plan"];
  exp -> tool [label="tool call"];
  tool -> mlflow [label="log run"];
  mlflow -> planner [label="  leaderboard", constraint=false];
  {rank=same; mlflow; test;}
  mlflow -> final [label="budget used up"];
  test -> final;
}
"""


def show_how():
    diagram, cards = st.columns([1, 1], gap="large")
    with diagram, st.container(border=True):
        st.markdown('<div class="section">Architecture</div>', unsafe_allow_html=True)
        st.graphviz_chart(ARCH, width="content")
    a, b, c = (cards, cards, cards)
    with a, st.container(border=True):
        st.markdown("**Tools**")
        st.markdown("The experimenter can only call `train_and_evaluate`, which trains one of "
                    "4 models (logistic regression, random forest, LightGBM, neural network), "
                    "scores it on validation, and logs it to MLflow.")
    with b, st.container(border=True):
        st.markdown("**Loop**")
        st.markdown("Each round, the planner reads the leaderboard from MLflow and picks the "
                    "next experiment: explore all model types, then tune the best one and fix "
                    "overfitting. It stops when the budget runs out.")
    with c, st.container(border=True):
        st.markdown("**Check**")
        st.markdown("The test set is never given to the agents. After the budget is used, "
                    "the code calls `final_test()` once. The decision cutoff and feature "
                    "importance come from validation data only.")
    st.caption("Stack: AutoGen · Claude · scikit-learn · LightGBM · MLflow · Streamlit · Docker")


# ======================================================================= page
saved = load_saved_runs()
if "summary" not in st.session_state and saved:
    st.session_state.summary = next(iter(saved.values()))

# ---- sidebar: settings ----
with st.sidebar:
    st.markdown("### New run")
    source = st.radio("Dataset", ["Credit card default (UCI)", "Upload a CSV"])
    df_up, target = None, None
    if source == "Upload a CSV":
        up = st.file_uploader("CSV file", type="csv")
        if up:
            df_up = pd.read_csv(up)
            binary = [c for c in df_up.columns if df_up[c].nunique(dropna=True) == 2]
            if binary:
                target = st.selectbox("Target column", binary, index=len(binary) - 1)
            else:
                st.error("The target must be a column with exactly 2 values.")
    else:
        st.caption("30,000 credit card customers. Predict who defaults next month.")
    budget = st.slider("Budget (experiments)", 3, MAX_BUDGET, 10)

    api_key, allowed = None, True
    if APP_PASSWORD:
        access = st.segmented_control("API access", ["My API key", "Owner password"],
                                      default="My API key")
        if access == "Owner password":
            pw = st.text_input("Password", type="password")
            allowed = bool(pw) and pw == APP_PASSWORD
            api_key = OWNER_KEY if allowed else None
        else:
            api_key = st.text_input("Anthropic API key", type="password",
                                    help="Used only for this run. Never stored.")
            allowed = bool(api_key)
    elif not OWNER_KEY:
        api_key = st.text_input("Anthropic API key", type="password")
        allowed = bool(api_key)
    st.caption(f"About ${0.007 * budget:.2f} in API calls")
    go = st.button("Run agent", type="primary", width="stretch",
                   disabled=not allowed or (source == "Upload a CSV" and target is None))

    replay = False
    if saved:
        st.divider()
        st.markdown("### Saved runs")
        current = st.session_state.summary["experiment_name"]
        options = dict(saved)
        if current not in options:
            options = {current: st.session_state.summary, **options}
        names = list(options)
        pick = st.selectbox("Run", names, index=names.index(current),
                            format_func=lambda n: f"{n.replace('session-', '')} · "
                                                  f"{len(options[n]['steps'])} experiments",
                            label_visibility="collapsed")
        if pick != current:
            st.session_state.summary = options[pick]
        replay = st.button("Replay this run", width="stretch", icon=":material/replay:")

# ---- header ----
st.title("AutoML Agent")
st.markdown('<div class="subtitle">Two AI agents build a credit default model on their own: '
            'they choose models, read validation scores from MLflow, fix overfitting, and '
            'iterate within a fixed budget. The test set is opened once, at the end.</div>',
            unsafe_allow_html=True)
pills = "".join(f'<span class="pill">{p}</span>' for p in
                ["AutoGen", "Claude", "scikit-learn", "LightGBM", "MLflow", "Streamlit", "Docker"])
if GITHUB_URL:
    pills += f'<a class="pill" href="{GITHUB_URL}" target="_blank">GitHub ↗</a>'
if MLFLOW_URL:
    pills += f'<a class="pill" href="{MLFLOW_URL}" target="_blank">MLflow log ↗</a>'
st.markdown(pills, unsafe_allow_html=True)

tab_run, tab_results, tab_how = st.tabs(["Agent run", "Results", "How it works"])

with tab_run:
    if go:
        if df_up is not None:
            try:
                session = ExperimentSession.from_dataframe(df_up, target, budget=budget)
            except ValueError as e:
                st.error(str(e))
                st.stop()
        else:
            with st.spinner("Loading data..."):
                session = ExperimentSession.from_prepared_splits(budget=budget)
        st.session_state.view_id = st.session_state.get("view_id", 0) + 1
        with st.container(key=f"view-{st.session_state.view_id}"):
            result = run_live(session, api_key)
        if result:
            st.session_state.summary = result
            st.session_state.replayed = True
    elif "summary" in st.session_state:
        first_visit = not st.session_state.get("replayed", False)
        st.session_state.replayed = True
        if replay:   # new key = fresh screen, so old messages don't linger faded
            st.session_state.view_id = st.session_state.get("view_id", 0) + 1
        with st.container(key=f"view-{st.session_state.get('view_id', 0)}"):
            show_run(st.session_state.summary, animate=first_visit or replay)
    else:
        st.info("No runs yet. Set a budget in the sidebar and click **Run agent**.")

with tab_results:
    if "summary" in st.session_state:
        show_results(st.session_state.summary)
    else:
        st.write("Results appear here after the first run.")

with tab_how:
    show_how()
