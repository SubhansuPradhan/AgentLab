"""
AgentLab Streamlit app.

Run from the project root:
    streamlit run frontend/app.py

Tabs: Home (upload and run), one tab per agent, and the full report with a
PDF download. Delete the old frontend/pages/ folder, otherwise Streamlit
adds those files to the sidebar as extra pages.
"""
import io
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from frontend.pdf_report import build_report_pdf, fetch_logo

LOGO_URL = "https://i.postimg.cc/ZK59Kpf6/Gemini-Generated-Image-x4637zx4637zx463.png"
DESCRIPTION = (
    "AgentLab is a multi-agent AI scientist that runs the whole machine learning workflow on your data. "
    "Upload a CSV and choose what to predict. Specialist agents then clean and explore the data, propose "
    "hypotheses, compare and tune models, and explain the predictions with SHAP. A critic agent re-checks "
    "the result for data leaks, overfitting and weak signal, sending the pipeline back to fix what it can, "
    "and a formal report is produced at the end."
)
STEP_TEXT = {
    "validate": "Validated and cleaned the data",
    "eda": "Explored the data",
    "hypothesis": "Proposed hypotheses and screened the features",
    "feature_engineering": "Prepared the features",
    "model_selection": "Compared models and decided whether to tune",
    "explainability": "Explained the model with SHAP",
    "verify_hypotheses": "Checked the hypotheses against the explanations",
    "critic": "Critic reviewed the result",
    "apply_revision": "Critic removed flagged columns, running again",
    "report": "Wrote the report",
}
VERDICT_LABEL = {
    "approve": "Approved",
    "approve_with_warnings": "Approved with warnings",
    "revise": "Not approved, revision required",
}

st.set_page_config(page_title="AgentLab", page_icon="🧪", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """<style>
    [data-testid="stSidebar"], [data-testid="stSidebarNav"],
    [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {display: none !important;}
    .block-container {padding-top: 2.5rem;}
    </style>""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_logo() -> bytes:
    return fetch_logo(LOGO_URL)


def show_image(path, caption=None):
    if not path or not Path(str(path)).exists():
        return
    try:
        st.image(str(path), caption=caption, width="stretch")
    except Exception:
        st.image(str(path), caption=caption)


def result():
    return st.session_state.get("result")


def need_result():
    r = result()
    if r is None:
        st.info("Run an analysis from the Home tab first. This tab shows what this agent produced.")
    return r


def verdict_banner(verdict: str, text: str = ""):
    box = {"approve": st.success, "approve_with_warnings": st.warning, "revise": st.error}.get(verdict, st.info)
    box(f"**Decision: {VERDICT_LABEL.get(verdict, verdict)}**" + (f"\n\n{text}" if text else ""))


def table(data, **kw):
    """DataFrame from rows, shown without the index."""
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    st.dataframe(df, hide_index=True, **kw)


def describe_step(node: str, update: dict) -> str:
    text = STEP_TEXT.get(node, node)
    if node == "critic":
        c = update.get("critique") or {}
        text += f": {VERDICT_LABEL.get(c.get('verdict'), c.get('verdict'))}"
        if c.get("flags"):
            text += f" ({len(c['flags'])} finding(s))"
    if node == "apply_revision" and update.get("dropped_by_critic"):
        text = f"Critic removed {', '.join(update['dropped_by_critic'])} and is running the pipeline again"
    return text


def run_with_progress(df, target, name, run_id, critic_on, max_rev, n_trials):
    from graph.workflow import stream_pipeline

    state = {"raw_df": df, "target_column": target, "dataset_name": name, "run_id": run_id,
             "critic_enabled": critic_on, "max_revisions": max_rev, "n_trials": n_trials}
    with st.status("Running the AgentLab pipeline...", expanded=True) as status:
        for node, update in stream_pipeline(df, target, dataset_name=name, run_id=run_id,
                                            critic_enabled=critic_on, max_revisions=max_rev, n_trials=n_trials):
            state.update(update)
            st.write(describe_step(node, update))
        status.update(label="Analysis complete", state="complete", expanded=False)
    return state

def render_header():
    """Logo and name in the left column, description in the right column, above the tabs."""
    try:
        left, right = st.columns([1, 3], vertical_alignment="center")
    except TypeError:                      # older Streamlit without vertical_alignment
        left, right = st.columns([1, 3])
    with left:
        a, b, c = st.columns([1, 2, 1])
        with b:
            st.image(LOGO_URL, width=110)
        st.markdown("<h1 style='text-align:center;margin:0'>AgentLab</h1>", unsafe_allow_html=True)
    with right:
        st.markdown(f"<p style='opacity:0.85;font-size:1.02rem;margin:0'>{DESCRIPTION}</p>", unsafe_allow_html=True)
# ----------------------------------------------------------------------
# tab 1: home
# ----------------------------------------------------------------------
def render_home():
    uploaded = st.file_uploader("Upload a dataset (CSV)", type=["csv"])
    prev = result()
    if uploaded is None:
        if prev is not None:
            st.caption(f"Last run: {prev.get('dataset_name')} ({prev.get('run_id')}). Open the tabs above to see it.")
        return

    try:
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
    except Exception as e:
        st.error(f"Could not read that file as a CSV: {e}")
        return

    st.caption(f"{uploaded.name}: {len(df):,} rows and {len(df.columns)} columns")
    table(df.head(8))

    target = st.selectbox("What do you want to predict?", options=list(df.columns), index=None,
                          placeholder="Choose the target column")
    with st.expander("Options"):
        critic_on = st.toggle("Critic agent active", value=True,
                              help="Off means the critic only records problems and never changes the pipeline.")
        max_rev = st.slider("Maximum revisions", 0, 3, 2)
        n_trials = st.slider("Tuning trials", 5, 50, 20)

    if st.button("Run analysis", type="primary", disabled=target is None):
        name = Path(uploaded.name).stem
        run_id = f"{name}_{datetime.now():%Y%m%d_%H%M%S}"
        try:
            st.session_state.result = run_with_progress(df, target, name, run_id, critic_on, max_rev, n_trials)
            st.session_state.pop("pdf", None)
            st.success("Done. Open the tabs above to see each agent's work, and the last tab for the full report.")
        except Exception as e:
            st.error(str(e))
            with st.expander("Technical details"):
                st.code(traceback.format_exc())


# ----------------------------------------------------------------------
# agent tabs
# ----------------------------------------------------------------------
def render_validation():
    s = need_result()
    if s is None:
        return
    vr = s.get("validation_report") or {}
    c = st.columns(4)
    c[0].metric("Rows", vr.get("n_rows", "n/a"))
    c[1].metric("Columns", vr.get("n_columns", "n/a"))
    c[2].metric("Duplicates removed", vr.get("duplicate_rows", "n/a"))
    c[3].metric("Columns with gaps", len(vr.get("missing_by_column") or {}))
    st.markdown(s.get("validation_summary") or "No summary was produced.")
    if vr.get("missing_by_column"):
        st.subheader("Missing values")
        table([{"column": k, "missing": v["count"], "percent": v["pct"]} for k, v in vr["missing_by_column"].items()])
    if vr.get("constant_columns"):
        st.caption("Constant columns dropped: " + ", ".join(vr["constant_columns"]))
    if vr.get("high_cardinality_columns"):
        st.caption("Likely identifier columns: " + ", ".join(vr["high_cardinality_columns"]))


def render_eda():
    s = need_result()
    if s is None:
        return
    st.markdown(s.get("eda_summary") or "No summary was produced.")
    plots = [p for p in (s.get("eda_plots") or []) if Path(str(p)).exists()]
    for i in range(0, len(plots), 2):
        cols = st.columns(2)
        for col, p in zip(cols, plots[i:i + 2]):
            with col:
                show_image(p, Path(p).stem.replace("_", " ").title())
    summary = (s.get("eda_stats") or {}).get("numeric_summary")
    if summary:
        with st.expander("Numeric summary"):
            st.dataframe(pd.DataFrame(summary).round(3))


def render_hypotheses():
    s = need_result()
    if s is None:
        return
    st.markdown(s.get("hypothesis_summary") or "No summary was produced.")
    if s.get("leakage_suspects"):
        st.warning("Suspected leaks found before modeling: " + ", ".join(f"`{c}`" for c in s["leakage_suspects"]))
    results = {r["id"]: r for r in (s.get("hypothesis_results") or [])}
    if s.get("hypotheses"):
        st.subheader("Hypotheses and what the model showed")
        table([{"id": h["id"], "type": h["type"], "hypothesis": h["claim"],
                "outcome": (results.get(h["id"]) or {}).get("status", "not checked"),
                "detail": (results.get(h["id"]) or {}).get("detail", "")} for h in s["hypotheses"]])
    if s.get("feature_screening"):
        with st.expander("Single-feature screen, best first"):
            table([{k: r.get(k) for k in ("feature", "kind", "metric", "solo_score", "rho")} for r in s["feature_screening"]])
    if s.get("excluded_columns"):
        with st.expander("Columns left out of testing"):
            table([{"column": k, "reason": v} for k, v in s["excluded_columns"].items()])


def render_features():
    s = need_result()
    if s is None:
        return
    fdf = s.get("feature_df")
    if fdf is not None:
        c = st.columns(2)
        c[0].metric("Rows", f"{len(fdf):,}")
        c[1].metric("Model-ready columns", len(fdf.columns))
    st.markdown(s.get("preprocessing_summary") or "No summary was produced.")
    prep = s.get("preprocessing_report") or {}
    if prep:
        rows = []
        for col, r in prep.items():
            step = r.get("scale_strategy") if r.get("type") == "numeric" else r.get("encode_strategy")
            rows.append({"column": col, "type": r.get("type"), "missing %": round(100 * r.get("missing_pct", 0), 1),
                         "imputation": r.get("impute_strategy"), "scaling / encoding": step})
        table(rows)
    if fdf is not None:
        with st.expander("Preview of the prepared table"):
            st.dataframe(fdf.head(10))


def render_model():
    s = need_result()
    if s is None:
        return
    c = st.columns(4)
    c[0].metric("Problem type", s.get("task_type", "n/a"))
    c[1].metric("Recommended model", s.get("model_name", "n/a"))
    c[2].metric("Settings used", s.get("final_model_choice", "default"))
    c[3].metric("Tuning run", "yes" if s.get("tuning_ran") else "no")
    rec = s.get("model_recommendation") or {}
    if rec.get("reasons"):
        st.subheader("Why this model")
        for r in rec["reasons"]:
            st.markdown(f"- {r}")
    st.markdown(s.get("model_selection_summary") or "No summary was produced.")
    if s.get("stats_table"):
        st.subheader("Every model, scored on the held-out test set")
        st.dataframe(pd.DataFrame(s["stats_table"]).T.round(4))
    plan = s.get("tuning_plan") or {}
    if plan:
        st.subheader("Fine-tuning decision")
        (st.info if plan.get("required") else st.warning)(
            ("Fine-tuning required. " if plan.get("required") else "Fine-tuning not required. ") + " ".join(plan.get("reasons") or []))
        if plan.get("params"):
            table([{"setting": p["name"], "range": p["range"], "meaning": p["meaning"]} for p in plan["params"]])
    if s.get("tuning_comparison"):
        st.subheader("Default versus tuned")
        st.dataframe(pd.DataFrame(s.get("comparison_table") or {}).T.round(4))
        st.caption(f"Verdict: {s['tuning_comparison']['verdict']}")
    cols = st.columns(2)
    with cols[0]:
        show_image(s.get("confusion_matrix_path"), "Confusion matrix")
    with cols[1]:
        show_image(s.get("tuning_plot_path"), "Tuning search")


def render_explainability():
    s = need_result()
    if s is None:
        return
    flags = s.get("explainability_flags") or {}
    if flags.get("reliability_note"):
        (st.warning if flags.get("near_chance") or flags.get("possible_leakage") else st.info)(flags["reliability_note"])
    st.markdown(s.get("explainability_summary") or "No summary was produced.")
    imp = s.get("feature_importance") or []
    if imp:
        top = pd.DataFrame(imp[:15])
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Feature importance")
            st.bar_chart(top.set_index("feature")["share_pct"])
        with c2:
            st.subheader("Details")
            table(top[["rank", "feature", "share_pct", "direction"]])
    plots = [p for p in (s.get("shap_plots") or []) if Path(str(p)).exists()]
    for i in range(0, len(plots), 2):
        cols = st.columns(2)
        for col, p in zip(cols, plots[i:i + 2]):
            with col:
                show_image(p, Path(p).stem.replace("_", " ").title())


def render_critic():
    s = need_result()
    if s is None:
        return
    c = s.get("critique") or {}
    verdict_banner(c.get("verdict", "not reviewed"),
                   f"Mode: {c.get('mode', 'n/a')}. Revisions used: {s.get('revision_count', 0)} of {c.get('max_revisions', 'n/a')}.")
    m = c.get("measured") or {}
    if m.get("value") is not None:
        cols = st.columns(4)
        cols[0].metric(m.get("metric", "Score"), f"{m['value']:.3f}")
        cols[1].metric("Train score", f"{m.get('train_score', float('nan')):.3f}")
        cols[2].metric("Test score", f"{m.get('test_score', float('nan')):.3f}")
        cols[3].metric("Test rows", m.get("n_test", "n/a"))
    flags = c.get("flags") or []
    if flags:
        st.subheader("Findings, most serious first")
        table([{"severity": f["severity"], "finding": f["title"], "evidence": f["evidence"],
                "automatic fix": ", ".join(f["fix"]["columns"]) if f.get("fix") else "none"} for f in flags])
    else:
        st.success("None of the checks found a problem.")
    st.markdown(s.get("critic_summary") or "")
    if s.get("revision_history"):
        st.subheader("Revision history")
        table([{"round": h["round"], "columns removed": ", ".join(h["dropped"]), "reason": "; ".join(h.get("reasons", []))}
               for h in s["revision_history"]])


# ----------------------------------------------------------------------
# last tab: the full report and the PDF
# ----------------------------------------------------------------------
def render_report():
    s = need_result()
    if s is None:
        return
    c = s.get("critique") or {}
    left, right = st.columns([4, 1])
    with left:
        st.header(f"Decision report: {s.get('dataset_name', 'dataset')}")
        st.caption(f"Target: {s.get('target_column')} | Model: {s.get('model_name')} | Report ID: {s.get('run_id')}")
    with right:
        try:
            if st.session_state.get("pdf_run") != s.get("run_id"):
                with st.spinner("Building the PDF..."):
                    st.session_state.pdf = build_report_pdf(s, logo_bytes=get_logo())
                    st.session_state.pdf_run = s.get("run_id")
            st.download_button("Download full report (PDF)", data=st.session_state.pdf, type="primary",
                               file_name=f"AgentLab_Report_{s.get('run_id', 'run')}.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"The PDF could not be built: {e}")

    verdict_banner(c.get("verdict", "not reviewed"), f"{len(c.get('flags') or [])} finding(s) from the independent review.")
    for title, body in (s.get("report_sections") or {}).items():
        st.subheader(title)
        st.markdown(body)
    figs = [p for k in ("eda_plots", "shap_plots") for p in (s.get(k) or [])]
    figs += [s.get(k) for k in ("confusion_matrix_path", "tuning_plot_path") if s.get(k)]
    figs = [p for p in figs if Path(str(p)).exists()]
    if figs:
        st.subheader("Figures")
        for i in range(0, len(figs), 2):
            cols = st.columns(2)
            for col, p in zip(cols, figs[i:i + 2]):
                with col:
                    show_image(p, Path(p).stem.replace("_", " ").title())


# ----------------------------------------------------------------------
TABS = ["Home", "Validation", "EDA", "Hypotheses", "Features", "Model", "Explainability", "Critic", "Full Report"]
RENDERERS = [render_home, render_validation, render_eda, render_hypotheses, render_features,
             render_model, render_explainability, render_critic, render_report]

render_header()
for tab, render in zip(st.tabs(TABS), RENDERERS):
    with tab:
        render()