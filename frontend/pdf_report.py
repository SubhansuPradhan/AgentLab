"""
Formal PDF report for an AgentLab run.

Structure: cover page with document control and the decision, table of
contents, numbered sections, and two appendices. Every section reads from
the final pipeline state and degrades gracefully when something was not
produced, so the PDF always builds.

    pdf_bytes = build_report_pdf(state, logo_bytes=fetch_logo(URL))
"""
import io
import os
import re
import textwrap
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, CondPageBreak, Frame, HRFlowable, Image, KeepTogether,
    NextPageTemplate, PageBreak, PageTemplate, Paragraph, Preformatted, Spacer,
    Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

NAVY = colors.HexColor("#1F3A5F")
GREY = colors.HexColor("#5A6270")
LIGHT = colors.HexColor("#F3F5F8")
RULE = colors.HexColor("#C9D0DA")

VERDICTS = {
    "approve": ("APPROVED", "#E8F5E9", "#2E7D32"),
    "approve_with_warnings": ("APPROVED WITH WARNINGS", "#FFF8E1", "#B26A00"),
    "revise": ("NOT APPROVED, REVISION REQUIRED", "#FDECEA", "#B3261E"),
}

_REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
    "\u2026": "...", "\u2022": "-", "\u2192": "->", "\u2265": ">=", "\u2264": "<=",
    "\u00d7": "x", "\u00a0": " ", "\u2212": "-",
}


def fetch_logo(url: str, timeout: int = 5) -> Optional[bytes]:
    """Download the logo once. Returns None if the network is unavailable."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def _clean(text: Any) -> str:
    """The built-in PDF fonts only cover Latin-1, so map or replace anything else."""
    s = str(text)
    for k, v in _REPLACEMENTS.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


def _esc(text: Any) -> str:
    return _clean(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline(text: str) -> str:
    """Escape, then turn **bold** and `code` into PDF markup."""
    s = _esc(text)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', s)
    return s


def _num(x: Any, digits: int = 4) -> str:
    try:
        v = float(x)
        return "n/a" if v != v else f"{v:.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _styles() -> Dict[str, ParagraphStyle]:
    base = dict(fontName="Helvetica", fontSize=10, leading=14, textColor=colors.black)
    return {
        "body": ParagraphStyle("Body", alignment=TA_JUSTIFY, spaceAfter=6, **base),
        "bullet": ParagraphStyle("Bullet", alignment=TA_LEFT, leftIndent=16, bulletIndent=4, spaceAfter=3, **base),
        "h1": ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=NAVY, spaceBefore=10, spaceAfter=2, keepWithNext=1),
        "h2": ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#2B2F36"), spaceBefore=8, spaceAfter=4, keepWithNext=1),
        "title": ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=28, leading=32, textColor=NAVY, spaceAfter=6),
        "subtitle": ParagraphStyle("Subtitle", fontName="Helvetica", fontSize=13, leading=17, textColor=GREY, spaceAfter=18),
        "cell": ParagraphStyle("Cell", fontName="Helvetica", fontSize=8.5, leading=11),
        "cellb": ParagraphStyle("CellB", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=colors.white),
        "caption": ParagraphStyle("Caption", fontName="Helvetica-Oblique", fontSize=8.5, leading=11, textColor=GREY, alignment=TA_CENTER, spaceAfter=10),
        "mono": ParagraphStyle("Mono", fontName="Courier", fontSize=7.5, leading=9.5),
        "toc_title": ParagraphStyle("TocTitle", fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=NAVY, spaceAfter=10),
        "toc1": ParagraphStyle("Toc1", fontName="Helvetica", fontSize=10.5, leading=20, leftIndent=0),
        "verdict": ParagraphStyle("Verdict", fontName="Helvetica-Bold", fontSize=12, leading=16),
    }


class _Doc(BaseDocTemplate):
    def __init__(self, buf, meta: Dict[str, str]):
        super().__init__(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2.3 * cm,
                         bottomMargin=2.0 * cm, title=f"AgentLab Report: {meta['dataset']}",
                         author="AgentLab", subject="Automated machine learning analysis")
        self.meta = meta
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="main")
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[frame], onPage=self._draw_cover),
            PageTemplate(id="body", frames=[frame], onPage=self._draw_body),
        ])

    def afterFlowable(self, flowable):
        style = getattr(flowable, "style", None)
        if isinstance(flowable, Paragraph) and style is not None and style.name == "H1":
            self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))

    def _draw_cover(self, canvas, doc):
        canvas.saveState()
        w, h = A4
        canvas.setFillColor(NAVY)
        canvas.rect(0, h - 1.1 * cm, w, 1.1 * cm, stroke=0, fill=1)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GREY)
        canvas.drawString(2 * cm, 1.2 * cm, "Generated automatically by AgentLab, a multi-agent machine learning analysis system.")
        canvas.restoreState()

    def _draw_body(self, canvas, doc):
        canvas.saveState()
        w, h = A4
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.6)
        canvas.line(2 * cm, h - 1.7 * cm, w - 2 * cm, h - 1.7 * cm)
        canvas.line(2 * cm, 1.6 * cm, w - 2 * cm, 1.6 * cm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GREY)
        canvas.drawString(2 * cm, h - 1.5 * cm, _clean(f"AgentLab Analysis Report  |  {self.meta['dataset']}"))
        canvas.drawString(2 * cm, 1.1 * cm, _clean(f"Report ID: {self.meta['run_id']}"))
        canvas.drawRightString(w - 2 * cm, 1.1 * cm, f"Page {doc.page}")
        canvas.restoreState()


# ----------------------------------------------------------------------
# small building blocks
# ----------------------------------------------------------------------
def _h1(text: str, S) -> List:
    rule = HRFlowable(width="100%", thickness=0.8, color=NAVY, spaceAfter=8)
    rule.keepWithNext = 1      # keep the heading with the first lines under it
    return [CondPageBreak(6 * cm), Paragraph(_esc(text), S["h1"]), rule]


def _para(text: str, S, style: str = "body"):
    return Paragraph(_inline(text), S[style])


def _markdown(text: Optional[str], S) -> List:
    """Turn the agents' markdown summaries into formal paragraphs and bullets."""
    if not text or not str(text).strip():
        return [_para("Not available for this run.", S)]
    out, para, code, in_code = [], [], [], False

    def flush():
        if para:
            out.append(Paragraph(_inline(" ".join(para)), S["body"]))
            para.clear()

    for raw in str(text).splitlines():
        line = raw.rstrip()
        if line.strip().startswith("```"):
            if in_code:
                wrapped = [w for l in code for w in (textwrap.wrap(_clean(l), 95) or [""])]
                out.append(Preformatted("\n".join(wrapped), S["mono"]))
                out.append(Spacer(1, 6))
                code, in_code = [], False
            else:
                flush()
                in_code = True
            continue
        if in_code:
            code.append(line)
            continue
        stripped = line.strip()
        if not stripped:
            flush()
        elif stripped.startswith("#"):
            flush()
            out.append(Paragraph(_inline(stripped.lstrip("#").strip()), S["h2"]))
        elif re.match(r"^[-*]\s+", stripped):
            flush()
            out.append(Paragraph(_inline(re.sub(r"^[-*]\s+", "", stripped)), S["bullet"], bulletText="-"))
        else:
            para.append(stripped)
    flush()
    return out


def _table(rows: List[List[Any]], widths: List[float], S, zebra: bool = True) -> Table:
    data = []
    for i, row in enumerate(rows):
        style = S["cellb"] if i == 0 else S["cell"]
        data.append([Paragraph(_esc(c), style) for c in row])
    t = Table(data, colWidths=widths, repeatRows=1)
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if zebra:
        for i in range(2, len(rows), 2):
            cmds.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(cmds))
    return t


def _kv_table(pairs: List[tuple], S, widths=(5 * cm, 12 * cm)) -> Table:
    data = [[Paragraph(f"<b>{_esc(k)}</b>", S["cell"]), Paragraph(_esc(v), S["cell"])] for k, v in pairs]
    t = Table(data, colWidths=list(widths))
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, RULE), ("BACKGROUND", (0, 0), (0, -1), LIGHT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def _figure(path: Optional[str], caption: str, S, max_w=15.5 * cm, max_h=9.5 * cm):
    if not path or not os.path.exists(path):
        return []
    try:
        img = Image(path)
        scale = min(max_w / img.imageWidth, max_h / img.imageHeight, 1.0)
        img = Image(path, width=img.imageWidth * scale, height=img.imageHeight * scale)
    except Exception:
        return []
    return [KeepTogether([img, Spacer(1, 3), Paragraph(_esc(caption), S["caption"])])]


def _decision_box(verdict: str, basis: str, S, width) -> Table:
    label, bg, fg = VERDICTS.get(verdict, (verdict.upper(), "#F3F5F8", "#2B2F36"))
    cell = [Paragraph(f'<font color="{fg}">DECISION: {_esc(label)}</font>', S["verdict"]),
            Spacer(1, 3), Paragraph(_inline(basis), S["cell"])]
    t = Table([[cell]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
        ("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor(fg)),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


# ----------------------------------------------------------------------
# content
# ----------------------------------------------------------------------
def _meta(state: Dict[str, Any]) -> Dict[str, str]:
    c = state.get("critique") or {}
    return {
        "dataset": str(state.get("dataset_name", "dataset")),
        "run_id": str(state.get("run_id", "n/a")),
        "target": str(state.get("target_column", "n/a")),
        "task": str(state.get("task_type", "n/a")),
        "model": f"{state.get('model_name', 'n/a')} ({state.get('final_model_choice', 'default')} settings)",
        "verdict": str(c.get("verdict", "not reviewed")),
        "mode": str(c.get("mode", "n/a")),
        "date": datetime.now().strftime("%d %B %Y, %H:%M"),
    }


def _basis(state: Dict[str, Any]) -> str:
    c = state.get("critique") or {}
    flags = c.get("flags") or []
    if not flags:
        return "The independent review found no problems in the checks that were run."
    counts = c.get("severity_counts") or {}
    parts = [f"{n} {sev}" for sev, n in sorted(counts.items(), key=lambda kv: ["critical", "high", "medium", "low"].index(kv[0]))]
    basis = f"The independent review raised {len(flags)} finding(s): {', '.join(parts)}. See section 8."
    if state.get("dropped_by_critic"):
        basis += " Columns removed during revision: " + ", ".join(f"'{x}'" for x in state["dropped_by_critic"]) + "."
    if c.get("mode") == "audit_only":
        basis += " The critic was in audit-only mode, so none of these findings were acted on."
    return basis


def _cover(meta, state, S, logo_bytes) -> List:
    story: List = [Spacer(1, 1.2 * cm)]
    if logo_bytes:
        try:
            img = Image(io.BytesIO(logo_bytes))
            scale = min(4.2 * cm / img.imageWidth, 4.2 * cm / img.imageHeight)
            story += [Image(io.BytesIO(logo_bytes), width=img.imageWidth * scale, height=img.imageHeight * scale), Spacer(1, 0.6 * cm)]
        except Exception:
            pass
    story += [
        Paragraph("AgentLab Analysis Report", S["title"]),
        Paragraph(_esc(f"Automated machine learning analysis of {meta['dataset']}"), S["subtitle"]),
        HRFlowable(width="100%", thickness=1.2, color=NAVY, spaceAfter=14),
        _kv_table([
            ("Report ID", meta["run_id"]), ("Generated", meta["date"]), ("Dataset", meta["dataset"]),
            ("Target variable", meta["target"]), ("Problem type", meta["task"]),
            ("Recommended model", meta["model"]), ("Independent review mode", meta["mode"]),
        ], S),
        Spacer(1, 0.8 * cm),
        _decision_box(meta["verdict"], _basis(state), S, 17 * cm),
    ]
    return story


def _stats_rows(stats: Dict[str, Dict[str, float]], task: str) -> List[List[str]]:
    if not stats:
        return []
    if task == "classification":
        cols = [("Accuracy", "Accuracy"), ("Precision", "Precision"), ("Recall", "Recall"), ("F1", "F1"), ("AUC_ROC", "AUC-ROC")]
    else:
        cols = [("MAE", "MAE"), ("MSE", "MSE"), ("R2", "R2"), ("MAPE", "MAPE (%)")]
    rows = [["Model"] + [c[1] for c in cols]]
    for name, vals in stats.items():
        rows.append([name] + [_num(vals.get(k), 3) for k, _ in cols])
    return rows


def _sections(state: Dict[str, Any], meta, S) -> List:
    W = 17 * cm
    secs: Dict[str, str] = state.get("report_sections") or {}
    task = meta["task"]
    story: List = []

    # 1. Executive summary
    story += _h1("1. Executive Summary", S)
    story.append(_decision_box(meta["verdict"], _basis(state), S, W))
    story.append(Spacer(1, 8))
    story += _markdown(secs.get("Executive Summary"), S)

    # 2. Dataset and scope
    story += _h1("2. Dataset and Scope", S)
    story += _markdown(secs.get("Dataset and Target"), S)
    vr = state.get("validation_report") or {}
    if vr:
        story.append(Paragraph("Data quality at intake", S["h2"]))
        story.append(_kv_table([
            ("Rows", vr.get("n_rows", "n/a")), ("Columns", vr.get("n_columns", "n/a")),
            ("Duplicate rows removed", vr.get("duplicate_rows", "n/a")),
            ("Columns with missing values", len(vr.get("missing_by_column") or {})),
            ("Constant columns dropped", ", ".join(vr.get("constant_columns") or []) or "none"),
        ], S))

    # 3. Data preparation
    story += _h1("3. Data Preparation", S)
    story += _markdown(state.get("validation_summary"), S)
    story += _markdown(state.get("preprocessing_summary"), S)
    prep = state.get("preprocessing_report") or {}
    if prep:
        rows = [["Column", "Type", "Treatment"]]
        for col, r in list(prep.items())[:20]:
            if r.get("type") == "numeric":
                treat = f"impute: {r.get('impute_strategy', 'n/a')}; scale: {r.get('scale_strategy', 'n/a')}"
            else:
                treat = f"impute: {r.get('impute_strategy', 'n/a')}; encode: {r.get('encode_strategy', 'n/a')}"
            rows.append([col, r.get("type", ""), treat])
        story += [Paragraph("Preprocessing decisions", S["h2"]), _table(rows, [5.5 * cm, 2.5 * cm, 9 * cm], S)]
        if len(prep) > 20:
            story.append(_para(f"Showing 20 of {len(prep)} columns.", S, "caption"))

    # 4. Exploratory analysis
    story += _h1("4. Exploratory Analysis", S)
    story += _markdown(state.get("eda_summary"), S)
    for p in (state.get("eda_plots") or [])[:6]:
        story += _figure(p, os.path.splitext(os.path.basename(p))[0].replace("_", " ").title(), S)

    # 5. Hypotheses
    story += _h1("5. Hypotheses", S)
    story += _markdown(state.get("hypothesis_summary"), S)
    hyps = state.get("hypotheses") or []
    results = {r["id"]: r for r in (state.get("hypothesis_results") or [])}
    if hyps:
        rows = [["ID", "Type", "Hypothesis", "Outcome"]]
        for h in hyps:
            rows.append([h.get("id", ""), h.get("type", ""), h.get("claim", ""), (results.get(h.get("id")) or {}).get("status", "not checked")])
        story += [Paragraph("Hypotheses and outcomes", S["h2"]), _table(rows, [1.2 * cm, 2.3 * cm, 10.5 * cm, 3 * cm], S)]
    if state.get("leakage_suspects"):
        story.append(Spacer(1, 4))
        story.append(_para("Suspected data leaks identified before modeling: " + ", ".join(f"'{c}'" for c in state["leakage_suspects"]) + ".", S))

    # 6. Model selection and tuning
    story += _h1("6. Model Selection and Tuning", S)
    story += _markdown(state.get("model_selection_summary"), S)
    stats_rows = _stats_rows(state.get("stats_table") or {}, task)
    if stats_rows:
        n = len(stats_rows[0]) - 1
        story += [Paragraph("Test-set performance of every model compared", S["h2"]),
                  _table(stats_rows, [4.2 * cm] + [(W - 4.2 * cm) / n] * n, S)]
    plan = state.get("tuning_plan") or {}
    if plan:
        story.append(Paragraph("Fine-tuning decision", S["h2"]))
        story.append(_para(("Fine-tuning was required. " if plan.get("required") else "Fine-tuning was not required. ")
                           + " ".join(plan.get("reasons") or []), S))
    comp = state.get("tuning_comparison")
    if comp:
        story.append(_kv_table([
            ("Default settings, test score", _num(comp.get("baseline_test"))), ("Tuned settings, test score", _num(comp.get("tuned_test"))),
            ("Change on test set", f"{comp.get('relative_test_gain_pct', 0):+.2f} percent"), ("Verdict", comp.get("verdict", "n/a")),
            ("Model adopted", state.get("final_model_choice", "default")),
        ], S))
    story += _figure(state.get("confusion_matrix_path"), "Confusion matrix of the final model on the test set", S, max_h=8 * cm)
    story += _figure(state.get("tuning_plot_path"), "Optimization history of the tuning search", S, max_h=8 * cm)

    # 7. Explainability
    story += _h1("7. Explainability", S)
    story += _markdown(state.get("explainability_summary"), S)
    imp = state.get("feature_importance") or []
    if imp:
        rows = [["Rank", "Feature", "Share of importance", "Direction of effect"]]
        for r in imp[:10]:
            rows.append([r.get("rank", ""), r.get("feature", ""), f"{r.get('share_pct', 0):.1f} percent", r.get("direction", "")])
        story += [Paragraph("Ten most influential features", S["h2"]), _table(rows, [1.4 * cm, 5.5 * cm, 3.6 * cm, 6.5 * cm], S)]
    for p in (state.get("shap_plots") or [])[:4]:
        story += _figure(p, os.path.splitext(os.path.basename(p))[0].replace("_", " ").title(), S)

    # 8. Independent review
    story += _h1("8. Independent Review", S)
    c = state.get("critique") or {}
    story.append(_para(f"Verdict: {VERDICTS.get(meta['verdict'], (meta['verdict'],))[0]}. Mode: {meta['mode']}.", S))
    flags = c.get("flags") or []
    if flags:
        rows = [["Severity", "Finding", "Evidence"]] + [[f["severity"].upper(), f["title"], f["evidence"]] for f in flags]
        story += [Paragraph("Findings", S["h2"]), _table(rows, [2.0 * cm, 4.6 * cm, 10.4 * cm], S)]
    story += [Paragraph("Reviewer's commentary", S["h2"])] + _markdown(state.get("critic_summary"), S)
    hist = state.get("revision_history") or []
    if hist:
        rows = [["Round", "Columns removed", "Reason"]] + [[h["round"], ", ".join(h["dropped"]) or "none", "; ".join(h.get("reasons", []))] for h in hist]
        story += [Paragraph("Revision history", S["h2"]), _table(rows, [1.6 * cm, 6 * cm, 9.4 * cm], S)]

    # 9. Conclusion
    story += _h1("9. Conclusion and Recommendations", S)
    story += _markdown(secs.get("Conclusion and Recommendations"), S)

    # Appendix A
    story += _h1("Appendix A. Methodology", S)
    info = state.get("train_test_split_info") or {}
    models = list((state.get("model_metrics") or {}).keys())
    method = [
        f"Data split: {info.get('train_size', 'n/a')} training rows and {info.get('test_size', 'n/a')} held-out test rows, random seed {info.get('random_state', 42)}.",
        "Models compared: " + (", ".join(models) if models else "not recorded") + ", each scored with 5-fold cross validation and once on the held-out test set.",
        "Model choice: the simplest model whose cross-validated score is within one standard error of the best (the one standard error rule).",
        "Tuning: decided by rules from measured facts, run with Optuna, and adopted only when the test score improved by at least 1 percent.",
        "Explanations: SHAP values on held-out rows, with LIME as a second opinion on the worst prediction.",
        "Independent review: the reviewer re-measures the model itself and checks: " + ", ".join(c.get("checks_run") or ["not recorded"]) + ".",
        f"Revision limit: {c.get('max_revisions', 'n/a')}. Revisions used: {state.get('revision_count', 0)}.",
    ]
    story += [Paragraph(_inline(m), S["bullet"], bulletText="-") for m in method]

    # Appendix B
    log = state.get("step_log") or []
    story += _h1("Appendix B. Run Log", S)
    if log:
        rows = [["Step", "Seconds", "Revision round"]] + [[s["step"], f"{s['seconds']:.1f}", s["revision"]] for s in log]
        story.append(_table(rows, [7 * cm, 4 * cm, 6 * cm], S))
        story.append(_para(f"Total pipeline time: {sum(s['seconds'] for s in log):.1f} seconds.", S, "caption"))
    else:
        story.append(_para("No run log was recorded.", S))
    return story


def build_report_pdf(state: Dict[str, Any], logo_bytes: Optional[bytes] = None) -> bytes:
    """Build the formal PDF for a finished run and return it as bytes."""
    S = _styles()
    meta = _meta(state)
    buf = io.BytesIO()
    doc = _Doc(buf, meta)

    toc = TableOfContents()
    toc.levelStyles = [S["toc1"]]
    toc.dotsMinLevel = 0

    story: List = _cover(meta, state, S, logo_bytes)
    story += [NextPageTemplate("body"), PageBreak(), Paragraph("Contents", S["toc_title"]), toc, PageBreak()]
    story += _sections(state, meta, S)
    doc.multiBuild(story)
    return buf.getvalue()