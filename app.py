"""PID Log Analyzer for line-follower robots — Streamlit app.

Run:  python -m streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analysis.diagnose import Thresholds, Finding, diagnose
from analysis.load import LogError, load_log
from analysis.metrics import compute_metrics, find_events

SAMPLE_DIR = Path(__file__).parent / "sample_data"

SEV_CONFIG = {
    "bad": {
        "color": "#ff0055",
        "bg": "rgba(255, 0, 85, 0.08)",
        "badge": "CRITICAL FAULT",
        "icon": "🛑",
        "border": "#ff0055",
        "glow": "0 0 18px rgba(255, 0, 85, 0.35)",
    },
    "warn": {
        "color": "#ffaa00",
        "bg": "rgba(255, 170, 0, 0.08)",
        "badge": "ATTENTION REQ",
        "icon": "⚠️",
        "border": "#ffaa00",
        "glow": "0 0 18px rgba(255, 170, 0, 0.35)",
    },
    "info": {
        "color": "#00f0ff",
        "bg": "rgba(0, 240, 255, 0.08)",
        "badge": "TELEMETRY NOTE",
        "icon": "ℹ️",
        "border": "#00f0ff",
        "glow": "0 0 18px rgba(0, 240, 255, 0.35)",
    },
    "ok": {
        "color": "#39ff14",
        "bg": "rgba(57, 255, 20, 0.08)",
        "badge": "SYSTEM NOMINAL",
        "icon": "✅",
        "border": "#39ff14",
        "glow": "0 0 18px rgba(57, 255, 20, 0.35)",
    },
}

st.set_page_config(
    page_title="PID Log Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- CYBER CSS INJECTION
CYBER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600;700&family=Share+Tech+Mono&display=swap');

:root {
  --neon-cyan: #00f0ff;
  --neon-pink: #ff0055;
  --neon-lime: #39ff14;
  --neon-amber: #ffaa00;
  --void-bg: #07090e;
  --text-main: #f1f5f9;
  --text-bright: #ffffff;
}

/* Global Body Font & Size */
html, body, [class*="css"], .stApp {
  font-family: 'Rajdhani', sans-serif !important;
  font-size: 15px !important;
  color: var(--text-main) !important;
  background-color: var(--void-bg) !important;
  background-image: 
    radial-gradient(circle at 12% 15%, rgba(0, 240, 255, 0.07) 0%, transparent 45%),
    radial-gradient(circle at 88% 85%, rgba(255, 0, 85, 0.07) 0%, transparent 45%),
    linear-gradient(rgba(0, 240, 255, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 240, 255, 0.035) 1px, transparent 1px) !important;
  background-size: 100% 100%, 100% 100%, 36px 36px, 36px 36px !important;
}

/* Paragraphs and general text */
p, span, label, div {
  font-size: 15px;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
  font-family: 'Orbitron', sans-serif !important;
  letter-spacing: 1.2px !important;
  color: #ffffff !important;
}

/* Neon Header HUD */
.mecha-hud-banner {
  background: linear-gradient(180deg, rgba(12, 16, 26, 0.95) 0%, rgba(8, 11, 18, 0.85) 100%);
  border: 1px solid rgba(0, 240, 255, 0.28);
  border-left: 5px solid var(--neon-cyan);
  border-radius: 6px;
  padding: 18px 24px;
  margin-bottom: 22px;
  box-shadow: 0 0 25px rgba(0, 240, 255, 0.15), inset 0 0 20px rgba(0, 240, 255, 0.04);
}

.mecha-top-telemetry {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  font-family: 'Share Tech Mono', monospace;
  font-size: 14px;
  letter-spacing: 1.5px;
  color: #94a3b8;
  border-bottom: 1px solid rgba(0, 240, 255, 0.15);
  padding-bottom: 6px;
}

.hud-pulse-tag {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--neon-cyan);
  font-weight: 600;
  text-shadow: 0 0 8px rgba(0, 240, 255, 0.7);
}

.hud-beacon {
  width: 8px;
  height: 8px;
  background-color: var(--neon-lime);
  border-radius: 50%;
  box-shadow: 0 0 10px var(--neon-lime);
  animation: neon-blink 1.4s infinite alternate;
}

@keyframes neon-blink {
  0% { opacity: 0.35; transform: scale(0.85); }
  100% { opacity: 1; transform: scale(1.2); }
}

.mecha-main-title {
  font-family: 'Orbitron', sans-serif !important;
  font-size: 2.1rem !important;
  font-weight: 900 !important;
  color: #ffffff !important;
  margin: 0 !important;
  text-shadow: 0 0 15px rgba(0, 240, 255, 0.6) !important;
}

.mecha-subtitle {
  font-family: 'Rajdhani', sans-serif !important;
  font-size: 1.05rem !important;
  font-weight: 600 !important;
  color: #e2e8f0 !important;
  letter-spacing: 0.8px !important;
  margin-top: 5px !important;
  margin-bottom: 0 !important;
}

/* Sidebar Styling - Brighter Text & Clear Labels */
section[data-testid="stSidebar"] {
  background-color: #090c14 !important;
  border-right: 1px solid rgba(0, 240, 255, 0.22) !important;
}

section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
  color: var(--neon-cyan) !important;
  font-family: 'Orbitron', sans-serif !important;
  font-size: 1rem !important;
  letter-spacing: 1.2px !important;
  text-shadow: 0 0 10px rgba(0, 240, 255, 0.4) !important;
}

section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] p {
  color: #f1f5f9 !important;
  font-size: 14.5px !important;
  font-weight: 500 !important;
}

/* Metric Cards */
[data-testid="stMetric"] {
  background: rgba(12, 16, 26, 0.85) !important;
  border: 1px solid rgba(0, 240, 255, 0.28) !important;
  border-radius: 6px !important;
  padding: 14px 18px !important;
  box-shadow: 0 0 15px rgba(0, 240, 255, 0.08) !important;
  transition: all 0.25s ease !important;
}

[data-testid="stMetric"]:hover {
  border-color: var(--neon-cyan) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 0 25px rgba(0, 240, 255, 0.35) !important;
}

[data-testid="stMetricLabel"] p {
  font-family: 'Orbitron', sans-serif !important;
  font-size: 0.76rem !important;
  color: #cbd5e1 !important;
  text-transform: uppercase !important;
  letter-spacing: 1.2px !important;
  font-weight: 600 !important;
}

[data-testid="stMetricValue"] div {
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 1.85rem !important;
  font-weight: 700 !important;
  color: var(--neon-cyan) !important;
  text-shadow: 0 0 12px rgba(0, 240, 255, 0.7) !important;
}

/* Primary Action Buttons */
div.stButton > button[kind="primary"] {
  font-family: 'Orbitron', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  background: linear-gradient(135deg, #00f0ff 0%, #7928ca 50%, #ff0055 100%) !important;
  color: #ffffff !important;
  border: 1px solid var(--neon-cyan) !important;
  border-radius: 4px !important;
  padding: 0.65rem 1.6rem !important;
  box-shadow: 0 0 20px rgba(0, 240, 255, 0.4) !important;
}

div.stButton > button[kind="primary"]:hover {
  transform: translateY(-2px) scale(1.02) !important;
  box-shadow: 0 0 35px rgba(0, 240, 255, 0.8), 0 0 45px rgba(255, 0, 85, 0.5) !important;
}

/* Tabs */
[data-baseweb="tab-list"] {
  background: rgba(10, 14, 24, 0.85) !important;
  border-bottom: 2px solid rgba(0, 240, 255, 0.2) !important;
  border-radius: 6px 6px 0 0 !important;
  gap: 8px !important;
}

[data-baseweb="tab"] {
  font-family: 'Orbitron', sans-serif !important;
  font-size: 0.85rem !important;
  letter-spacing: 1.2px !important;
  color: #94a3b8 !important;
  padding: 10px 24px !important;
  text-transform: uppercase !important;
}

[data-baseweb="tab"][aria-selected="true"] {
  color: var(--neon-cyan) !important;
  text-shadow: 0 0 12px rgba(0, 240, 255, 0.85) !important;
  border-bottom: 3px solid var(--neon-cyan) !important;
  background: rgba(0, 240, 255, 0.05) !important;
}

/* Findings Card with Brighter Legible Text */
.cyber-finding-card {
  background: rgba(11, 15, 25, 0.95);
  border-radius: 6px;
  padding: 18px 22px;
  margin-bottom: 14px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  transition: transform 0.2s ease;
}

.cyber-finding-card:hover {
  transform: translateX(3px);
}

.cyber-card-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}

.cyber-badge {
  font-family: 'Orbitron', sans-serif;
  font-size: 0.68rem;
  font-weight: 800;
  letter-spacing: 1.2px;
  padding: 3px 8px;
  border-radius: 3px;
}

.cyber-card-title {
  font-family: 'Orbitron', sans-serif;
  font-size: 1.05rem;
  font-weight: 700;
  letter-spacing: 0.8px;
}

.cyber-card-body {
  font-size: 15px !important;
  color: #f8fafc !important;
  line-height: 1.55;
  margin-bottom: 10px;
  font-weight: 500;
}

.cyber-card-evidence {
  font-family: 'Share Tech Mono', monospace;
  font-size: 14px !important;
  color: #cbd5e1 !important;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

.cyber-evidence-tag {
  background: rgba(0, 240, 255, 0.09);
  border: 1px solid rgba(0, 240, 255, 0.3);
  color: var(--neon-cyan);
  padding: 2px 8px;
  border-radius: 3px;
}

/* File Uploader Dropzone */
section[data-testid="stFileUploadDropzone"] {
  background: rgba(10, 14, 24, 0.9) !important;
  border: 1px dashed var(--neon-cyan) !important;
  border-radius: 6px !important;
}

/* Selectbox & Inputs */
div[data-baseweb="select"] > div, 
div[data-baseweb="input"] > div {
  background: #0d121f !important;
  border-color: rgba(0, 240, 255, 0.35) !important;
  color: var(--neon-cyan) !important;
  font-family: 'Share Tech Mono', monospace !important;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
  width: 7px;
  height: 7px;
}
::-webkit-scrollbar-track {
  background: #07090e;
}
::-webkit-scrollbar-thumb {
  background: rgba(0, 240, 255, 0.35);
  border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
  background: var(--neon-cyan);
}
</style>
"""
st.markdown(CYBER_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers

def run_label(name: str, df: pd.DataFrame) -> str:
    """Build a human-readable label like 'run_good (Kp=2.5, Kd=0.08)'."""
    bits = []
    for col in ("kp", "ki", "kd"):
        if col in df and df[col].notna().any():
            bits.append(f"{col.capitalize()}={df[col].median():g}")
    return f"{name} ({', '.join(bits)})" if bits else name


def load_runs(uploads, use_samples: bool):
    """Load uploaded files; fall back to sample data if nothing uploaded."""
    runs: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for up in uploads or []:
        try:
            df = load_log(up)
            runs[run_label(Path(up.name).stem, df)] = df
        except LogError as exc:
            errors.append(f"**{up.name}**: {exc}")
    if use_samples and not runs:
        for p in sorted(SAMPLE_DIR.glob("run_*.csv")):
            df = load_log(p)
            runs[run_label(p.stem, df)] = df
    return runs, errors


def error_plot(df: pd.DataFrame, events: list[dict]) -> go.Figure:
    """Build a Plotly figure with glowing error trace (+ output)."""
    has_out = "output" in df.columns
    fig = make_subplots(
        rows=2 if has_out else 1, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4] if has_out else [1.0],
        vertical_spacing=0.08,
        subplot_titles=(
            "Error (setpoint − measured)",
            "PID output"
        ) if has_out else ("Error (setpoint − measured)",),
    )

    # 1. Error Trace: Glowing Neon Cyan
    fig.add_trace(
        go.Scatter(
            x=df["t"], y=df["error"], name="error",
            line=dict(width=2.2, color="#00f0ff"),
            mode="lines"
        ),
        row=1, col=1,
    )
    fig.add_hline(
        y=0, line_dash="dot", line_color="rgba(0, 240, 255, 0.45)",
        line_width=1.2, row=1, col=1
    )

    # 2. Big-Error Episodes: Glowing Neon Amber Highlight
    for ev in events:
        fig.add_vrect(
            x0=ev["start_s"],
            x1=ev["start_s"] + max(ev["duration_s"], 0.02),
            fillcolor="rgba(255, 170, 0, 0.22)",
            line=dict(color="rgba(255, 170, 0, 0.75)", width=1.2, dash="dot"),
            row=1, col=1,
        )

    # 3. Output Trace: Glowing Neon Lime
    if has_out:
        fig.add_trace(
            go.Scatter(
                x=df["t"], y=df["output"], name="output",
                line=dict(width=1.8, color="#39ff14"),
                mode="lines"
            ),
            row=2, col=1,
        )

    fig.update_xaxes(
        title_text="time (s)",
        gridcolor="rgba(0, 240, 255, 0.08)",
        zerolinecolor="rgba(0, 240, 255, 0.2)",
        tickfont=dict(family="Share Tech Mono, monospace", color="#00f0ff", size=11),
        title_font=dict(family="Orbitron, sans-serif", color="#00f0ff", size=11),
        row=2 if has_out else 1, col=1
    )
    fig.update_yaxes(
        gridcolor="rgba(0, 240, 255, 0.08)",
        zerolinecolor="rgba(0, 240, 255, 0.2)",
        tickfont=dict(family="Share Tech Mono, monospace", color="#00f0ff", size=11),
        title_font=dict(family="Orbitron, sans-serif", color="#00f0ff", size=11),
    )

    fig.update_layout(
        height=540 if has_out else 380,
        showlegend=False,
        paper_bgcolor="#080b12",
        plot_bgcolor="#06080e",
        font=dict(family="Rajdhani, Orbitron, sans-serif", color="#e2e8f0"),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(family='Orbitron, sans-serif', size=12, color='#00f0ff')

    return fig


def fmt(v, digits=3):
    """Format a metric value for display."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


# ---------------------------------------------------------------- HUD BANNER

st.markdown("""
<div class="mecha-hud-banner">
  <div class="mecha-top-telemetry">
    <span class="hud-pulse-tag"><span class="hud-beacon"></span> Offline CSV analysis</span>
  </div>
  <h1 class="mecha-main-title">🤖 PID Log Analyzer</h1>
  <p class="mecha-subtitle">Line-follower robot control loop diagnostics & rule-based diagnosis</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.markdown("### Input")
    uploads = st.file_uploader(
        "CSV logs (one or more runs)", type=["csv"],
        accept_multiple_files=True,
    )
    use_samples = st.checkbox(
        "Use simulated sample runs when nothing is uploaded", value=True,
    )

    st.markdown("### Settings")
    out_limit = st.number_input(
        "Max |output| your motors accept (0 = auto)", min_value=0.0,
        value=0.0, help="Used to detect clipping. 0 means auto-detect.",
    )

    st.markdown("### Thresholds (tune to your robot)")
    th = Thresholds(
        min_osc_hz=st.number_input(
            "Wobble faster than (Hz) counts as oscillation",
            min_value=0.2, value=1.0, step=0.1,
            help="High-frequency swings count as oscillation; slow drift is track layout.",
        ),
        slow_event_s=st.number_input(
            "Big-error episode longer than (s) = sluggish",
            min_value=0.05, value=0.35, step=0.05,
        ),
    )

    st.divider()
    sidebar_start = st.button("🚀 Start Analysis", type="primary", use_container_width=True)


# ---------------------------------------------------------------- session state

if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

current_uploads = tuple(u.name for u in (uploads or []))
if "last_uploads" not in st.session_state:
    st.session_state.last_uploads = current_uploads
elif st.session_state.last_uploads != current_uploads:
    st.session_state.last_uploads = current_uploads
    st.session_state.analyzed = False

if sidebar_start:
    st.session_state.analyzed = True
    st.toast("Analysis updated!", icon="🚀")


# ---------------------------------------------------------------- load

runs, load_errors = load_runs(uploads, use_samples)
for msg in load_errors:
    st.error(msg)

if not runs:
    st.info(
        "Upload one or more CSV logs in the sidebar. Needed columns: "
        "`time_ms`, and `error` (or `setpoint` + `measured`). Optional: `output`, `kp`, `kd`."
    )
    st.stop()

if not st.session_state.analyzed:
    run_count = len(runs)
    source_desc = f"{run_count} uploaded run(s)" if uploads else f"{run_count} sample run(s)"
    st.markdown(f"""
    <div style="background: rgba(12, 16, 26, 0.9); border: 1px solid rgba(0, 240, 255, 0.3); border-radius: 8px; padding: 24px; box-shadow: 0 0 25px rgba(0, 240, 255, 0.15);">
      <h3 style="color: #00f0ff; margin-top: 0; font-family: 'Orbitron', sans-serif;">Ready to Analyze</h3>
      <p style="color: #f1f5f9; font-size: 15px;">Loaded <b>{source_desc}</b> ready for evaluation.</p>
      <p style="color: #cbd5e1; font-size: 14.5px;">Click the <b>Start Analysis</b> button below or in the sidebar to run diagnostics.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    if st.button("🚀 Start Analysis", type="primary", key="main_start_btn"):
        st.session_state.analyzed = True
        st.rerun()
    st.stop()


# ---------------------------------------------------------------- compute

results: dict[str, tuple[dict, list[Finding]]] = {}
for name, df in runs.items():
    m = compute_metrics(df, output_limit=out_limit or None)
    results[name] = (m, diagnose(m, th))


# ---------------------------------------------------------------- tabs

tab_a, tab_c, tab_h = st.tabs(["Analysis", "Compare runs", "How it works"])


# ---------------------------------------------------------------- analysis
with tab_a:
    name = st.selectbox("Run", list(runs))
    df = runs[name]
    m, findings = results[name]

    # Metric cards
    cols = st.columns(6)
    cols[0].metric("RMS error", fmt(m["rms"]))
    cols[1].metric("Mean |error|", fmt(m["mae"]))
    cols[2].metric("Dominant wobble", f"{fmt(m['spec_freq_hz'], 2)} Hz")
    cols[3].metric(
        "Decay ratio", fmt(m["decay_ratio"], 2),
        help="<1 settling, ~1 sustained, >1 growing",
    )
    cols[3].caption("<1 settling · ≈1 sustained · >1 growing")
    cols[4].metric(
        "Output clipped",
        f"{m['saturation']*100:.0f}%" if m["saturation"] is not None else "–",
    )
    cols[5].metric("Output jitter", fmt(m["jitter_ratio"], 2))

    # Findings
    st.markdown("<h3 style='color: #00f0ff; margin-top: 24px;'>Findings</h3>", unsafe_allow_html=True)
    for f in findings:
        cfg = SEV_CONFIG.get(f.severity, SEV_CONFIG["ok"])
        evidence_html = ""
        if f.evidence:
            tags = "".join(f"<span class='cyber-evidence-tag'>{k} = {v}</span>" for k, v in f.evidence.items())
            evidence_html = f"<div class='cyber-card-evidence'><span style='color: #cbd5e1;'>Evidence:</span> {tags}</div>"

        st.markdown(f"""
        <div class="cyber-finding-card" style="border-left: 4px solid {cfg['border']}; box-shadow: {cfg['glow']};">
          <div class="cyber-card-header">
            <span class="cyber-badge" style="background: {cfg['color']}; color: #000;">{cfg['badge']}</span>
            <span class="cyber-card-title" style="color: {cfg['color']};">{cfg['icon']} {f.title}</span>
          </div>
          <div class="cyber-card-body">{f.advice}</div>
          {evidence_html}
        </div>
        """, unsafe_allow_html=True)

    # Plot
    st.markdown("<h3 style='color: #00f0ff; margin-top: 24px;'>Run plot</h3>", unsafe_allow_html=True)
    st.caption("Orange bands = unusually large errors (e.g. entering a curve)")
    events = m.get("events", [])
    st.plotly_chart(error_plot(df, events), use_container_width=True)

    # All metrics expander
    with st.expander("All metrics"):
        display_m = {k: v for k, v in m.items() if k != "events"}
        st.dataframe(
            pd.DataFrame(
                {"metric": list(display_m),
                 "value": [fmt(v, 4) for v in display_m.values()]}
            ),
            hide_index=True,
            use_container_width=True,
        )


# ---------------------------------------------------------------- compare
with tab_c:
    st.markdown("<h3 style='color: #00f0ff;'>Ranked by RMS error (lower is better)</h3>", unsafe_allow_html=True)
    rows = []
    for n, (m, fs) in results.items():
        worst = max(
            fs, key=lambda f: ["ok", "info", "warn", "bad"].index(f.severity)
        )
        cfg = SEV_CONFIG.get(worst.severity, SEV_CONFIG["ok"])
        rows.append({
            "run": n,
            "RMS error": round(m["rms"], 4),
            "mean |error|": round(m["mae"], 4),
            "decay ratio": fmt(m["decay_ratio"], 2),
            "wobble (Hz)": round(m["spec_freq_hz"], 2),
            "clipped %": (
                round(m["saturation"] * 100, 1)
                if m["saturation"] is not None else None
            ),
            "jitter": fmt(m["jitter_ratio"], 2),
            "main finding": f"{cfg['icon']} {worst.title}",
        })
    table = pd.DataFrame(rows).sort_values("RMS error").reset_index(drop=True)

    st.dataframe(table, hide_index=True, use_container_width=True)

    # Glowing Plotly Bar Chart
    fig_bar = go.Figure(go.Bar(
        x=table["run"],
        y=table["RMS error"],
        marker=dict(
            color=table["RMS error"],
            colorscale=[[0, "#39ff14"], [0.35, "#00f0ff"], [0.7, "#ffaa00"], [1.0, "#ff0055"]],
            line=dict(color="#00f0ff", width=1.5)
        ),
        text=table["RMS error"].round(4),
        textposition="outside",
        textfont=dict(family="Share Tech Mono, monospace", color="#00f0ff", size=12)
    ))
    fig_bar.update_layout(
        paper_bgcolor="#080b12",
        plot_bgcolor="#06080e",
        font=dict(family="Orbitron, sans-serif", color="#00f0ff"),
        margin=dict(l=20, r=20, t=30, b=50),
        xaxis=dict(gridcolor="rgba(0, 240, 255, 0.08)", tickangle=-20),
        yaxis=dict(gridcolor="rgba(0, 240, 255, 0.08)", title="RMS ERROR (lower is better)"),
        height=380,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("""
    <div style="background: rgba(255, 170, 0, 0.08); border-left: 3px solid #ffaa00; padding: 10px 16px; border-radius: 4px; font-size: 14.5px; color: #fde047;">
      ⚠️ <b>Warning:</b> Only compare runs from the same track and base speed. RMS also hides oscillation that happens to be small — read the main finding too.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------- how it works
with tab_h:
    st.markdown("""
### How it works

#### 1. CSV format
Header row, then one row per control loop:
```csv
time_ms,error,output,kp,kd
```
*(Required: timestamp + `error` or `setpoint` + `measured`. Optional: `output`, `kp`, `kd`)*

#### 2. What is measured
- **Swings & Decay Ratio**: Each time the error crosses zero (with a noise-scaled deadband) is one swing. The **decay ratio** is how big each swing is compared with the last one:
  - `< 1.0` $\rightarrow$ Settling (stable damping)
  - `~ 1.0` $\rightarrow$ Sustained oscillation (limit cycle)
  - `> 1.1` $\rightarrow$ Growing oscillation (unstable loop)
- **Dominant Wobble (Welch Spectrum)**: The strongest repeating frequency and how much of the signal's non-DC power it holds. Wobbles $> 1.0$ Hz indicate controller instability, whereas $< 1.0$ Hz reflects physical track curvature.
- **Big-Error Episodes**: Stretches where $|error|$ is far above its usual size, e.g. entering a curve. Evaluated post-bias correction so constant offset doesn't trigger false events.
- **Output Clipping & Jitter**: How often the PID output hits its limit, and how twitchy it is ($\sigma_{\Delta u} / \sigma_u$, a sign that $K_d$ is amplifying sensor noise).

#### 3. Key design decision
A line follower has no step input (setpoint is always error = 0), so classic overshoot / rise-time / settling-time don't apply directly. This tool uses swing/decay analysis, spectrum, and "big-error episodes" (curve entries) instead.

#### 4. Limits (please read)
The findings are rules of thumb with adjustable thresholds, not a tuning guarantee. They were tuned on simulated data and must be validated on real robot logs. They cannot see the track, so a slow wobble caused by the track layout can look like a controller problem. Wobble slower than the oscillation threshold (default 1 Hz) is treated as the track. Always confirm by changing the gain and re-running.
    """)
