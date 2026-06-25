"""
🏭 Coke Quality Cockpit — CSR & CRI predictor
SAIL Bokaro Steel Plant · coke-oven decision-support dashboard

Loads coke_quality_model.pkl (produced by the Colab notebook) and predicts
Coke Strength after Reaction (CSR) and Coke Reactivity Index (CRI) from the
top operational levers. The feature engineering below is byte-for-byte identical
to the training notebook so predict-time inputs match train-time inputs exactly.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

# ----------------------------------------------------------------------------------
# Page config + industrial "control-room" theme
# ----------------------------------------------------------------------------------
st.set_page_config(page_title="Coke Quality Cockpit · CSR & CRI",
                   page_icon="🏭", layout="wide", initial_sidebar_state="expanded")

EMBER = "#E8521E"   # molten orange  (CRI accent)
STEEL = "#2A9DF4"   # steel blue     (CSR accent)
SLATE = "#37474F"   # gunmetal
CHARCOAL = "#15191E"
PANEL = "#1E242B"
TEXT = "#E6EAEE"
MUTED = "#8A97A3"

st.markdown(f"""
<style>
  .stApp {{ background: radial-gradient(1200px 600px at 20% -10%, #20262e 0%, {CHARCOAL} 55%); color: {TEXT}; }}
  section[data-testid="stSidebar"] > div {{ background: {PANEL}; border-right: 1px solid #2c343d; }}
  h1, h2, h3, h4 {{ color: {TEXT}; letter-spacing: .2px; }}
  .block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; }}

  .hero {{
     background: linear-gradient(135deg, {SLATE} 0%, #222a31 60%);
     border: 1px solid #313a44; border-left: 5px solid {EMBER};
     border-radius: 14px; padding: 18px 22px; margin-bottom: 18px;
     box-shadow: 0 8px 30px rgba(0,0,0,.35);
  }}
  .hero h1 {{ margin: 0; font-size: 1.6rem; }}
  .hero p  {{ margin: 4px 0 0 0; color: {MUTED}; font-size: .92rem; }}

  .metric-card {{
     background: {PANEL}; border: 1px solid #2c343d; border-radius: 12px;
     padding: 14px 16px; height: 100%;
  }}
  .metric-card .label {{ color: {MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: .6px; }}
  .metric-card .value {{ color: {TEXT}; font-size: 1.35rem; font-weight: 700; margin-top: 2px; }}
  .metric-card .sub   {{ color: {MUTED}; font-size: .78rem; }}

  .pill {{ display:inline-block; padding: 3px 10px; border-radius: 999px; font-size:.74rem;
           font-weight:600; letter-spacing:.4px; }}
  .pill-good {{ background: rgba(46,160,67,.18); color:#4ade80; border:1px solid #2ea04344; }}
  .pill-mid  {{ background: rgba(232,153,30,.18); color:#fbbf24; border:1px solid #e8991e44; }}
  .pill-warn {{ background: rgba(232,82,30,.18);  color:#fb7755; border:1px solid #e8521e55; }}

  .footnote {{ color:{MUTED}; font-size:.8rem; border-top:1px solid #2c343d; padding-top:10px; margin-top:8px; }}
  .stSlider label {{ color: {TEXT} !important; font-weight:600; }}
  div[data-testid="stMetricValue"] {{ color: {TEXT}; }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
# Feature engineering — MUST stay identical to the training notebook
# ----------------------------------------------------------------------------------
def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["Ash_FC_ratio"] = frame["Coal_Ash"] / frame["Coal_FC"].replace(0, np.nan)
    frame["Coke_Ash_VM"]  = frame["Coke_Ash"] * frame["Coke_VM"]
    return frame

# ----------------------------------------------------------------------------------
# Load the model bundle
# ----------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_bundle():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in ("coke_quality_model.pkl",
              os.path.join(here, "coke_quality_model.pkl"),
              os.path.join(here, "artifacts", "coke_quality_model.pkl")):
        if os.path.exists(p):
            return joblib.load(p), p
    return None, None

bundle, bundle_path = load_bundle()

if bundle is None:
    st.error("⚠️ **coke_quality_model.pkl not found.** Upload it to the repository root "
             "(next to `app.py`). Generate it by running the Colab notebook.")
    st.stop()

pipelines       = bundle["pipelines"]
best_model_name = bundle.get("best_model_name", {t: "model" for t in pipelines})
FINAL_FEATURES  = bundle["final_features"]
BASE_FEATURES   = bundle["base_features"]
DASH            = bundle.get("dashboard_features", BASE_FEATURES[:10])
RANGES          = bundle.get("feature_ranges", {})
MEDIANS         = bundle.get("feature_medians", {})
FRIENDLY        = bundle.get("friendly_names", {})
METRICS         = bundle.get("metrics", {})
TARGETS         = list(pipelines.keys())

def label_of(f):
    v = FRIENDLY.get(f)
    if isinstance(v, (list, tuple)) and v:
        return v[0]
    return f
def unit_of(f):
    v = FRIENDLY.get(f)
    if isinstance(v, (list, tuple)) and len(v) > 1:
        return v[1]
    return ""
def help_of(f):
    v = FRIENDLY.get(f)
    if isinstance(v, (list, tuple)) and len(v) > 2:
        return v[2]
    return ""

# ----------------------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <h1>🏭 Coke Quality Cockpit — CSR &amp; CRI</h1>
  <p>SAIL Bokaro Steel Plant · predict coke strength &amp; reactivity from coal-blend and coke lab inputs</p>
</div>
""", unsafe_allow_html=True)

# Model info cards
c1, c2, c3, c4 = st.columns(4)
cri_m = METRICS.get("CRI", {}); csr_m = METRICS.get("CSR", {})
with c1:
    st.markdown(f"""<div class="metric-card"><div class="label">CRI model</div>
      <div class="value">{best_model_name.get('CRI','—')}</div>
      <div class="sub">CV R² {cri_m.get('R2', float('nan')):.2f} · MAE {cri_m.get('MAE', float('nan')):.2f}</div></div>""",
      unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card"><div class="label">CSR model</div>
      <div class="value">{best_model_name.get('CSR','—')}</div>
      <div class="sub">CV R² {csr_m.get('R2', float('nan')):.2f} · MAE {csr_m.get('MAE', float('nan')):.2f}</div></div>""",
      unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card"><div class="label">Training rows</div>
      <div class="value">{cri_m.get('n', '—')}</div>
      <div class="sub">Blend ⋈ Coke (2012–2022)</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card"><div class="label">Live levers</div>
      <div class="value">{len(DASH)}</div>
      <div class="sub">top operational drivers</div></div>""", unsafe_allow_html=True)

st.write("")

# ----------------------------------------------------------------------------------
# Sidebar — input levers
# ----------------------------------------------------------------------------------
def slider_for(f):
    r = RANGES.get(f, {})
    med = float(MEDIANS.get(f, r.get("median", 0.0)))
    lo  = float(r.get("min", med * 0.8 if med else 0.0))
    hi  = float(r.get("max", med * 1.2 if med else 1.0))
    if hi <= lo:
        hi = lo + (abs(lo) * 0.1 + 1.0)
    rng = hi - lo
    step = max(round(rng / 100.0, 4), 1e-4)
    # integer-ish levers
    if f in ("LTGK_ord",):
        lo, hi, step, med = float(int(lo)), float(round(hi)), 1.0, float(round(med))
    val = float(min(max(med, lo), hi))
    u = unit_of(f)
    lbl = f"{label_of(f)}" + (f" ({u})" if u else "")
    return st.sidebar.slider(lbl, min_value=round(lo, 3), max_value=round(hi, 3),
                             value=round(val, 3), step=step, help=help_of(f))

st.sidebar.markdown("### ⚙️ Operating point")
st.sidebar.caption("Adjust the top drivers. All other inputs are held at dataset medians.")

def is_coal(f):
    return f.startswith(("Coal", "Scr", "Crush", "FSI", "LTGK"))

coal_levers = [f for f in DASH if is_coal(f)]
coke_levers = [f for f in DASH if not is_coal(f)]

user_vals = {}
if coal_levers:
    st.sidebar.markdown("#### 🪨 Coal blend")
    for f in coal_levers:
        user_vals[f] = slider_for(f)
if coke_levers:
    st.sidebar.markdown("#### 🔥 Coke lab tests")
    for f in coke_levers:
        user_vals[f] = slider_for(f)

reset = st.sidebar.button("↺ Reset to medians", use_container_width=True)
if reset:
    st.rerun()

# ----------------------------------------------------------------------------------
# Build the feature row (sliders + medians) → engineer → predict
# ----------------------------------------------------------------------------------
row = {f: float(MEDIANS.get(f, 0.0)) for f in BASE_FEATURES}
row.update(user_vals)
X1 = pd.DataFrame([row])
X1 = add_engineered_features(X1)
X1 = X1.reindex(columns=FINAL_FEATURES)

pred = {t: float(pipelines[t].predict(X1)[0]) for t in TARGETS}
cri = pred.get("CRI", float("nan"))
csr = pred.get("CSR", float("nan"))

# ----------------------------------------------------------------------------------
# Gauges + interpretation
# ----------------------------------------------------------------------------------
def gauge(value, title, vmin, vmax, bar_color, zones, better):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 46, "color": TEXT}},
        title={"text": title, "font": {"size": 18, "color": MUTED}},
        gauge={
            "axis": {"range": [vmin, vmax], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": zones,
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": TEXT})
    return fig

g1, g2 = st.columns(2)
with g1:
    zones = [
        {"range": [12, 24], "color": "rgba(46,160,67,.30)"},
        {"range": [24, 28], "color": "rgba(232,153,30,.30)"},
        {"range": [28, 40], "color": "rgba(232,82,30,.30)"},
    ]
    st.plotly_chart(gauge(cri, "CRI — Coke Reactivity Index", 12, 40, EMBER, zones, "lower"),
                    use_container_width=True)
with g2:
    zones = [
        {"range": [45, 58], "color": "rgba(232,82,30,.30)"},
        {"range": [58, 64], "color": "rgba(232,153,30,.30)"},
        {"range": [64, 78], "color": "rgba(46,160,67,.30)"},
    ]
    st.plotly_chart(gauge(csr, "CSR — Coke Strength after Reaction", 45, 78, STEEL, zones, "higher"),
                    use_container_width=True)

def pill(text, kind):
    return f'<span class="pill pill-{kind}">{text}</span>'

cri_kind = "good" if cri < 24 else ("mid" if cri <= 28 else "warn")
cri_txt  = "Low reactivity" if cri < 24 else ("Moderate reactivity" if cri <= 28 else "High reactivity")
csr_kind = "good" if csr > 64 else ("mid" if csr >= 58 else "warn")
csr_txt  = "Strong coke" if csr > 64 else ("Acceptable strength" if csr >= 58 else "Weak coke")

st.markdown(f"""
<div class="metric-card">
  <div class="label">Read-out</div>
  <div style="margin-top:8px; line-height:1.9;">
    <b>CRI {cri:.1f}</b> &nbsp; {pill(cri_txt, cri_kind)} &nbsp;—&nbsp;
    lower CRI means the coke resists CO₂ attack better in the blast furnace.<br>
    <b>CSR {csr:.1f}</b> &nbsp; {pill(csr_txt, csr_kind)} &nbsp;—&nbsp;
    higher CSR means the coke stays strong after reaction, supporting the burden.
  </div>
</div>
""", unsafe_allow_html=True)

# Current operating point table
with st.expander("📋 Current operating point (your levers)"):
    show = pd.DataFrame({
        "Driver":  [label_of(f) for f in DASH],
        "Value":   [round(user_vals.get(f, MEDIANS.get(f, np.nan)), 3) for f in DASH],
        "Unit":    [unit_of(f) for f in DASH],
    })
    st.dataframe(show, use_container_width=True, hide_index=True)

# Leaderboard (optional, from bundle)
results = bundle.get("results", {})
if results:
    with st.expander("🏁 Model leaderboards (from training)"):
        for t in TARGETS:
            lb = results.get(t, {}).get("leaderboard")
            if lb is not None:
                st.markdown(f"**{t}**")
                st.dataframe(pd.DataFrame(lb).round(3), use_container_width=True, hide_index=True)

st.markdown(f"""
<div class="footnote">
  CRI/CSR depend partly on coal-rank &amp; rheology signals not captured in this dataset and on day-to-day
  process variation, so the models explain a <b>moderate</b> share of variance with a <b>low absolute error</b>
  (a few CRI points, ~1–2 CSR points). Use this as early-warning decision-support — confirm critical calls with the lab test.
  &nbsp;·&nbsp; Model bundle: <code>{os.path.basename(bundle_path)}</code>
</div>
""", unsafe_allow_html=True)
