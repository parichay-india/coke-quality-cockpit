"""
Coke Quality Cockpit — predict CSR & CRI from routine coke measurements.

Loads coke_quality_model.pkl (produced by the Colab notebook) and predicts the two
coke-quality indices from eight quick coke-plant measurements. The feature engineering
here is byte-identical to the notebook so predictions match exactly.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

# ----------------------------------------------------------------------------- theme
st.set_page_config(page_title="CokeSense : Coke Quality Cockpit · CSR & CRI", page_icon="🔥", layout="wide")
st.markdown("""
<style>
:root { --ember:#E8521E; --steel:#2A9DF4; --ink:#12181F; --panel:#1B242D; }
.stApp { background: radial-gradient(1200px 600px at 20% -10%, #22303c 0%, #12181F 55%); color:#E6ECF2; }
section[data-testid="stSidebar"] { background:#161E26; border-right:1px solid #2A3844; }
h1,h2,h3,h4 { color:#F2F6FA !important; letter-spacing:.2px; }
.block-container { padding-top:1.4rem; }
.kpi { background:linear-gradient(180deg,#1E2833,#171F27); border:1px solid #2C3A47; border-radius:16px;
       padding:14px 18px; box-shadow:0 6px 22px rgba(0,0,0,.35); }
.badge { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600;
         background:#26333f; color:#9fd0ff; border:1px solid #34485a; margin-right:6px; }
.small { color:#93A2B1; font-size:13px; }
.stSlider label { color:#D7E1EA !important; font-weight:600; }
hr { border-color:#2A3844; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------- engineering (must match training)
def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["FC_Ash_ratio"] = frame["Coke_FC"] / frame["Coke_Ash"].replace(0, np.nan)
    frame["Strength_idx"] = frame["M40"] - frame["M10"]
    frame["Ash_VM"]       = frame["Coke_Ash"] * frame["Coke_VM"]
    return frame

@st.cache_resource
def load_bundle():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in ("coke_quality_model.pkl",
              os.path.join(here, "coke_quality_model.pkl"),
              os.path.join(here, "artifacts", "coke_quality_model.pkl")):
        if os.path.exists(p):
            return joblib.load(p), p
    return None, None

bundle, path = load_bundle()
if bundle is None:
    st.error("⚠️ **coke_quality_model.pkl not found.** Put it in the repository root, next to app.py.")
    st.stop()

PIPELINES = bundle["pipelines"]
BASE      = bundle["base_features"]
FINAL     = bundle["final_features"]
RANGES    = bundle["feature_ranges"]
MEDIANS   = bundle["feature_medians"]
NAMES     = bundle["friendly_names"]
METRICS   = bundle["metrics"]
BOUNDS    = bundle["target_bounds"]
BEST      = bundle["best_model_name"]

# ----------------------------------------------------------------------------- header
st.markdown("# 🔥 Coke Quality Cockpit")
st.markdown('<div class="small">Predicting <b>CSR</b> (strength after reaction) and '
            '<b>CRI</b> (reactivity index) from routine coke measurements — an early-warning aid, '
            'not a replacement for the laboratory reaction test.</div>', unsafe_allow_html=True)
st.markdown(f'<span class="badge">CRI model · {BEST["CRI"]}</span>'
            f'<span class="badge">CSR model · {BEST["CSR"]}</span>'
            f'<span class="badge">trained on ~1,250 days · SAIL Bokaro</span>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------- sliders
st.sidebar.markdown("## ⚙️ Coke measurements")
st.sidebar.markdown('<div class="small">Set the eight routine measurements; predictions update live.</div>',
                    unsafe_allow_html=True)
vals = {}
for f in BASE:
    lo, med, hi = RANGES[f]["min"], RANGES[f]["median"], RANGES[f]["max"]
    step = max((hi - lo) / 100.0, 0.001)
    vals[f] = st.sidebar.slider(NAMES.get(f, f), float(round(lo, 3)), float(round(hi, 3)),
                                float(round(med, 3)), step=float(round(step, 3)))
if st.sidebar.button("↺ Reset to typical blend"):
    st.rerun()

# ----------------------------------------------------------------------------- predict
row = {f: vals.get(f, MEDIANS[f]) for f in BASE}
X = add_engineered_features(pd.DataFrame([row])).reindex(columns=FINAL)
pred = {t: float(PIPELINES[t].predict(X)[0]) for t in ("CRI", "CSR")}

# quality banding
def cri_status(v):
    if v < 21: return "Excellent", "#2ecc71"
    if v < 25: return "On-spec", "#f1c40f"
    return "Watch", "#e74c3c"
def csr_status(v):
    if v > 67: return "Excellent", "#2ecc71"
    if v > 63: return "On-spec", "#f1c40f"
    return "Watch", "#e74c3c"

def gauge(title, value, lo, hi, better_high, mean):
    s_txt, s_col = (csr_status if better_high else cri_status)(value)
    if better_high:
        steps = [{"range":[lo,63],"color":"#4a2222"},{"range":[63,67],"color":"#4a4322"},{"range":[67,hi],"color":"#22432e"}]
    else:
        steps = [{"range":[lo,21],"color":"#22432e"},{"range":[21,25],"color":"#4a4322"},{"range":[25,hi],"color":"#4a2222"}]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(value,2),
        number={"font":{"size":46,"color":"#F2F6FA"}},
        title={"text":f"<b>{title}</b><br><span style='font-size:13px;color:{s_col}'>{s_txt}</span>","font":{"size":20,"color":"#E6ECF2"}},
        gauge={"axis":{"range":[lo,hi],"tickcolor":"#7f8c8d","tickfont":{"color":"#93A2B1"}},
               "bar":{"color":"#E8521E" if not better_high else "#2A9DF4","thickness":0.28},
               "bgcolor":"rgba(0,0,0,0)","borderwidth":0,"steps":steps,
               "threshold":{"line":{"color":"#DDE6EE","width":3},"thickness":0.85,"value":mean}}))
    fig.update_layout(height=300, margin=dict(l=20,r=20,t=70,b=10), paper_bgcolor="rgba(0,0,0,0)", font={"color":"#E6ECF2"})
    return fig

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(gauge("CRI — reactivity (lower is better)", pred["CRI"], BOUNDS["CRI"][0], BOUNDS["CRI"][1], False, 22.1),
                    use_container_width=True, config={"displayModeBar":False})
with c2:
    st.plotly_chart(gauge("CSR — strength after reaction (higher is better)", pred["CSR"], BOUNDS["CSR"][0], BOUNDS["CSR"][1], True, 65.2),
                    use_container_width=True, config={"displayModeBar":False})

# dashed marker = dataset mean
st.markdown('<div class="small">The thin white marker on each dial is the historical plant average '
            '(CRI ≈ 22.1, CSR ≈ 65.2). Predictions far from it are worth confirming with the lab.</div>',
            unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------- context row
a, b_, c = st.columns(3)
with a:
    st.markdown('<div class="kpi"><h4>Predicted CRI</h4>'
                f'<div style="font-size:30px;font-weight:700;color:#E8521E">{pred["CRI"]:.2f}</div>'
                f'<div class="small">typical error ± {METRICS["CRI"]["MAE"]:.2f} · CV R² {METRICS["CRI"]["R2"]:.2f}</div></div>',
                unsafe_allow_html=True)
with b_:
    st.markdown('<div class="kpi"><h4>Predicted CSR</h4>'
                f'<div style="font-size:30px;font-weight:700;color:#2A9DF4">{pred["CSR"]:.2f}</div>'
                f'<div class="small">typical error ± {METRICS["CSR"]["MAE"]:.2f} · CV R² {METRICS["CSR"]["R2"]:.2f}</div></div>',
                unsafe_allow_html=True)
with c:
    st.markdown('<div class="kpi"><h4>How to use this</h4>'
                '<div class="small">The process is tightly controlled, so predictions cluster near the plant '
                'average. Use the dials to spot when a coke sample is trending off-spec — hours before the '
                'slow CSR/CRI reaction test — then confirm in the lab.</div></div>', unsafe_allow_html=True)

with st.expander("What the model is using"):
    st.markdown(
        "The model reads eight routine coke measurements — moisture, ash, volatile matter, fixed carbon, "
        "the Micum **M40** (strength) and **M10** (abrasion), and the **+80 mm** / **-25 mm** size fractions — "
        "plus three engineered features (fixed-carbon/ash ratio, M40−M10, ash×VM). The strongest drivers in "
        "this dataset are the **coke size fractions and M40**. Because CSR and CRI are governed mainly by coal "
        "properties that vary little here, accuracy is moderate but the typical error is small, which is what "
        "makes the tool useful for early warning.")
st.caption("Coke Quality Cockpit · SAIL Bokaro · predictions are decision support, not a lab substitute.")
