# 🔥 Coke Quality Cockpit — CSR & CRI Prediction

Predicts two metallurgical coke-quality indices — **CSR** (Coke Strength after Reaction, higher is better)
and **CRI** (Coke Reactivity Index, lower is better) — from routine coke-plant measurements, using data from
**SAIL Bokaro Steel Plant** (Research & Control Laboratory, 2012–2023).

CSR/CRI normally come from a slow furnace reaction test. This tool estimates them in seconds from quick
proximate and Micum measurements, giving an **early-warning signal** — not a replacement for the lab test.

## 🎛️ Live dashboard
Eight sliders (coke moisture, ash, VM, fixed carbon, Micum M40 & M10, +80 mm & −25 mm size fractions) → live
CRI and CSR gauges. Built with Streamlit + Plotly.

## 📊 Honest performance
The plant runs a very stable process, so CSR/CRI vary little day-to-day — which caps achievable R². The models
are accurate in absolute terms, which is what makes them useful for early warning:

| Target | Deployed model | CV R² | Typical error (MAE) |
|---|---|---|---|
| CRI | K-Nearest-Neighbours | ≈ 0.31 | ≈ 0.71 points |
| CSR | Voting ensemble (RF + ExtraTrees + XGBoost) | ≈ 0.39 | ≈ 1.08 points |

Fifteen models were compared by 4-fold cross-validation; see the notebook and report for the full analysis.

## 📁 Repository layout
```
├── app.py                     # Streamlit dashboard
├── requirements.txt           # pinned to match the model file
├── coke_quality_model.pkl     # trained models + metadata (~9 MB, compressed)
├── .streamlit/config.toml     # dark industrial theme
├── DEPLOYMENT_GUIDE.md         # how to update this repo & redeploy on Streamlit Cloud
├── notebook/
│   └── CSR_CRI_Prediction_Coke_Oven.ipynb   # full pipeline (Colab-ready)
└── docs/
    └── Coke_Quality_Project_Report.docx     # complete project report
```

## ▶️ Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy / update on Streamlit Cloud
See **DEPLOYMENT_GUIDE.md**. In short: replace `app.py`, `requirements.txt` and `coke_quality_model.pkl`
in this repo through the GitHub website, commit, and Streamlit Cloud redeploys automatically.

## 🔄 Retraining
Open the notebook in Google Colab, upload `Coke_Oven_Data_Set_New.xlsx`, and run all cells. It rebuilds
`coke_quality_model.pkl` and a matching `requirements.txt`. Replace those two files in the repo to ship the
new model.

---
*Decision support for coke-oven operations — always confirm with the standard CSR/CRI reaction test.*
