# Coke Quality Cockpit — predicting CSR & CRI from coke-oven data

An end-to-end machine-learning project that predicts two key measures of metallurgical coke quality —
**CSR** (Coke Strength after Reaction) and **CRI** (Coke Reactivity Index) — directly from the routine
coal-blend and coke measurements a plant already records. It ships as an interactive
**Streamlit dashboard**: move ten sliders, see predicted CSR and CRI on live gauges.

> Built on the SAIL Bokaro Research & Control Laboratory dataset (≈1,221 usable days, 2012–2022).
> Best models: **Random Forest for CRI**, **Extra Trees for CSR**.
> Honest performance — moderate R² with small typical error (≈0.7 CRI points, ≈1.0 CSR points).
> It is an **early-warning decision-support tool**, not a replacement for the laboratory test.

---

## Repository layout

```
.
├── app.py                          # the Streamlit dashboard (entry point)
├── requirements.txt                # Python dependencies (versions matched to the model)
├── coke_quality_model.pkl          # the trained model bundle (loaded by app.py)
├── .streamlit/
│   └── config.toml                 # dark industrial theme
├── DEPLOYMENT_GUIDE.md             # step-by-step, browser-only deployment
├── notebook/
│   └── CSR_CRI_Prediction_Coke_Oven.ipynb   # full training + model comparison (Colab/T4)
└── docs/
    └── Coke_Quality_Project_Report.docx     # the complete, beginner-friendly project report
```

**Important:** `app.py`, `requirements.txt`, and `coke_quality_model.pkl` must stay in the repository
**root** — that is where the app and Streamlit Community Cloud look for them.

---

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Publish it online (no command line)

Follow **`DEPLOYMENT_GUIDE.md`**. In short: create a free GitHub repo, upload these files through the
website, then connect the repo to [Streamlit Community Cloud](https://share.streamlit.io), which builds
and hosts the app automatically and gives you a shareable link.

---

## Retraining the model

Open `notebook/CSR_CRI_Prediction_Coke_Oven.ipynb` in Google Colab (T4 GPU), upload
`Coke_Oven_Dataset.xls`, and run all cells. On the first run it trains the full model zoo and
compares ~20 models; on later runs it asks whether to rebuild or reuse the saved model. When it
finishes it downloads a fresh **`coke_quality_model.pkl`** and a matching **`requirements.txt`** —
replace the two files in this repo with those, and redeploy.

> Keep the `.pkl` and `requirements.txt` from the **same** run together, so the dashboard's library
> versions match the versions that built the model.

---

## How it works (one paragraph)

The notebook cleans the Excel data, merges the coal-blend (inputs) and coke (outputs) sheets on the
date, engineers two extra features (the ash-to-fixed-carbon ratio dominates both targets), and trains
every model inside an identical `KNNImputer → StandardScaler → model` pipeline. It cross-validates and
ranks them, saves the winner per target into one portable `.pkl` (along with the exact feature list,
the ten dashboard levers, slider ranges, and median fallbacks), and the dashboard reuses that bundle
with byte-identical feature engineering so its predictions match the notebook exactly. The full story,
including a step-by-step code walkthrough for readers new to coke-making and data science, is in
`docs/Coke_Quality_Project_Report.docx`.
