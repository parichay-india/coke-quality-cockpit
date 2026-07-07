# 🚀 Update your existing repo & redeploy on Streamlit Cloud

You already have this app deployed. This guide updates it **in place** with the new-dataset model — no new
repository, no command line. Streamlit Community Cloud redeploys automatically on every commit, so the whole
update takes ~5 minutes in the browser.

---

## What changed in this version

| File | Change | Why it matters |
|---|---|---|
| `coke_quality_model.pkl` | **New model** (KNN for CRI, Voting ensemble for CSR), trained on the richer dataset. **~9 MB, compressed.** | Must be replaced — it is the brain of the app. |
| `app.py` | **New UI**: eight *coke-measurement* sliders (moisture, ash, VM, fixed carbon, M40, M10, +80 mm, −25 mm) instead of the old coal-blend sliders. | The feature set changed, so the dashboard changed. |
| `requirements.txt` | Adds **`xgboost==3.3.0`**; versions matched to the new model. | The CSR Voting ensemble contains an XGBoost model, so the pickle needs `xgboost` to load. |
| `.streamlit/config.toml` | Unchanged (dark theme). | Optional to re-upload. |

> The new `.pkl` is **9 MB**, comfortably under GitHub's 25 MB browser-upload limit — no Git LFS needed.

---

## Step 1 — Replace the four files on GitHub

Do this in your existing repository, in the browser.

### 1a · Replace the model, the app, and requirements

1. Open your repo on GitHub → click **Add file ▸ Upload files**.
2. Drag in the new **`app.py`**, **`requirements.txt`**, and **`coke_quality_model.pkl`** from this bundle.
   Uploading files with the same names **overwrites** the old ones.
3. Scroll down, keep the default commit message, and click **Commit changes**.

### 1b · (Optional) refresh the theme

Only if you want the exact theme in this bundle: open `.streamlit/config.toml` in your repo → pencil icon →
paste the new contents → **Commit changes**. If you're happy with your current theme, skip this.

> **Keep the deploy-critical files in the repo root.** `app.py`, `requirements.txt`, and
> `coke_quality_model.pkl` must sit at the top level of the repo (next to each other), exactly where they
> were before. The notebook and report can live in sub-folders.

---

## Step 2 — Let Streamlit Cloud redeploy

Committing to the branch your app watches triggers an automatic rebuild. Watch it from
<https://share.streamlit.io> → your app.

* If it doesn't pick up the change within a minute, open the app's **⋮ menu ▸ Reboot app**.
* The build should finish in well under a minute — the pinned versions all have prebuilt wheels, so there is
  **no slow source-compile**.

That's it. Your existing URL now serves the new model.

---

## Step 3 — Confirm it works

Open your app URL. You should see **eight coke-measurement sliders** and two gauges. With the sliders at their
default (typical) positions the gauges should read roughly **CRI ≈ 23** and **CSR ≈ 64** — the plant's normal
operating point. Move a slider (e.g. raise **Coke +80 mm**) and both gauges should respond instantly.

---

## Troubleshooting

**Build fails on `xgboost` / "No module named xgboost".**
Make sure the **new** `requirements.txt` (which includes `xgboost==3.3.0`) replaced the old one. The CSR model
needs it to unpickle.

**Unpickling error / `InconsistentVersionWarning` / "cannot load model".**
The library versions must match the ones that built the `.pkl`. Use the `requirements.txt` from **this** bundle
unchanged — it pins `scikit-learn==1.8.0`, `numpy==2.4.4`, `pandas==3.0.2`, `scipy==1.17.1`, `xgboost==3.3.0`,
`joblib==1.5.3`, all with prebuilt wheels for current Python.

**"coke_quality_model.pkl not found."**
The `.pkl` must be in the repo **root**, next to `app.py` (not inside a folder). Re-upload it there.

**App shows the old sliders.**
The browser or Streamlit cached the old app. **Reboot app** from the ⋮ menu and hard-refresh the page.

**Upload rejected as too large.**
It shouldn't be — the model is ~9 MB. If you somehow have the uncompressed 40 MB version, re-export it from the
notebook (the save step uses `compress=6`).

---

## Optional: pin the Python version

In the app's **Advanced settings** you can set **Python 3.12** (the version the model was built on). It isn't
required — the pinned packages have wheels for Python 3.11–3.14 — but it mirrors the build environment exactly.

---

## File checklist (repo root)

- [ ] `app.py` (new, eight coke sliders)
- [ ] `requirements.txt` (new, includes `xgboost`)
- [ ] `coke_quality_model.pkl` (new, ~9 MB)
- [ ] `.streamlit/config.toml` (optional refresh)
