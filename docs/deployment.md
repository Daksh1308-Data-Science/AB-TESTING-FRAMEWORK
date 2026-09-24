# Deployment: Streamlit Community Cloud

The dashboard is a Streamlit app, deployed to **Streamlit Community Cloud**
(its native host) straight from this GitHub repo. This is the recommended
path: no build config, no Vercel serverless hacks, and the committed
`data/` dataset ships inside the repo.

- App entry file: `dashboard/app.py`
- Repo: [`Daksh1308-Data-Science/AB-TESTING-FRAMEWORK`](https://github.com/Daksh1308-Data-Science/AB-TESTING-FRAMEWORK), branch `master`
- Deployed URL (example): `https://ab-testing-framework.streamlit.app`

> Why not Vercel? Streamlit keeps a WebSocket open for the whole dashboard
> session, which fights Vercel's serverless model (session duration caps,
> per-instance cold starts, per-instance recompute). Community Cloud is the
> one-click, free, supported host. If a Vercel presence is ever required,
> the pragmatic split is Vercel for the front page + a link out to this app.

## Prerequisites

- Public GitHub repo with the app (this repo already satisfies everything):
  - `requirements.txt` with version floors — Cloud installs the latest
    compatible versions.
  - `pymc` stays commented out — `bayesian.py`'s `engine="auto"` falls back
    to exact closed-form Beta sampling, so the deploy installs fast and the
    app works without the heavy optional dependency.
  - Real data committed at `data/ab_data.csv` + `data/countries.csv` —
    `abtest/real_data.py` resolves them relative to the repo root, so no
    external storage is needed.
  - No secrets or environment variables required.

## Deploy (one-time, ~2 minutes)

1. Go to **https://share.streamlit.io** and **Sign in with GitHub**.
   Authorize access for the `Daksh1308-Data-Science` account.
2. Click **Deploy an app**.
3. Select:
   - Repository: `Daksh1308-Data-Science/AB-TESTING-FRAMEWORK`
   - Branch: `master`
   - Main file path: `dashboard/app.py`
4. Expand **Advanced settings** → Python version: **3.12** (or 3.11/3.13 —
   the code is version-agnostic; 3.14 is not yet offered by the platform).
5. Click **Deploy** and watch the build log. The first build installs
   numpy/scipy/pandas/statsmodels/plotly/streamlit (a few minutes).

## Post-deploy checklist

Run through these once the app is live:

1. URL loads — title + sidebar appear.
2. **Frequentist** tab — z-test decision table and lift forest plot render.
3. Sidebar → **Data source: Real: Udacity `ab_data.csv`** → the data-quality
   block shows `Raw rows 294,478 → Clean 286,690`, SRM **OK**, and the
   treatment row reads `p = 0.23, reject = False` (the honest null result).
   This proves the committed CSV resolved correctly on the server.
4. **Bayesian** tab — posteriors, P(best), expected loss, and the sequential
   chart render; the engine note should report `closed_form`.
5. **Simulations** tab — the four sims complete (seeded; cached after first
   run).

## Updating the live app

- **Every push to `master` auto-redeploys.** There is no CI step.
  (Other branches can be re-deployed manually from the Cloud dashboard.)
- To roll back: open the app in the Cloud dashboard → **Redeploy** → pick a
  previous commit.

## Free-tier behavior to expect

- Apps **sleep when idle**; the first visit after an idle period is slower
  (cold start + recompute of the seeded simulations), then `st.cache_data`
  makes reruns fast.
- The heavy tabs (Simulations, and the real-data Sequential monitor) are
  cached server-side per instance — expect a one-time wait on first load.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Build fails on a dependency | Version floor too low for Python 3.12 | Raise the floor in `requirements.txt` (e.g. `numpy>=1.24` → newer), push, redeploy |
| Real-data block shows errors | `data/` not committed or path moved | Keep CSVs at `data/` under the repo root |
| App renders but a tab errors | Runtime exception on server | Reproduce locally, fix, push (auto-redeploy) |
| First load very slow | Cold start + sims recompute | Normal on free tier; wait for spinners |