# AGENTS.md — Ground rules for working in this repo

This file is the contract for any AI agent (or human) changing this codebase.
**Read it fully before writing or modifying any file.** If a task conflicts
with it, stop and ask the user.

## 1. Mission

Build an **A/B testing framework for binary conversion metrics** with:

- Frequentist testing (two-proportion z-test, confidence intervals)
- Power analysis / sample-size calculation
- Multiple-testing correction (Bonferroni + Benjamini-Hochberg)
- Bayesian testing (posterior, probability of being best, expected loss)
- Bayesian sequential early stopping
- A Streamlit dashboard exposing all of the above
- Simulation scripts that **measure** the project's headline results

## 2. Non-negotiable decisions (confirmed with the user)

1. **Conversion-only binary metrics.** No continuous metrics (no t-tests, no
   Normal-Normal models, no revenue/time-series metrics).
2. **Sequential testing is Bayesian only.** No frequentist alpha-spending, no
   O'Brien-Fleming, no SPRT.
3. **Bayesian engine:** use `pymc` when importable; otherwise fall back to
   exact closed-form Beta sampling. **`pymc3` is renamed/deprecated — never
   install or import it.**
4. **Multi-arm tests** compare each variant against the control; multiple-testing
   corrections apply across all variant-vs-control comparisons.
5. **The four headline numbers are outputs of simulations, never hardcoded:**
   - false-positive rate with peeking (~15% — must be *measured*, unknown exact value)
   - false-positive rate with framework (~5% controlled)
   - power 80%
   - time-to-decision ~35% faster
   Any doc claiming these numbers must cite `abtest/simulations.py` outputs.

## 3. Repo layout (these files are the contract — add to them, don't reinvent)

```
AB Testing Framework/
├── data/                        # real-world Udacity e-commerce A/B dataset (committed)
│   ├── ab_data.csv              #   user_id, timestamp, group, landing_page, converted
│   └── countries.csv            #   user_id, country (optional feature data)
├── abtest/                     # core library: pure, deterministic, testable
│   ├── __init__.py
│   ├── data_generator.py       # synthetic conversion data, seeded
│   ├── real_data.py            # load/clean real ab_data.csv, SRM check, chronological split
│   ├── frequentist.py          # two-proportion z-test, CI, lift
│   ├── power.py                # sample size, power curves
│   ├── multiple_testing.py     # Bonferroni + BH wrappers
│   ├── bayesian.py             # posterior, prob_best, expected_loss
│   ├── sequential.py           # Bayesian early-stopping rules
│   └── simulations.py          # peeking / power / multi-test / time-to-decision
├── dashboard/
│   └── app.py                  # Streamlit app, exactly 4 tabs, synthetic OR real data
├── tests/                      # pytest, known-value fixtures
├── docs/
│   ├── best_practices.md
│   └── executive_template.md
├── requirements.txt
├── README.md
└── AGENTS.md
```

## 4. Module contracts

### `abtest/data_generator.py`
- `generate_conversion_data(n_controls, n_variants, cr_control, cr_variants, seed=None) -> DataFrame`
  Columns: `user_id`, `group` (`control`/`V1`/`V2`/...), `converted` (0/1).
  Must be fully seeded/reproducible.

### `abtest/frequentist.py`
- `z_test_two_proportion(control_successes, control_total, variant_successes, variant_total, alternative="two-sided") -> ZTestResult`
  Thin wrapper over `statsmodels.stats.proportion`. Include `z`, `p_value`, and
  compute the exact test used (report method in `method` field).
- `confidence_interval_lift(control_successes, control_total, variant_successes, variant_total, alpha=0.05) -> dict`
  Returns rate estimates, absolute lift, relative lift, and CIs (rate CI per arm
  + CI on lift) as plain dict with `ci_low` / `ci_high`.

### `abtest/power.py`
- `min_sample_size(base_rate, mde, alpha=0.05, power=0.8, alternative="two-sided") -> int`
  Uses `statsmodels.stats.proportion.proportion_effectsize` +
  `NormalIndPower`. Must be validated against statsmodels' documented example.
- `power_curve(base_rate, mde, max_n, alpha=0.05, power_target=0.8) -> DataFrame`
  Columns: `n` (per arm), `power`. Returns rows for plotting.
- `expected_duration(n_per_arm, daily_traffic_per_arm) -> float` (days)

### `abtest/multiple_testing.py`
- `correct_pvalues(p_values, method="fdr_bh", alpha=0.05) -> dict`
  `method` in `{"bonferroni", "fdr_bh"}`. Wraps
  `statsmodels.stats.multitest.multipletests`. Returns `p_corrected` (array),
  `reject` (bool array), `alpha`, `method`.
- `decide_variants(z_results: list[dict], method="fdr_bh", alpha=0.05) -> DataFrame`
  One row per variant: raw p, corrected p, reject decision, lift, CI.
  Applied across variant-vs-control comparisons.

### `abtest/bayesian.py`
- `posterior_samples(control_successes, control_total, variant_successes=None, variant_total=None, n_samples=20000, engine="auto") -> dict`
  `engine` in `{"auto", "pymc", "closed_form"}`. `auto` = PyMC if importable,
  else closed-form Beta sampling. Returns array(s) of posterior draws.
- `prob_best(results: dict) -> dict` — probability each arm is the best, via
  Monte Carlo over posterior draws. Multi-arm capable.
- `expected_loss(results: dict, ...) -> dict` — expected loss per variant vs
  the best arm; include a helper converting loss + cost ratio into a decision
  (`choose_variant(...)`).

### `abtest/sequential.py`
- `sequential_result(successes, totals, stop_prob_best=0.95, max_obs=None, ...) -> dict`
  Track P(best) at each observation for the two-arm case; return
  `{stopped: bool, obs_at_stop, p_best_at_stop, trajectory: DataFrame}`.
  No frequentist sequential machinery.

### `abtest/simulations.py`  (each measurable, seeded, prints + returns numbers)
- `simulate_peeking(...) -> dict` — true-null tests, peek after every N users,
  stop when p<0.05; return measured false-positive rate.
- `simulate_power(...) -> dict` — measured power of the framework's sample-size
  recommendation at effect sizes.
- `simulate_multiple_testing(...) -> dict` — k true-null variants, how many
  false positives raw vs Bonferroni vs BH.
- `simulate_time_to_decision(...) -> dict` — fixed-sample vs Bayesian sequential,
  average observations to decision across many trials.
- All four return `dict`s with `{metric, value}`-style results that dashboards
  and README can cite. No hardcoded headline numbers.

### `abtest/real_data.py`
- `load_ab_data(path=None) -> DataFrame` — raw `data/ab_data.csv` with parsed
  timestamps; `FileNotFoundError` / `ValueError` on bad input.
- `load_countries(path=None) -> DataFrame` — `user_id`, `country`.
- `clean_ab_data(df) -> DataFrame` — drops users who saw both pages (multi-row
  users); one row per user; sorted by `timestamp` then `user_id` (arrival
  order for sequential replay).
- `srm_check(df, alpha=0.05) -> dict` — Sample Ratio Mismatch check on the
  control share; must contain CI, `passes`, `verdict`.
- `chronological_split(df, checkpoints=100) -> dict` — cumulative
  `successes`/`totals` arrays of shape `(2, checkpoints)` (row 0 = control,
  row 1 = treatment) for `sequential.sequential_result`.

### `dashboard/app.py`
- Streamlit sidebar controls: data source (synthetic or real `ab_data.csv`),
  daily traffic, base rate, MDE, alpha, power, variant count (multi-arm),
  successes/totals input or "generate from generator" toggle.
- **Exactly 4 tabs, fixed order and names:**
  1. `Planner` — sample size, power curve, expected duration
  2. `Frequentist` — z-tests, CIs, forest plot, Bonferroni + BH corrections
  3. `Bayesian` — posterior plots, P(best), expected loss, sequential chart
  4. `Simulations` — the four sims from `simulations.py`, interactive params
- Heavy compute behind `st.cache_data`.
- With the **real** data source: planner uses the *observed* control rate and
  observed lift; a collapsible data-quality block shows raw/clean row counts
  and the SRM verdict; the sequential monitor uses `pbest_method="mc"` when
  samples per arm exceed 20k (exact sum is too slow at that scale).

## 5. Build order & Definition of Done per milestone

### M0 — Scaffold
- `git init`, venv, `requirements.txt`, package skeleton, empty `tests/`.
- DoD: `python -c "import abtest"` works.

### M1 — Frequentist (Week 1)
- `data_generator.py`, `frequentist.py`, `power.py`, `multiple_testing.py`,
  peeking sim in `simulations.py`.
- DoD: sample size reproduces statsmodels' documented example; `correct_pvalues`
  matches documented `multipletests` outputs; `simulate_peeking` yields an
  inflated FP rate (roughly 10-30%, whatever it measures — record the number).

### M2 — Bayesian (Week 2)
- `bayesian.py` (PyMC + fallback), `sequential.py`, `simulate_time_to_decision`,
  `simulate_power`, `simulate_multiple_testing`.
- DoD: posterior sanity (P(best)→1 on huge samples); time-to-decision sim runs;
  `simulate_power` ≈ 0.8 when using `min_sample_size` defaults.

### M3 — Dashboard & docs (Week 3)
- `dashboard/app.py`, README, `docs/best_practices.md`,
  `docs/executive_template.md`.
- DoD: `streamlit run dashboard/app.py` works with all 4 tabs populated by real
  library calls (no placeholder numbers).

## 6. Conventions

- **Pure functions in `abtest/`:** no I/O, no plotting, no file writes inside
  library modules. Plotting lives in `dashboard/` / `simulations.py`.
- **Determinism:** every random operation takes a `seed`; fix `np.random.seed`
  inside generators and sims.
- **Tests are known-value fixtures, not smoke tests:** hand-computed z/p cases,
  statsmodels documented examples, `multipletests` documented outputs.
- **Dependencies:** add nothing to `requirements.txt` without a reason and a
  note; `pymc` stays last and optional. Never add `pymc3`.
- **Dashboard uses Plotly; Matplotlib allowed in simulations.**
- Keep public function signatures as specified above; rename only with user
  approval and update this file.

## 7. Pitfalls (recurring agent mistakes)

- **`pymc3` does not exist** — use `pymc` or the Beta fallback.
- **Don't hardcode the headline numbers** — they are simulation outputs.
- **Don't add scope** (continuous metrics, frequentist sequential, Bayesian
  hierarchical models, guardrail metrics) — the user explicitly cut these.
- **Don't "save time" by skipping tests** — DoD gates above require them.
- **Windows/PyMC:** a failed PyMC install must never break the dashboard; the
  `auto` engine + fallback guarantees this.

## 8. Definition of done (whole project)

1. `pytest` green.
2. `streamlit run dashboard/app.py` runs, all 4 tabs functional.
3. Every claim in README's features/expected-results section cites a number
   produced by `abtest/simulations.py`.