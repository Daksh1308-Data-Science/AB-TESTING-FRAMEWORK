# Experiment Executive Summary

> Copy this template for each experiment. Replace `[bracketed]` fields.
> A decision is only defensible if the *pre-registered* plan is shown alongside
> what actually happened.

---

## 1. Summary

| Field | Value |
|---|---|
| Experiment | `[name/ID]` |
| Hypothesis | `[We believe <change> moves conversion by <MDE> because <mechanism>]` |
| Primary metric | conversion rate (`[definition, e.g., paid signups / sessions]`) |
| Status | `[Decision made / Still running / Stopped]` |
| Decision | `[Ship variant X / Keep control / More data needed]` |

## 2. Pre-registered plan (set before the test started)

| Parameter | Value |
|---|---|
| Control conversion rate (assumed) | `[e.g., 3.0%]` |
| Minimum detectable effect (MDE) | `[e.g., +0.5pp absolute]` |
| Significance level α | 0.05 |
| Power | 0.80 |
| Sample size per arm | `[from the planner]` |
| Expected duration | `[sample size ÷ daily traffic]` |
| Multiple-testing correction | `[none / Bonferroni / BH]` |
| Stopping rule | `[fixed sample / Bayesian threshold, spec.]` |

## 3. What actually happened

| Metric | Planned | Realized |
|---|---|---|
| Sample size per arm | `[..]` | `[..]` |
| Duration (days) | `[..]` | `[..]` |
| Decisions before end of test | none | `[none / list]` |

## 4. Results — Frequentist

| Variant | n | Conversions | Rate | Lift (rel.) | 95% CI on lift | Raw p | Corrected p | Reject? |
|---|---|---|---|---|---|---|---|---|
| Control | | | | — | — | — | — | — |
| `[V1]` | | | | | | | | |

Correction applied: `[Bonferroni / BH]`

## 5. Results — Bayesian

| Variant | Posterior mean rate | P(variant best) | Expected loss vs best |
|---|---|---|---|
| Control | | | |
| `[V1]` | | | |

Prior: `[Beta(1,1) / other]`

## 6. Interpretation

- `[1–3 sentences: do frequentist and Bayesian agree? What is driving the
  result? Any caveats — early stop, small n, multiple variants?]`

## 7. Risks & limitations

- `[e.g., stopped early vs planned N; correction may be conservative; metric
  influenced by seasonality; novelty effect]`

## 8. Recommendation

- `[Ship / don't ship / rerun with larger N / follow-up experiment]`