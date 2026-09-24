# A/B Testing Best Practices

A practical guide for running conversion A/B tests correctly — and for using
this framework to enforce it.

## 0. Should this test even run?

Before spending traffic and calendar weeks:

- **Is there a hypothesis?** "We believe changing X will move conversion by Y
  because of Z." No mechanism = no test.
- **Is the change worth it?** Use the planner: if the minimum detectable effect
  (MDE) requires more traffic/days than you can give, the test can't be powered
  properly. Either increase traffic, accept a larger MDE, or don't run it.
- **One primary metric.** Decide the single KPI the decision will be based on.
  Everything else is context, not a decision driver.

## 1. Power and sample size — do this first, not last

- Default to **80% power** and **α = 0.05** (two-sided).
- Pick the **MDE before the test starts**. The MDE is the smallest effect you
  care about *and* can plausibly believe. It is not "whatever the data shows."
- Sample size scales with the *square* of the inverse of the MDE: halving the
  MDE quadruples the required sample. Small lifts are expensive to detect.
- Compute the **expected duration** = sample size ÷ daily traffic. If the test
  would run longer than is operationally acceptable, revisit the MDE or traffic,
  — never just shorten the test and keep the same MDE.

## 2. Every analysis starts with a data-quality check (SRM)

Before any significance test, confirm the experiment was **actually
randomized** with a sample-ratio-mismatch (SRM) check: does the control share
match the intended 50/50 split, within its confidence interval? A failing SRM
(e.g. control share outside the CI) means the groups differ for reasons other
than the treatment — caches, device/browser artifacts, or a buggy assignment
— and no lift estimate can be trusted until it's fixed.

- Run it on the *cleaned* data (one row per experiment unit, e.g. per user).
- In this framework: `real_data.srm_check` on the real Udacity dataset
  returns control share 49.98% (95% CI [49.8%, 50.2%]) — **passes**.
- Also clean your data first: on the real dataset, 3,894 users saw *both*
  pages and are dropped before analysis (294,478 rows → 286,690).

## 3. The peeking trap

- Looking at results early and stopping when p < 0.05 **inflates the false
  positive rate** well above 5% — see `simulations.simulate_peeking` for the
  measured number with your peek frequency and sample size.
- **Rules that prevent it:**
  - Pre-register the sample size and the decision rule before observing data.
  - Decide *in advance* who can look, and write down the decision in the report.
  - If you must monitor, use the Bayesian sequential chart with a pre-set
    stopping threshold (P(best) ≥ threshold, or expected loss below a cost
    ratio) — see §6.

## 4. Multiple testing

- Any time you compare **more than one variant** (A/B/C/n, or multiple metrics),
  the probability of a false positive compounds.
- With k independent tests at α = 0.05, the chance of *at least one* false
  positive is `1 − 0.95^k` — already ~14% with just 3 tests.
- Use `multiple_testing.correct_pvalues`:
  - **Bonferroni** — conservative; controls family-wise error rate. Fine for a
    handful of hypotheses; overly strict with many.
  - **Benjamini-Hochberg (BH / FDR)** — controls the false discovery rate; more
    power, the usual default when comparing variants against a control.
- Decide on the correction *before* running, and apply it to the decisions,
  not just the reported p-values.

## 5. Frequentist vs. Bayesian — read both

- The **p-value** answers: "if there were truly no effect, how often would we
  see data this extreme?" It is *not* the probability the variant is better.
- The **posterior P(variant > control)** answers the question product teams
  actually ask: "how likely is B better than A, given the data and prior?"
- Run both in this framework. When they disagree, the reason is usually sample
  size or a strong prior — investigate, don't pick the answer you like.
- Default to a **weak/no prior** (Beta(1,1) = uniform) unless you have a real
  reason to encode prior knowledge.

## 6. Sequential testing / early stopping

- Fixed-sample tests are the gold standard for p-value validity — the cost is
  waiting the full planned duration even when the answer is obvious.
- With a **Bayesian sequential rule** you may stop early when the evidence is
  decisive (posterior P(best) above a threshold or expected loss below a cost
  ratio), because Bayesian updating is always valid — but the *decision rule*
  must be set in advance, or you're just peeking with extra steps.
- Treat the sequential chart as a monitoring tool: set the threshold, and if the
  test stops early, report *both* the pre-planned and realized sample sizes.

## 7. Reporting

Use `docs/executive_template.md`. A decision is only defensible if the report
shows: the pre-registered plan (MDE, sample size, correction), the realized
sample, both analyses, and whether any decisions were made before the end.

## 8. Common failure checklist

- [ ] Hypothesis stated before the test
- [ ] SRM / randomization check passed on cleaned data (§2)
- [ ] Sample size computed from a pre-chosen MDE (80% power, α = 0.05)
- [ ] Expected duration fits the operational window
- [ ] Multiple-testing correction chosen up front (BH for variant comparisons)
- [ ] Stopping rule fixed before observing data
- [ ] No decision made by "peeking"
- [ ] Both frequentist and Bayesian read side by side
- [ ] Executive template filled out with realized vs. planned numbers