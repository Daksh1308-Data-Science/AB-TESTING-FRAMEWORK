"""Streamlit dashboard for the A/B testing framework.

Run from the project root:  streamlit run dashboard/app.py

Exactly four tabs, in order: Planner, Frequentist, Bayesian, Simulations.
Data sources: synthetic (seeded) or the real Udacity e-commerce A/B dataset
(data/ab_data.csv).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import beta as beta_dist

from abtest.data_generator import generate_conversion_data
from abtest.real_data import load_ab_data, clean_ab_data, srm_check, chronological_split
from abtest.frequentist import variant_result
from abtest.power import min_sample_size, power_curve, expected_duration
from abtest.multiple_testing import decide_variants
from abtest.bayesian import posterior_samples, prob_best, expected_loss, choose_variant
from abtest.sequential import sequential_result
from abtest.simulations import (
    simulate_peeking,
    simulate_power,
    simulate_multiple_testing,
    simulate_time_to_decision,
)

st.set_page_config(page_title="A/B Testing Framework", layout="wide")
st.title("A/B Testing Framework")
st.caption(
    "Frequentist + Bayesian experimentation with power analysis, multiple-testing "
    "correction, Bayesian sequential monitoring, and simulation-measured results."
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Experiment parameters")
    data_source = st.radio(
        "Data source",
        ["Synthetic (seeded)", "Real: Udacity ab_data.csv"],
        horizontal=True,
    )
    is_real = data_source.startswith("Real")

    base_rate = st.number_input(
        "Control conversion rate (planning)", 0.001, 0.999, 0.05, step=0.001, format="%.3f"
    )
    mde = st.number_input(
        "Minimum detectable effect (absolute lift)",
        0.0001, 0.5, 0.01, step=0.001, format="%.3f",
    )
    alpha = st.number_input("Significance level α", 0.001, 0.5, 0.05, step=0.005, format="%.3f")
    power_target = st.number_input("Power target", 0.5, 0.99, 0.80, step=0.05, format="%.2f")
    daily_traffic = st.number_input("Daily traffic per arm", 10, 1_000_000, 1000, step=100)
    if is_real:
        st.caption("Real experiment: two arms (control vs. treatment), univariate.")
        use_synth = False
        n_variants = 1
    else:
        use_synth = st.checkbox("Use synthetic data (seeded)", value=True)
        n_variants = st.number_input("Number of variants (A/B/.../n)", 1, 10, 1, step=1)
    correction = st.selectbox("Multiple-testing correction", ["fdr_bh", "bonferroni"])
    seed = st.number_input("Seed", 0, 2**31 - 1, 42, step=1)


# ---------------- Cached heavy compute ----------------
@st.cache_data(show_spinner="Generating synthetic data...")
def build_data(n_controls, n_variants, cr_control, cr_variants, seed):
    return generate_conversion_data(
        int(n_controls), int(n_variants), cr_control, list(cr_variants), seed=int(seed)
    )


@st.cache_data(show_spinner="Loading real data...")
def load_real():
    raw = load_ab_data()
    clean = clean_ab_data(raw)
    stats = {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "duplicate_users_dropped": int(raw["user_id"].nunique() - clean["user_id"].nunique()),
    }
    return clean, stats


@st.cache_data(show_spinner="Sampling posteriors...")
def run_posteriors(x_c, n_c, x_vs, n_vs, n_samples, engine, seed):
    return posterior_samples(x_c, n_c, x_vs, n_vs, n_samples=n_samples, engine=engine, seed=seed)


@st.cache_data(show_spinner="Running sequential monitor...")
def run_seq(successes, totals, method, samples, seed):
    return sequential_result(
        successes, totals, stop_prob_best=0.95, pbest_method=method, n_pbest_samples=samples, seed=seed
    )


@st.cache_data(show_spinner="Simulating peeking...")
def run_peek(cr, n, peek, a, trials, s):
    return simulate_peeking(cr=cr, n_per_arm=n, peek_every=peek, alpha=a, n_trials=trials, seed=s)


@st.cache_data(show_spinner="Measuring power...")
def run_pow(br, mde, a, pt, trials, s):
    return simulate_power(base_rate=br, mde=mde, alpha=a, power_target=pt, n_trials=trials, seed=s)


@st.cache_data(show_spinner="Simulating multiple testing...")
def run_multi(k, n, cr, a, trials, s):
    return simulate_multiple_testing(k=k, n_per_arm=n, cr=cr, alpha=a, n_trials=trials, seed=s)


@st.cache_data(show_spinner="Simulating time to decision...")
def run_timed(br, mde, a, pt, trials, s):
    return simulate_time_to_decision(
        base_rate=br, mde=mde, alpha=a, power_target=pt, n_trials=trials, seed=s, max_obs=5000
    )


# ---------------- Data resolution ----------------
if is_real:
    df, real_stats = load_real()
    variant_labels = ["treatment"]
elif use_synth:
    # Synthetic: variants are seeded to convert at exactly base_rate + mde.
    variant_rates = [min(0.999, base_rate + mde)] * int(n_variants)
    # Synthetic data is generated at the full planned sample size.
    synth_n = int(min_sample_size(base_rate, mde, alpha=alpha, power=power_target))
    df = build_data(synth_n, int(n_variants), base_rate, variant_rates, seed)
    variant_labels = [f"V{i + 1}" for i in range(int(n_variants))]
else:
    df = None
    variant_labels = [f"V{i + 1}" for i in range(int(n_variants))]


def counts_for(group):
    sub = df[df.group == group]
    return int(sub.converted.sum()), int(len(sub))


if df is not None:
    ctrl_x, ctrl_n = counts_for("control")
    variant_counts = [counts_for(label) for label in variant_labels]
else:
    ctrl_x, ctrl_n = 0, 0
    variant_counts = [(0, 0)] * len(variant_labels)

# Effective experiment parameters: for real data, plan around the observed
# control rate and the observed lift; for synthetic/manual, use sidebar inputs.
if is_real and df is not None:
    obs_control_rate = ctrl_x / ctrl_n
    obs_variant_rate = variant_counts[0][0] / variant_counts[0][1] if variant_counts[0][1] else 0.0
    base_eff = obs_control_rate
    mde_eff = abs(obs_variant_rate - obs_control_rate) or 0.0005
else:
    base_eff = base_rate
    mde_eff = mde

n_arm_plan = min_sample_size(base_eff, max(mde_eff, 1e-4), alpha=alpha, power=power_target)

# ---------------- Tabs ----------------
tab_planner, tab_freq, tab_bayes, tab_sims = st.tabs(
    ["Planner", "Frequentist", "Bayesian", "Simulations"]
)

with tab_planner:
    st.subheader("Experiment planner")
    c1, c2, c3 = st.columns(3)
    c1.metric("Required sample size per arm", f"{n_arm_plan:,}")
    c2.metric("Expected duration (days)", f"{expected_duration(n_arm_plan, daily_traffic):.1f}")
    c3.metric("Daily traffic per arm", f"{daily_traffic:,}")
    st.caption(
        f"Power analysis via statsmodels normal approximation "
        f"(α={alpha:.3f}, power={power_target:.2f}, base={base_eff:.4f}, MDE={mde_eff:+.4f}). "
        "Duration = sample size ÷ daily traffic."
    )
    pc = power_curve(base_eff, max(mde_eff, 1e-4), max_n=max(n_arm_plan, 2))
    fig = px.line(pc, x="n", y="power", title="Power vs. sample size per arm")
    fig.add_hline(y=power_target, line_dash="dash", annotation_text="target power")
    fig.add_vline(x=n_arm_plan, line_dash="dot", annotation_text="required n")
    st.plotly_chart(fig, width="stretch")

with tab_freq:
    st.subheader("Frequentist analysis (two-proportion z-test)")

    if is_real:
        with st.expander("Data quality & sanity check (real data)", expanded=True):
            d1, d2, d3 = st.columns(3)
            d1.metric("Raw rows", f"{real_stats['raw_rows']:,}")
            d2.metric("Clean rows (one per user)", f"{real_stats['clean_rows']:,}")
            d3.metric("Dropped (saw both pages)", f"{real_stats['duplicate_users_dropped']:,}")
            srm = srm_check(df)
            sc1, sc2 = st.columns(2)
            sc1.metric(
                "Control share",
                f"{srm['control_share']:.2%}",
                help=f"95% CI [{srm['ci_low']:.2%}, {srm['ci_high']:.2%}]",
            )
            sc2.metric("Sample-ratio check", srm["verdict"])
        st.caption(
            f"Observed: control {ctrl_x:,}/{ctrl_n:,} = {ctrl_x/ctrl_n:.2%}"
            f" · treatment {variant_counts[0][0]:,}/{variant_counts[0][1]:,} = "
            f"{variant_counts[0][0]/variant_counts[0][1]:.2%}"
        )

    if df is None:
        st.caption("Enter observed successes and totals per arm.")
        ctrl_x = st.number_input("Control successes", 0, 1_000_000, 500, step=10)
        ctrl_n = st.number_input("Control total", 1, 1_000_000_000, 10_000, step=100)
        variant_counts = []
        for i in range(int(n_variants)):
            subs = st.number_input(
                f"{variant_labels[i]} successes", 0, 1_000_000, 550, step=10, key=f"s{i}"
            )
            tot = st.number_input(
                f"{variant_labels[i]} total", 1, 1_000_000_000, 10_000, step=100, key=f"t{i}"
            )
            variant_counts.append((subs, tot))

    results = [
        variant_result(ctrl_x, int(ctrl_n), int(x_v), int(n_v), variant=label)
        for label, (x_v, n_v) in zip(variant_labels, variant_counts)
    ]
    decisions = decide_variants(results, method=correction, alpha=alpha)
    st.dataframe(decisions, width="stretch")
    st.caption(
        f"Correction: **{correction}** at α={alpha:.3f}. "
        "Green markers are variants that survive correction."
    )

    fig_f = go.Figure()
    for _, row in decisions.iterrows():
        color = "#2e7d32" if row["reject"] else "#9e9e9e"
        lo, hi = row["ci_low"], row["ci_high"]
        fig_f.add_trace(
            go.Scatter(
                x=[row["absolute_lift"]],
                y=[row["variant"]],
                mode="markers",
                marker=dict(color=color, size=12),
                error_x=dict(
                    type="data", symmetric=False,
                    array=[hi - row["absolute_lift"]],
                    arrayminus=[row["absolute_lift"] - lo],
                ),
                name=row["variant"],
            )
        )
    fig_f.add_vline(x=0, line_dash="dash")
    fig_f.update_layout(
        title="Absolute lift vs. control (95% CI)",
        xaxis_title="absolute lift",
        yaxis_title="",
    )
    st.plotly_chart(fig_f, width="stretch")

with tab_bayes:
    st.subheader("Bayesian analysis (Beta-Bernoulli)")
    c1, c2 = st.columns([1, 2])
    engine = c1.selectbox("Posterior engine", ["auto", "closed_form", "pymc"])
    n_posts = c1.slider("Posterior samples", 1000, 100_000, 20_000, step=1000)

    try:
        posteriors = run_posteriors(
            ctrl_x, ctrl_n,
            [x_v for x_v, _ in variant_counts],
            [n_v for _, n_v in variant_counts],
            n_posts,
            engine,
            seed,
        )
        pb = prob_best(posteriors)
        loss = expected_loss(posteriors)
        decision = choose_variant(loss)

        st.caption(f"Engine used: **{posteriors['engine']}** — {posteriors['note']}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Recommended decision", decision["chosen"])
        m2.metric("P(best) of choice", f"{pb.get(decision['chosen'], 0):.1%}")
        m3.metric("Expected loss of choice", f"{loss.get(decision['chosen'], 0):.4f}")

        fig1 = px.bar(
            x=list(pb.keys()), y=list(pb.values()),
            labels={"x": "arm", "y": "P(best)"},
            title="Probability of being the best arm",
        )
        st.plotly_chart(fig1, width="stretch")

        fig2 = px.bar(
            x=list(loss.keys()), y=list(loss.values()),
            labels={"x": "arm", "y": "expected loss (rate units)"},
            title="Expected loss (regret) of choosing each arm",
        )
        st.plotly_chart(fig2, width="stretch")

        observed_rates = [ctrl_x / ctrl_n] + [x / n for x, n in variant_counts if n > 0]
        upper = max(observed_rates) * 3.0 if observed_rates else base_eff + mde_eff
        xgrid = np.linspace(0.0, min(1.0, upper), 400)
        fig3 = go.Figure()
        fig3.add_trace(
            go.Scatter(
                x=xgrid, y=beta_dist.pdf(xgrid, ctrl_x + 1, ctrl_n - ctrl_x + 1),
                name="control", mode="lines",
            )
        )
        for label, (x_v, n_v) in zip(variant_labels, variant_counts):
            fig3.add_trace(
                go.Scatter(
                    x=xgrid, y=beta_dist.pdf(xgrid, x_v + 1, n_v - x_v + 1),
                    name=label, mode="lines",
                )
            )
        fig3.update_layout(
            title="Posterior densities (exact Beta posterior, Beta(1,1) prior)",
            xaxis_title="conversion rate", yaxis_title="density",
        )
        st.plotly_chart(fig3, width="stretch")

        if len(variant_labels) == 1 and df is not None:
            st.subheader("Sequential monitoring (two-arm, replayed in arrival order)")
            if "timestamp" in df.columns:
                # Real data: replay checkpoints in true arrival order.
                split = chronological_split(df, checkpoints=50)
                successes, totals = split["successes"], split["totals"]
            else:
                # Synthetic data: checkpoint by equal per-arm counts.
                vgroup = variant_labels[0]
                step_every = max(1, int(ctrl_n // 50))
                idx = np.arange(step_every - 1, int(ctrl_n), step_every)
                ctrl = df[df.group == "control"]
                var = df[df.group == vgroup]
                successes = np.vstack(
                    [ctrl.converted.cumsum().to_numpy()[idx], var.converted.cumsum().to_numpy()[idx]]
                )
                totals = np.vstack([idx + 1, idx + 1])
            method = "mc" if ctrl_n > 20_000 else "exact"
            seq = run_seq(successes, totals, method, 4000, int(seed))
            traj = seq["trajectory"]
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=traj["n_per_arm"], y=traj["p_control_best"], name="P(control best)"))
            fig4.add_trace(go.Scatter(x=traj["n_per_arm"], y=traj["p_variant_best"], name="P(variant best)"))
            fig4.add_hline(y=0.95, line_dash="dash", annotation_text="stop threshold")
            fig4.update_layout(
                title="P(best) over time (Bayesian sequential)",
                xaxis_title="users per arm", yaxis_title="probability",
            )
            st.plotly_chart(fig4, width="stretch")
            if seq["stopped"]:
                st.success(
                    f"Stopped at {seq['obs_at_stop']:,} users per arm — decision: "
                    f"**{seq['stopped_decision']}** (P(best) = {seq['p_best_at_stop']:.3f})."
                )
            else:
                st.info(
                    f"Not stopped within {int(ctrl_n):,} users per arm — the planned horizon "
                    f"(or the real test itself) is the decision boundary."
                )
        elif len(variant_labels) > 1:
            st.caption("Sequential monitoring supports two-arm tests (set variants to 1).")
        else:
            st.caption("Sequential monitoring needs a data source (synthetic or real).")
    except ImportError as exc:
        st.error(str(exc))

with tab_sims:
    st.subheader("Simulations — measured properties of the framework")
    st.caption(
        "Every headline number this project claims is measured here, seeded and "
        "reproducible. Simulation sample sizes are capped for dashboard runtime."
    )
    sc1, sc2, sc3 = st.columns(3)
    n_trials = sc1.number_input("Simulation trials", 10, 1000, 250, step=10)
    sim_seed = sc2.number_input("Simulation seed", 0, 2**31 - 1, 42, step=1)
    n_sim = int(min(n_arm_plan, 20_000))

    st.markdown("##### 1. Why peeking is dangerous")
    peek_every = max(50, n_sim // 40)
    peek = run_peek(base_eff, n_sim, peek_every, alpha, n_trials, sim_seed)
    fp_df = pd.DataFrame(
        {
            "scenario": ["fixed design", "daily peeking"],
            "false positive rate": [
                peek["fixed_false_positive_rate"],
                peek["peeking_false_positive_rate"],
            ],
        }
    )
    st.plotly_chart(
        px.bar(
            fp_df, x="scenario", y="false positive rate", color="scenario",
            title="Measured false-positive rate under the null",
        ),
        width="stretch",
    )
    st.caption(
        f"n_trials={peek['n_trials']}, {peek['n_per_arm']:,} users per arm, peek every "
        f"{peek['peek_every']:,}, α={alpha:.3f}. Fixed design ≈ α; peeking inflates it."
    )

    st.markdown("##### 2. Measured power of the recommended sample size")
    pow_res = run_pow(base_eff, max(mde_eff, 1e-4), alpha, power_target, min(n_trials, 300), sim_seed)
    pow_df = pd.DataFrame(
        {
            "scenario": ["power target", "measured power"],
            "power": [pow_res["power_target"], pow_res["measured_power"]],
        }
    )
    st.plotly_chart(
        px.bar(pow_df, x="scenario", y="power", color="scenario", title="Planned vs. measured power"),
        width="stretch",
    )
    st.caption(
        f"Recommended n per arm: {pow_res['recommended_n_per_arm']:,} "
        f"(α={pow_res['alpha']:.3f}, MDE={pow_res['mde']:+.3f}). Measured power = "
        f"{pow_res['measured_power']:.1%} over {pow_res['n_trials']} trials."
    )

    st.markdown("##### 3. Multiple-testing correction works")
    multi = run_multi(
        int(n_variants), min(n_arm_plan, 2000), base_eff, alpha, min(n_trials, 150), sim_seed
    )
    multi_df = pd.DataFrame(
        {
            "scenario": ["raw", "Bonferroni", "BH"],
            "any false positive": [
                multi["raw_any_false_positive_rate"],
                multi["bonferroni_any_false_positive_rate"],
                multi["fdr_bh_any_false_positive_rate"],
            ],
        }
    )
    st.plotly_chart(
        px.bar(
            multi_df, x="scenario", y="any false positive", color="scenario",
            title=f"P(at least one false positive) across {multi['k']} true-null variants",
        ),
        width="stretch",
    )
    st.caption(
        f"1 − 0.95^{multi['k']} ≈ {1 - 0.95**multi['k']:.0%} expected with no correction; "
        "corrections bring it back toward α."
    )

    st.markdown("##### 4. Time to decision: fixed-sample vs. Bayesian sequential")
    timed = run_timed(base_eff, max(mde_eff, 1e-4), alpha, power_target, min(n_trials, 120), sim_seed)
    time_df = pd.DataFrame(
        {
            "scenario": ["fixed sample", "Bayesian sequential"],
            "users per arm to decision": [
                timed["fixed_observations_to_decision"],
                timed["sequential_observations_to_decision"],
            ],
        }
    )
    st.plotly_chart(
        px.bar(
            time_df, x="scenario", y="users per arm to decision", color="scenario",
            title="Time to decision under a real effect",
        ),
        width="stretch",
    )
    st.caption(
        f"{timed['pct_faster']:.0f}% faster on average "
        f"({timed['stopped_early_fraction']:.0%} of trials stopped early at "
        f"P(best) ≥ {timed['stop_prob_best']:.2f})."
    )