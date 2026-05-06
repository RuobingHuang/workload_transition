# Paper Results Strategy

This document separates the analyses into:

```text
Main results: use in the paper's primary Results section.
Secondary results: use as interpretation or supplementary material.
Not recommended as main results: keep for diagnostics/exploration only.
```

The goal is to make the paper focused rather than presenting every analysis.

## Core Message

The paper should not argue that every analysis is equally important. The clearest story is:

```text
1. Workload transitions impair overall performance.
2. Transition magnitude mainly drives immediate disruption.
3. Transition frequency shapes recovery and residual error.
4. High-intensity transition exposure reduces resilience to a later disturbance.
5. Under high intensity, the frequency pattern shows a trend-level U shape.
```

This supports the paper's title:

```text
Disruption or Arousal?
```

Suggested interpretation:

```text
High-magnitude transitions behave like strong disruptions that immediately degrade performance.
Frequency does not simply make performance worse linearly; instead, it appears to shape adaptation, recovery, and later resilience.
```

## Main Result 1: Transitions Impair Overall Performance

Use this analysis:

```text
visualize_transition_worse_than_base.py
trial_metrics.csv
transition_vs_base_visual_summary.csv
figures/transition_worse_than_base.png
```

Research question:

```text
RQ1: Do workload transition conditions lead to distinct levels of human performance?
```

What it studies:

```text
This analysis compares the six transition conditions against BASE at the whole-trial level.
It asks whether having workload transitions at all generally worsens performance.
```

Metrics:

```text
mean_dev
std_dev
rmse_dev
auc_dev
```

Main finding:

```text
All 24/24 transition-condition comparisons are worse than BASE across the four primary deviation metrics.
```

How to write it:

```text
Across all primary deviation metrics, transition conditions consistently impaired performance relative to BASE.
```

Role in paper:

```text
Use this as the opening performance result.
It establishes that workload transitions act as performance-disrupting events.
Do not spend too much space here; it is a baseline demonstration.
```

Recommended figure:

```text
figures/transition_worse_than_base.png
```

## Main Result 2: Magnitude Drives Immediate Disruption, Frequency Shapes Recovery

Use this analysis:

```text
run_02b_rq1_from_events.py
event_metrics_regular.csv
rq1_trial_aggregates_from_events_2_3.csv
run_02b_rq1_from_events_results2
```

Research questions:

```text
RQ1: Do workload transition magnitude and frequency shape event-level performance?
RQ2: How do transition conditions affect immediate disruption and short-term recovery?
```

What it studies:

```text
This analysis focuses on regular transition events.
Instead of averaging across the whole trial, it asks what happens immediately after each workload transition.
It aggregates transition-event responses to participant x condition, then runs a 2 intensity x 3 frequency repeated-measures ANOVA.
```

Important metrics:

```text
imm_mean:
Immediate post-transition deviation. Higher means stronger immediate disruption.

peak_mean:
Peak deviation in the early post-transition window. Higher means a larger transition shock.

trec_med:
Median recovery time. Higher means slower recovery.

recover_success_rate:
Proportion of events that successfully recovered. Higher means better recovery.

delta_mean:
Post-pre change around the transition. Higher positive values mean larger deterioration.

resid_5_10s_mean:
Residual error in the 5-10 s post-transition window. Higher means more lingering error.
```

Main findings:

```text
imm_mean:
intensity F = 51.22, p < 0.001

peak_mean:
intensity F = 53.24, p < 0.001
intensity x frequency p = 0.066, trend-level

trec_med:
intensity F = 14.34, p = 0.0020

recover_success_rate:
frequency F = 10.52, p = 0.0004
intensity F = 44.88, p < 0.001

delta_mean:
frequency F = 11.76, p = 0.0002
intensity F = 17.80, p = 0.0009

resid_5_10s_mean:
frequency F = 12.38, p = 0.0001
```

Interpretation:

```text
Transition magnitude/intensity mainly controls the strength of immediate disruption.
High-intensity transitions produce larger immediate deviation, larger peak responses, and slower recovery.

Transition frequency is more strongly related to recovery success and residual error.
This suggests that frequency shapes how participants adapt or stabilize after repeated workload changes.
```

How to write it:

```text
Event-level analyses showed that transition magnitude strongly shaped immediate disruption and peak response, whereas transition frequency was more strongly associated with recovery success and residual error.
```

Role in paper:

```text
Use this as a main result.
This result is important because it separates immediate disruption from recovery dynamics.
```

Recommended figure:

```text
Need a clean event-level figure if this becomes a central paper result.
Recommended panels:
1. imm_mean by intensity x frequency
2. peak_mean by intensity x frequency
3. recover_success_rate by intensity x frequency
4. resid_5_10s_mean by intensity x frequency
```

## Main Result 3: High-Intensity Transitions Reduce Next-Disturbance Resilience

Use this analysis:

```text
run_07_next_disturbance_resilience_fixed.py
visualize_next_disturbance_resilience.py
next_disturbance_response_fixed.csv
next_disturbance_rmanova_fixed.csv
next_disturbance_pairwise_fixed.csv
figures/next_disturbance_resilience_publication.pdf
```

Research questions:

```text
RQ1: Do transition magnitude and frequency jointly affect performance?
RQ2: Do transition conditions have delayed effects on subsequent performance?
```

What it studies:

```text
This analysis asks whether prior transition exposure affects response to a later disturbance.
It is a resilience analysis rather than a simple post-transition tail average.
```

Why it matters:

```text
This is one of the most theoretically useful results.
It connects the ecological disturbance idea to human performance:
transitions do not only impair immediate performance; they also change later recovery capacity.
```

Metrics:

```text
post_rmse:
RMSE after the target disturbance. Higher means worse response.

delta_rmse:
post_rmse - pre_rmse. Higher means larger disturbance-induced deterioration.

peak_deviation:
Maximum deviation after the target disturbance. Higher means stronger peak disruption.

post_auc:
Accumulated post-disturbance error. Higher means larger total error burden.
```

Main findings:

```text
post_rmse:
intensity p = 0.00037

delta_rmse:
intensity p = 0.00203
intensity x frequency p = 0.0349

peak_deviation:
intensity p = 0.000069

post_auc:
intensity p = 0.00041
```

Pairwise pattern:

```text
Within high intensity, HI_LF is worse than HI_MF for:
post_rmse
delta_rmse
peak_deviation
```

Interpretation:

```text
High-intensity transitions significantly reduce resilience to subsequent disturbance.
The significant delta_rmse intensity x frequency interaction means the effect of frequency depends on transition magnitude.
The high-intensity low-frequency condition appears especially disruptive.
```

How to write it:

```text
High-intensity transitions significantly increased response magnitude to the subsequent disturbance. Delta RMSE further showed a significant intensity-by-frequency interaction, indicating that frequency modulated the resilience cost of high-intensity transitions.
```

Role in paper:

```text
Use this as a main result.
This may be the most novel contribution because it shows delayed resilience impairment.
```

Recommended figure:

```text
figures/next_disturbance_resilience_publication.pdf
```

## Secondary Result: Trend-Level U-Shaped Frequency Pattern

Use this analysis:

```text
visualize_next_disturbance_u_shape.py
next_disturbance_u_shape_summary.csv
figures/next_disturbance_u_shape.pdf
```

What it studies:

```text
This analysis visualizes whether the frequency pattern is U-shaped under high-intensity transitions.
```

U-shape contrast:

```text
U = (LF + HF) / 2 - MF
```

Main findings:

```text
delta_rmse, HI:
U = 4.81, p = 0.074

peak_deviation, HI:
U = 6.02, p = 0.084
```

Interpretation:

```text
The pattern is visually U-shaped under high intensity:
LF is worst, MF is better, and HF rebounds.

However, the U-shape contrast is trend-level, not conventionally significant.
```

How to write it:

```text
Under high-intensity transitions, next-disturbance responses showed a trend-level U-shaped frequency pattern, with the lowest response under medium frequency and higher responses under low and high frequency.
```

Role in paper:

```text
Use this as an interpretation of the significant delta_rmse intensity x frequency interaction.
Do not make it a standalone significant claim.
This can be a subpanel, supplementary figure, or short paragraph in the next-disturbance section.
```

Avoid writing:

```text
Frequency had a significant U-shaped effect.
```

## Supplementary Only: 45 s Tail Analysis

Use this analysis only if needed:

```text
diagnose_tail_45s_analysis.py
run_04_tail_analysis.py
figures/tail_45s_diagnostic.png
```

What it studies:

```text
This asks whether performance remains worse during the final 45 s post-transition tail.
```

Main findings:

```text
tail_mean_dev:
one-sided p = 0.054

tail_rmse_dev:
one-sided p = 0.068

tail_std_dev:
one-sided p = 0.163

time_to_stable:
one-sided p = 0.149
```

Interpretation:

```text
The 45 s tail shows weak residual impairment trends, but no robust significant effect.
```

Role in paper:

```text
Do not use this as a main result.
Use it only as supplementary evidence or as an explanation that full-window tail averages may dilute short-lived recovery effects.
```

Suggested wording:

```text
The full 45 s post-transition tail window showed only weak residual impairment trends, suggesting that delayed effects may be transient or highly variable across participants.
```

## Exploratory Only: Tail Phase Segmentation

Use this analysis only for exploration:

```text
run_06_tail_mechanism_analysis.py
tail_phase_segmentation.csv
tail_phase_rmanova.csv
tail_phase_pairwise.csv
```

What it studies:

```text
This script divides the final 45 s tail into damage, recovery, and steady phases using participant-specific thresholds.
```

Why not use as a main result:

```text
1. Current statistical results are weak.
2. Phase definitions depend on threshold choices.
3. The method is harder to explain and easier to challenge.
4. The comments and actual constants differ:
   comments mention 2 s / 4 s sustained windows,
   but the code uses 5 s / 5 s.
```

Role in paper:

```text
Do not use as a main result.
Use only as exploratory or appendix material if needed.
```

## Recommended Paper Results Structure

### Result 1

Title:

```text
Workload transitions impair overall performance relative to baseline
```

Use:

```text
visualize_transition_worse_than_base.py
```

Purpose:

```text
Establish that transition conditions are generally performance-disrupting.
```

### Result 2

Title:

```text
Transition magnitude drives immediate disruption, while frequency shapes recovery
```

Use:

```text
run_02b_rq1_from_events.py
```

Purpose:

```text
Show that intensity and frequency affect different phases of transition response.
```

### Result 3

Title:

```text
High-intensity transitions reduce resilience to subsequent disturbance
```

Use:

```text
run_07_next_disturbance_resilience_fixed.py
visualize_next_disturbance_resilience.py
```

Purpose:

```text
Show delayed effects and reduced resilience.
```

### Optional Subsection

Title:

```text
High-intensity transitions show a trend-level U-shaped frequency pattern
```

Use:

```text
visualize_next_disturbance_u_shape.py
```

Purpose:

```text
Explain the significant interaction in delta_rmse.
```

## Final Recommendation

Use these as the paper's core analyses:

```text
1. visualize_transition_worse_than_base.py
2. run_02b_rq1_from_events.py
3. run_07_next_disturbance_resilience_fixed.py
4. visualize_next_disturbance_resilience.py
```

Use this as secondary interpretation:

```text
visualize_next_disturbance_u_shape.py
```

Do not main-text these unless needed:

```text
run_04_tail_analysis.py
diagnose_tail_45s_analysis.py
run_06_tail_mechanism_analysis.py
```

## Statistical Writing Notes

Use:

```text
p < 0.05 = significant
0.05 <= p < 0.10 = trend-level / marginal
```

Safe claims:

```text
Transitions impaired overall performance.
Transition magnitude strongly affected immediate disruption.
Transition frequency affected recovery success and residual error.
High-intensity transitions reduced next-disturbance resilience.
Delta RMSE showed a significant intensity x frequency interaction.
HI conditions showed a trend-level U-shaped frequency pattern.
```

Avoid:

```text
The U-shape was significant.
The 45 s tail showed strong recovery impairment.
Tail phase segmentation proved a mechanism.
```
