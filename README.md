# RL Evaluation, Training & Robustness Harness

A full pipeline for training a locomotion policy, evaluating it with proper
statistics, and testing how it holds up outside the exact conditions it was
trained on. Built to demonstrate not just that a policy can be trained, but
that its results can be trusted, questioned, and verified.

## What this project demonstrates

- Multi-seed statistical evaluation (Welch's t-test, Cohen's d effect size)
  to distinguish real differences from seed noise
- Full-scale PPO training (10M steps) using tuned hyperparameters from the
  official Stable-Baselines3 Zoo, with checkpointing throughout
- Robustness testing under physics conditions the policy never trained on
- A verified case of a later checkpoint performing worse than an earlier
  one, caught by comparing checkpoints rather than trusting the final model
- A working anomaly-detection script that automatically flags the exact
  training instability that coincided with that regression

## Project structure
```
train.py                     short training runs (used for seed comparisons)
evaluate.py                  loads a policy, computes independent metrics
evaluate_normalized.py       evaluation with per-checkpoint VecNormalize stats
metrics.py                   Welch's t-test + Cohen's d between configs
run_experiment.py            orchestrates multi-seed training + evaluation
train_full.py                full 10M-step training run with checkpointing
record_video.py              records an evaluation episode as mp4
robustness_test.py           tests a policy under varied friction/mass
check_leg_symmetry.py        logs per-joint range of motion
check_training_health.py     flags instability in a training log
models_full/                 20 checkpoints saved every 500k steps
results/                     evaluation output as JSON
videos/                      recorded episodes, including the friction test
```

## Part 1: Multi-seed statistical evaluation

RL training is highly sensitive to random seed. To demonstrate this
concretely, two configurations (baseline learning rate vs. 3x higher) were
each trained across 5 seeds and compared using independent metrics, not
just training reward.

| Metric | Baseline mean | High LR mean | p-value | Effect size (d) | Verdict |
|---|---|---|---|---|---|
| Mean return | 219.8 | 111.9 | 0.088 | 1.23 (large) | Not significant. Large effect, but 5 seeds isn't enough to confirm it |
| Mean energy cost | 0.89 | 1.58 | 0.021 | -2.17 (large) | **Significant.** High LR genuinely uses more energy |

Within a single configuration, seed-to-seed variance alone produced returns
ranging from 96.4 to 325.8, a 3.4x spread from identical code. This is the
concrete argument for why single-run comparisons in RL are unreliable.

## Part 2: Full training run and checkpoint comparison

A full 10M-step PPO run was trained on `Ant-v5`, using the tuned Zoo
hyperparameters and observation/reward normalization (`VecNormalize`).

**A checkpoint at 7M steps outperforms the final 10M-step checkpoint.**

Evaluated at **200 episodes per checkpoint**, each loaded with the
`VecNormalize` statistics saved at that same checkpoint:

| Metric | 7M steps | 10M steps (final) |
|---|---|---|
| Mean return | **3328.1** (sd 1528.1) | 2850.7 (sd 1627.4) |
| Mean episode length | **843.5** steps | 725.9 steps |
| Mean distance travelled | **141.6 m** | 121.7 m |
| Mean velocity | 0.1690 m/step | 0.1650 m/step |
| Early termination rate | **31.5%** | 45.5% |
| Mean energy cost | 0.829 | 0.867 |

Return difference 477.4, standard error 158, **t = 3.02**.
Termination rate difference 14.0 points, **z = 2.91**.
Cohen's d = 0.30, a modest effect measured precisely.

Two independent metrics agree at roughly the 3-sigma level.

### What actually degraded

Velocity is nearly identical between the two checkpoints: 0.1690 against
0.1650 m/step. The later policy walks just as fast. It falls over more
often, and the episode-length and return differences follow from that.

So the accurate statement is not that the later checkpoint is worse at
locomotion. It is that **the later checkpoint retains gait quality but
loses stability**. That distinction only appears because the harness reports
termination rate and distance separately rather than collapsing everything
into return.

### Training metrics

`approx_kl` climbed from 0.478 at 7.4M steps to 1.684 by the end, well
outside PPO's healthy range of roughly 0.01 to 0.05, with `clip_fraction`
reaching 0.796. That indicates the policy was updating far too aggressively
in the final stretch of training.

This is the explanation most consistent with the logs. A single training run
cannot establish causation, so it is reported as a coinciding condition
rather than a proven cause. More training steps did not mean a better policy
here.

## Part 3: Robustness testing

The 7M-step checkpoint was evaluated under physics conditions it never saw
during training.

> **Exploratory, n=10 episodes per scenario.** With a per-episode return
> standard deviation near 1500, differences of this size sit within noise.
> These are reported as observations to motivate further testing, not as
> established results.

| Scenario | Success rate | Mean return |
|---|---|---|
| Nominal (training conditions) | 90% | 4203.9 |
| Low friction (-40%) | 100% | 5562.7 |
| Very low friction (-60%) | 100% | 5610.2 |
| Heavier robot (+20% mass) | 70% | 2512.0 |
| Much heavier (+40% mass) | 60% | 2876.2 |

Increased mass appeared to degrade performance, as expected. Reduced
friction appeared to *improve* it, and recorded video confirms the motion is
genuine locomotion rather than an artifact of the episode-termination rule.
The underlying mechanism was not isolated. A natural next step is analyzing
contact forces during footfall, at a sample size large enough to separate
the effect from seed noise.

## Part 4: Gait asymmetry

Visual inspection of the recorded video suggested one leg moved less than
the others. Per-joint angle range was logged over a single full episode:

| Joint | Range of motion |
|---|---|
| hip_1 | 32.1° |
| hip_2 | 70.1° |
| hip_3 | 32.7° |
| hip_4 | 68.1° |

> Single-episode measurement. The 2x gap is large enough to be visible in
> video and consistent across the recording, but this is one rollout.

Hips on legs 1 and 3 show roughly half the range of legs 2 and 4, a
quantified asymmetric gait pattern rather than a training failure. This is a
documented, common outcome in quadruped RL, where policies often converge on
gaits that do not use all limbs symmetrically.

## Part 5: Automated instability detection

`check_training_health.py` reads a training log and automatically flags runs
where `approx_kl` or `clip_fraction` cross known danger thresholds. Verified
against the actual values from the 10M-step run:

```
Found 4 instability warning(s):
  WARNING at step 7377408: approx_kl=0.478 (>= 0.1)
  WARNING at step 7377408: clip_fraction=0.763 (>= 0.4)
  CRITICAL at step 10000384: approx_kl=1.684 (>= 0.5)
  WARNING at step 10000384: clip_fraction=0.796 (>= 0.4)
```

A `clip_fraction` of 0.796 means PPO's safety clipping was active on roughly
80% of updates, which is constant rather than occasional.

This automates a check that was originally caught by manually reading
training logs, turning an observation into a repeatable tool.

## A note on evaluation scripts

`evaluate.py` builds a bare `gym.make("Ant-v5")` with no normalization. It
predates the full training run and is not valid for evaluating policies from
`train_full.py`, which were trained under `VecNormalize`. Fed unnormalized
observations, those policies simply stand still: both the 7M and 10M
checkpoints travel under 0.5 m across a full 1000-step episode.

`evaluate_normalized.py` loads the statistics saved at each checkpoint, sets
`training=False` so they stop updating, and sets `norm_reward=False` so
returns are in raw environment units. All Part 2 numbers come from it.

Both scripts are kept, because the difference between them is a useful
illustration: an evaluation pipeline that silently diverges from the training
pipeline produces numbers that look plausible and mean nothing.

## What I'd build on next

- Wire the CSV logger into `train_full.py` so instability detection runs
  automatically during training, not just as a post-hoc check
- Rerun Part 3 at 100+ episodes per scenario with confidence intervals, so
  the friction result can be confirmed or dropped
- Investigate the low-friction performance change by logging contact forces
  during footfall
- Extend robustness testing to sensor noise and external perturbations
- Apply the same evaluation harness to a humanoid task
