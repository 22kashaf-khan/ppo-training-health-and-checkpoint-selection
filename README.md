# RL Evaluation, Training & Robustness Harness

A full pipeline for training a locomotion policy, evaluating it with proper
statistics, and testing how it holds up outside the exact conditions it was
trained on built to demonstrate not just that a policy can be trained, but
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
  training instability that caused that regression

## Project structure
```
train.py                     short training runs (used for seed comparisons)
evaluate.py                  loads a policy, computes independent metrics
statistics.py                Welch's t-test + Cohen's d between configs
run_experiment.py            orchestrates multi-seed training + evaluation
train_full.py                full 10M-step training run with checkpointing
record_video.py              records an evaluation episode as mp4
robustness_test.py           tests a policy under varied friction/mass
check_leg_symmetry.py        logs per-joint range of motion
check_training_health.py     flags instability in a training log
models_full/                 20 checkpoints saved every 500k steps
videos/                      recorded episodes, including the friction test
```

## Part 1 — Multi-seed statistical evaluation

RL training is highly sensitive to random seed. To demonstrate this
concretely, two configurations (baseline learning rate vs. 3x higher) were
each trained across 5 seeds and compared using independent metrics, not
just training reward.

| Metric | Baseline mean | High LR mean | p-value | Effect size (d) | Verdict |
|---|---|---|---|---|---|
| Mean return | 219.8 | 111.9 | 0.088 | 1.23 (large) | Not significant — large effect, but 5 seeds isn't enough to confirm it |
| Mean energy cost | 0.89 | 1.58 | 0.021 | -2.17 (large) | **Significant** — high LR genuinely uses more energy |

Within a single configuration, seed-to-seed variance alone produced returns
ranging from 96.4 to 325.8 — a 3.4x spread from identical code. This is the
concrete argument for why single-run comparisons in RL are unreliable.

## Part 2 — Full training run and checkpoint comparison

A full 10M-step PPO run was trained on `Ant-v5`, using the tuned Zoo
hyperparameters and observation/reward normalization (`VecNormalize`).

**A checkpoint at 7M steps outperformed the final 10M-step checkpoint**:

| Checkpoint | Episode length | Total reward |
|---|---|---|
| 7,000,000 steps | 1000/1000 (full) | **4627.6** |
| 8,500,000 steps | 842/1000 | 3678.6 |
| 10,000,000 (final) | 693/1000 | 2991.3 |

Training metrics explain why: `approx_kl` climbed from 0.478 at 7.4M steps
to 1.684 by the end — well outside PPO's healthy range (typically
0.01–0.05) — indicating the policy was updating too aggressively and
became less stable the longer training continued. More training steps did
not mean a better policy here.

## Part 3 — Robustness testing

The 7M-step checkpoint was evaluated under physics conditions it never saw
during training:

| Scenario | Success rate | Mean return |
|---|---|---|
| Nominal (training conditions) | 90% | 4203.9 |
| Low friction (-40%) | 100% | 5562.7 |
| Very low friction (-60%) | 100% | 5610.2 |
| Heavier robot (+20% mass) | 70% | 2512.0 |
| Much heavier (+40% mass) | 60% | 2876.2 |

Increased mass degraded performance as expected. Reduced friction
unexpectedly *improved* performance — verified via recorded video to be
genuine locomotion, not an artifact of the episode-termination rule. The
underlying mechanism wasn't further isolated; a natural next step would be
analyzing contact forces during footfall.

## Part 4 — Gait asymmetry

Visual inspection of the recorded video suggested one leg moved less than
the others. This was confirmed by logging per-joint angle range over a full
episode:

| Joint | Range of motion |
|---|---|
| hip_1 | 32.1° |
| hip_2 | 70.1° |
| hip_3 | 32.7° |
| hip_4 | 68.1° |

Hips on legs 1 and 3 show roughly half the range of legs 2 and 4 — a real,
quantified asymmetric gait pattern, not a training failure. This is a
documented, common outcome in quadruped RL, where policies often converge
on gaits that don't use all limbs symmetrically.

## Part 5 — Automated instability detection

`check_training_health.py` reads a training log and automatically flags
runs where `approx_kl` or `clip_fraction` cross known danger thresholds.
Verified against tonight's actual training values:

```
Found 4 instability warning(s):
  WARNING at step 7377408: approx_kl=0.478 (>= 0.1)
  WARNING at step 7377408: clip_fraction=0.763 (>= 0.4)
  CRITICAL at step 10000384: approx_kl=1.684 (>= 0.5)
  WARNING at step 10000384: clip_fraction=0.796 (>= 0.4)
```

This automates a check that was originally caught by manually reading
training logs — turning an observation into a repeatable tool.

## What I'd build on next

- Wire the CSV logger into `train_full.py` so instability detection runs
  automatically during training, not just as a post-hoc check
- Investigate the low-friction performance improvement by logging contact
  forces during footfall
- Extend robustness testing to sensor noise and external perturbations
- Apply the same evaluation harness to a humanoid task
