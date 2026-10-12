# Experiments and evaluation protocol

> **Simulation study.** All results describe the synthetic simulator in this repository. They are not evidence
> about any real mine.

## 1. Question

Does a masked Double DQN, trained on simulated shifts, dispatch trucks better than three classical rules, on
shifts it has never seen? "Better" is measured by the operational metrics below, and every claim has to be backed
by a paired comparison on identical seeds.

## 2. Scenario and seeds

| Item | Setting | Source |
|---|---|---|
| Fleet and mine | 20 trucks, 4 shovels, 1 depot, 480-minute shift | `configs/default.yaml` (`scenario`) |
| Fleet generation | NumPy seed 42 | `scenario.seed` |
| Training shifts | Reset seeds `10000 + k` for episode `k` | `training.seed_base` |
| Evaluation shifts | Reset seeds `50000 … 50029` (30 shifts) | `evaluation.seed_start`, `n_episodes` |
| Overlap check | Training and evaluation seed ranges must not overlap | `ProjectConfig.validate()` |

The scenario is fixed across all shifts. Only the randomness (breakdowns, repair times, travel and loading noise)
changes from seed to seed. Every policy sees exactly the same 30 shifts, so each comparison is **paired**.

## 3. Policies

| Policy | Trained? | Decision rule |
|---|---|---|
| DQN (masked Double, greedy) | Yes, on training shifts only | argmax of Q over valid pairs |
| `random_valid` | No | Uniform over valid pairs (seeded per shift) |
| `nearest_eligible` | No | Smallest nominal travel time, ties by FIFO truck order |
| `shortest_queue` | No | Smallest estimated loading-start time, ties by FIFO truck order |

Every policy is restricted to the same action mask, and the evaluation runs with
`invalid_action_policy: raise`, so a policy that picked a masked action would fail the run rather than being
penalised.

## 4. Protocol

1. **Development.** Architecture, reward weights, and hyperparameters were set from the defaults in `configs/`.
   The reported study does not include a hyperparameter sweep. Only one configuration was trained and reported.
2. **Training.** `scripts/train.py` runs on training seeds only. Every episode writes one raw row. Checkpoints are
   saved, and the final weights are `checkpoints/<run>/final.pt`.
3. **Evaluation.** `scripts/evaluate.py` loads the final checkpoint in inference mode. It runs the DQN and each
   baseline on seeds 50000–50029. No transitions are stored and no gradients run (a test verifies this).
4. **Analysis.** `scripts/compare_baselines.py` computes summaries and paired tests from the episode CSV. It writes
   the figures.
5. **Reporting.** Every number is taken from those files. Nothing is re-run or edited by hand.

**Disclosure.** During development, the three baselines were run once on seeds 50000–50009. The purpose was to
check that the scenario is non-trivial. That run is why the scenario's queueing regime is known. No scenario
parameter was changed afterwards. The check did not involve the DQN, and no hyperparameter was tuned on the
evaluation seeds.

## 5. Metrics

Each shift produces one row per policy (`results/evaluation/episodes.csv`):

| Requirement | Column(s) |
|---|---|
| Total payload delivered | `payload_t` (tonnes), `loads` |
| Average payload per shift | Mean of `payload_t` over shifts |
| Average truck waiting time | `avg_wait_min` (queue minutes per shovel arrival), `truck_wait_min` (total) |
| Total truck travel time | `truck_travel_min` |
| Shovel utilisation | `shovel_utilization` (loading minutes / (shovels × horizon)) |
| Shovel availability | `shovel_availability` (operating minutes / (shovels × horizon)) |
| Truck utilisation | `truck_utilization` (travel + loading + unloading minutes / (trucks × horizon)) |
| Shovel idle time | `shovel_idle_min` (operating but not loading) |
| Shovel starvation (dispatch-attributable idleness) | `shovel_starvation_min` |
| Breakdown count | `truck_breakdowns`, `shovel_breakdowns` |
| Invalid action count | `invalid_actions` |
| Unnecessary travel | `unnecessary_travel_min` (regret at dispatch) |
| Episode reward | `episode_return` |
| Evaluation runtime | `runtime_s` (wall-clock seconds per shift, including the policy) |

Utilisation and availability are always within [0, 1] (a test checks this). Payload is always consistent with the
number of loads and the capacity classes.

## 6. Statistics

- **Summary.** Mean, sample standard deviation, and percentile bootstrap 95% CI over the 30 shifts (5,000 resamples,
  fixed seed). Table: `results/evaluation/summary.csv`.
- **Paired comparisons.** For each metric and each baseline, the DQN is compared with that baseline on the same 30
  seeds. The reported quantities are the mean difference (DQN − baseline), its bootstrap 95% CI, the share of seeds
  where DQN is better (direction-aware, since lower waiting is better), and a two-sided sign-flip permutation p-value
  (10,000 permutations). Table: `results/evaluation/paired_comparisons.csv`.
- **Multiple comparisons.** Five metrics against three baselines is 15 tests. The p-values are reported
  uncorrected, so read them as descriptive, not as confirmatory.

A difference is only described as an improvement when the paired CI excludes zero **and** the direction is what the
metric's definition requires. Otherwise the result is reported as no detectable difference.

## 7. Budgets and what the results mean

| Run | Episodes | Purpose | Status |
|---|---|---|---|
| Smoke (`--quick`) | 3 | Checks the pipeline end to end | Not a model. Never report its metrics as performance. |
| Full (`configs/default.yaml`) | 300 | The reported run | See the results section in the README |

A single training run gives one set of weights. Differences between seeds of the same configuration are not
measured, so a single training run cannot speak to training variability. That is listed as future work.

## 8. Reproducing the results

```bash
python scripts/train.py --run-name dqn_default
python scripts/evaluate.py --checkpoint checkpoints/dqn_default/final.pt
python scripts/compare_baselines.py --figures docs/figures
```

With the same configuration, seeds, and library versions, the evaluation CSV is reproduced exactly, apart from
`runtime_s` (wall-clock time). Training on GPU versus CPU may differ slightly in floating point, so reproduce on the
same device type for bit-identical checkpoints.
