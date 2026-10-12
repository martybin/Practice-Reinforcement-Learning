# Architecture

## Layers

```
configs/*.yaml ──► config.py (typed, validated)
                       │
        ┌──────────────┴───────────────────────────────────────────────┐
        ▼                                                              ▼
environment/                                                     agents/
  scenario.py      synthetic fleet, nominal times (NumPy seed)       networks.py        Q-MLP
  entities.py      Truck, Shovel, Timer, states                      replay_buffer.py   ring buffer (NumPy)
  simulator.py     discrete-event engine, decision epochs            action_masking.py  masked argmax/max/sample
  observation.py   535-dim normalised features (decision time only)  dqn_agent.py       masked Double DQN
  action_codec.py  pair encode/decode, validity mask
  reward.py        weighted, logged reward components
  mining_env.py    Gymnasium env (reset/step/render/close)
        │                                                              │
        └──────────────┬───────────────────────────────────────────────┘
                       ▼
policies/          random_valid, nearest_eligible, shortest_queue (same DecisionView as DQN)
training/          train.py (loop, CSV, checkpoints, resume), evaluate.py (held-out protocol)
metrics/           operational_metrics.py (summaries, bootstrap CI, paired tests),
                   visualization.py (figures from CSVs)
app/streamlit_app.py  dashboard: calls the same simulator, checkpoints, and CSVs
scripts/           thin CLIs: train.py, evaluate.py, compare_baselines.py
```

The dependency direction is one-way: the environment knows nothing about agents, training, or metrics, so the
simulator can be tested and reused without torch.

## One decision, end to end

1. `env.step(a)` decodes `a` into (truck, shovel). If the mask marks it invalid, the penalty path runs.
2. `simulator.dispatch` moves the truck to `TRAVEL_EMPTY` and schedules its arrival (plus a possible breakdown).
3. `simulator.advance_to_decision` pops events in time order, accruing time-based statistics between them, until
   a truck is idle again or the shift ends.
4. `compute_reward` turns the counter deltas into weighted components.
5. `build_observation` and `current_action_mask` produce the next state (observation and mask).
6. The training loop stores `(s, a, r, s', terminated, truncated, mask, next_mask)` in the replay buffer. Then it
   runs one gradient step (masked Double DQN) if the schedule says so.

## Key decisions and why

| Decision | Alternative considered | Reason |
|---|---|---|
| Discrete-event engine with decision epochs | Fixed-step clock (for example, 1-minute ticks) | Exact event times and no discretisation error. The agent only acts when it has a real choice, which makes episodes short (about 200–300 decisions per shift). |
| Stale events invalidated by tokens | Removing events from the heap | O(log n) pushes and no heap surgery. Breakdown pause/resume becomes a token bump. |
| Observations use nominal durations | Exposing realised times | Prevents leakage of the random noise. This is a property the tests enforce. |
| Mask shared by exploration, action choice and TD targets | Mask only in exploration | Masking only one of them lets the agent learn values for actions it can never take. |
| Invalid action: penalise and do not change the state, with a cap | Silent clipping or a random valid substitute | Makes the failure visible in rewards and logs, and it cannot be farmed. |
| Huber loss, Adam, grad-norm clipping | MSE, SGD | Standard DQN practice. Huber bounds the gradient of large TD errors, and clipping guards against rare spikes. |
| Hard target sync every N gradient steps | Polyak (soft) updates | Simpler and has fewer hyperparameters. Soft updates are listed as future work. |
| Replay stored as NumPy ring arrays | Python deque of tuples | Preallocated arrays keep sampling to a few fancy-index operations. |
| Checkpoint excludes the replay buffer by default | Always saving it | A buffer at default size is about 200 MB. Resume restores the networks, optimiser, epsilon, and counters, and refills the buffer. |
| Evaluation seeds disjoint from training seeds, checked in config validation | Trusting the convention | Catches an accidental overlap at start-up, not in a results table. |

## Reproducibility

- The fleet is generated from `scenario.seed` with NumPy, in a fixed draw order.
- Each shift is driven by its reset seed (`training.seed_base + episode` for training and
  `evaluation.seed_start + i` for evaluation). Two runs with identical seeds and configuration produce identical
  trajectories, which the tests check.
- Random valid and exploratory choices use dedicated seeded generators (`np.random.Generator`).
- Every checkpoint embeds the full configuration, and `run_config.json` is written beside each run's logs.

## Outputs

| Path | Content | Committed? |
|---|---|---|
| `checkpoints/<run>/latest.pt`, `final.pt` | Weights, optimiser, counters, RNG state, configuration | No (`.gitignore`) |
| `results/training/<run>/episodes.csv` | One raw row per training episode | No |
| `results/training/<run>/run_config.json` | Full resolved configuration | No |
| `results/training/<run>/tensorboard/` | Optional TensorBoard scalars | No |
| `results/evaluation/episodes.csv` | One raw row per (policy, held-out seed) | No |
| `results/evaluation/summary.{csv,md}`, `paired_comparisons.csv` | Aggregated statistics | No |
| `results/figures/*.png` | Figures generated from the CSVs | No |
| `docs/figures/*.png` | A copy of the figures used in the README, from a real run | Yes |
