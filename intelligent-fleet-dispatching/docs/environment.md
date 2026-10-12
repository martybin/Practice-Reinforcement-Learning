# Environment and MDP formulation

> **Scope.** This is a **simulation** on **synthetic data**, built to study dispatch policies. It is not a
> validated model of any mine, and nothing here should be used to make real operational decisions.

## 1. Physical system

One open-pit shift (default 480 minutes) with:

- **1 depot / dump** (node 0). Trucks start here, and every load is tipped here.
- **S = 4 loading shovels** (nodes 1..4). Each shovel has its own loading rate (t/min).
- **T = 20 haul trucks**. Each truck has its own capacity (218, 240 or 290 t), speed factor (0.90–1.15),
  and reliability factor (0.7–1.4, multiplying breakdown MTBF).

Nominal one-way times (minutes) for a truck with speed factor 1.0 are in `configs/default.yaml`:
`depot_to_shovel = [8, 11, 14, 18]` and `shovel_to_depot = [9, 12.5, 15.5, 20]`. The actual time for truck *k*
is its speed factor times the nominal value. Shovel-to-shovel travel is not modelled, because trucks never drive
directly between shovels in this shift model. Adding that capability would mean extending the travel
configuration and the transition rules, not changing the interface.

A truck cycles through the states below. Only `IDLE` trucks are eligible for dispatch.

| State | Meaning | Ends with |
|---|---|---|
| `IDLE` | Waiting at the depot for a dispatch decision | Agent dispatches it to a shovel |
| `TRAVEL_EMPTY` | Returning empty to the assigned shovel | Arrival at the shovel (queue or load) |
| `QUEUED` | Waiting behind other trucks at a shovel | Its turn at the loading face |
| `LOADING` | Being loaded to capacity | Load complete |
| `TRAVEL_LOADED` | Hauling ore back to the depot | Arrival at the depot |
| `UNLOADING` | Tipping ore at the depot | Payload is delivered and the truck becomes `IDLE` |
| `MAINTENANCE` | Broken down and under repair | Repair completes, then the interrupted state resumes |

Shovels are `OPERATIONAL` or in `MAINTENANCE`. A shovel only loads one truck at a time, and its queue is FIFO.

## 2. Time model: discrete-event with decision epochs

The simulator (`environment/simulator.py`) is **event-driven**. A heap of timestamped events covers arrivals,
loading and unloading completions, breakdowns, and repairs. Durations are continuous, so queueing and breakdown
timing are exact, and there is no time-step discretisation error.

Decision epochs are the instants at which some truck is `IDLE` and some shovel is operational. The clock
advances from one epoch to the next, so the agent only acts when a real choice exists.

- **Stale events.** Each truck and shovel holds a token. Any state change that reschedules an activity bumps the
  token, which invalidates whatever was previously scheduled. A breakdown pauses the interrupted activity, with
  its elapsed progress saved, and the remaining time resumes after repair.
- **Breakdowns.** A truck can break down only while driving (`TRAVEL_EMPTY` or `TRAVEL_LOADED`). Trucks in queue,
  loading, or unloading are not exposed. A shovel can break down whenever it is operational. Times to failure
  are exponential with a mean of MTBF × reliability factor. Repair times are lognormal.
- **Disturbances (optional).** Lognormal multiplicative noise with median 1 on each trip, loading, and unloading
  duration. Setting `disturbances_enabled: false` removes it.

Time conservation is checked in the test suite: every truck's time across its states sums to exactly
`trucks × horizon`.

## 3. Information discipline: what the agent may observe

At a decision epoch, the agent may only use information that exists at that instant:

- Every truck's state, destination, payload, time in its current state, and expected remaining time.
- Every shovel's status, queue, loading truck, and committed backlog.
- Nominal (expected) travel and loading times. Those are known from the mine's plan.

It may **not** use:

- Realised travel, loading, or unloading durations, which contain the noise.
- The length of a breakdown repair, or when a future breakdown will occur.
- Any future event.

Remaining times in observations are computed from **nominal** durations minus progress, so the realised noise stays
hidden. `tests/test_simulator.py::test_observation_does_not_reveal_unobservable_noise` checks this property directly:
two shifts that differ only in their realised noise produce byte-identical observations after the same dispatch.

## 4. Observation

The observation is a flat `float32` vector of length **535** (for T=20, S=4). Every value is clipped to **[0, 1]**.
Time-valued features are divided by `env.time_scale_minutes` (60 by default). Counts are divided by the number of
trucks or shovels.

| Block | Count | Features (per entity) |
|---|---|---|
| Per truck | 20 × 17 = 340 | Status one-hot (7) · destination one-hot over depot + 4 shovels (5) · loaded flag · available flag (not in maintenance) · capacity / max capacity · expected remaining time / 60 · time in current status / 60 |
| Per shovel | 4 × 8 = 32 | Operational · rate / max rate · busy · expected remaining load time / 60 · queue length / T · trucks en route / T · backlog / 60 · busy fraction so far |
| Per (truck, shovel) pair | 80 × 2 = 160 | Nominal travel / 60 · estimated time until loading would start / 60 (travel + backlog) |
| Global | 3 | Remaining shift fraction · idle-truck fraction · operational-shovel fraction |

Pair index = `truck × S + shovel`, the same as the action index. Feature names are generated by
`observation.feature_names()` and checked in tests for length and uniqueness.

**Backlog estimate.** A shovel's backlog is the expected time already committed to it: a repair in progress, the
remaining load, every queued load, and every load of a truck en route to it. It is what makes the "estimated time
until loading" feature informative about congestion.

## 5. Action

- **Space.** `Discrete(T × S) = Discrete(80)`. Action `a` means "dispatch truck `a // S` to shovel `a % S`".
- **Validity.** Action `a` is valid iff truck `a // S` is `IDLE` and shovel `a % S` is `OPERATIONAL`. The mask
  is the outer product of those two vectors (`environment/action_codec.py`).
- **No WAIT action.** When several trucks are idle, the agent chooses which truck and which shovel. That is the
  full decision, so no separate "do nothing" choice is needed. Every idle truck is dispatched at the decision
  instant, so no idle time accrues. The only waiting in the system is queueing, and the reward already charges
  that. A WAIT action would need its own idle-time penalty. Without one, the agent could dodge the queue charge by
  holding trucks back at the depot. If a future study needs deliberate holding, add the penalty with the action.
- **Invalid actions.** Masked actions are never proposed by exploration, the greedy policy, or the baselines. If
  one reaches `step()` anyway, the policy setting decides what happens:
  - `penalize` (default): the reward takes `invalid_action` (−5), **no state changes**, and the counter increases.
    After `max_consecutive_invalid_actions` (10) in a row at one decision epoch, `InvalidActionError` is raised,
    because time does not advance and a buggy policy would otherwise loop forever.
  - `raise`: fail immediately.
  The penalty is large enough that exploiting invalid actions never pays, and the state is unchanged, so an invalid
  action can never be used to skip a decision.

**Trade-offs of a discrete action space for this problem.**

- *Pros.* Q-learning applies directly, the mask removes illegal choices cleanly, and the policy is easy to inspect
  (one Q-value per dispatch).
- *Cons.* The action count grows as T × S. A fleet of 200 trucks and 20 shovels would need 4,000 outputs, most of
  them masked, and the network would have to learn each truck's index separately. The representation is also tied
  to the fleet size: a policy trained for 20 trucks cannot be used for 25 without retraining. A permutation-invariant
  architecture (set or graph encoder over trucks and shovels, scoring each pair) would remove both limitations.
  That is listed under future work.

## 6. Reward

Per decision step, the reward is the weighted sum of components, accrued over the interval between this decision
and the next one (`environment/reward.py`, weights in `configs/reward.yaml`).

| Component | Raw quantity | Default coefficient | When it is charged |
|---|---|---|---|
| `payload` | Tonnes dumped at the depot / 100 | +1.0 | At `UNLOAD_DONE` |
| `truck_wait` | Truck-minutes spent in `QUEUED` | −0.05 | Continuously, while queued |
| `unnecessary_travel` | Nominal minutes by which the chosen shovel is farther than the nearest operational shovel | −0.02 | Once, at dispatch |
| `shovel_starvation` | Shovel-minutes that are operational, not loading, with an empty queue and no truck en route | −0.02 | Continuously |
| `truck_breakdown` | Truck breakdown events | −1.0 | Once per breakdown |
| `invalid_action` | Rejected actions | −5.0 | Once per rejected action |

Every component is logged separately (`info["reward_components"]`) and in the episode CSVs (`r_*` columns). Each
component can be disabled through `enabled: false`, which makes ablations a configuration change.

**Scale.** The payload term is per 100 t, so one 240 t load is worth 2.4. The waiting coefficient, −0.05 per
truck-minute, is roughly the opportunity cost of the fleet's average hauling rate (about 7 t/min, that is ~0.07
reward per truck-minute), rounded down.

**Avoiding double counting.**

- Waiting is charged only as accrued queue minutes. There is no extra per-queue-entry charge.
- A breakdown's repair time is not charged as waiting. The breakdown itself carries a single fixed cost.
- Unnecessary travel is charged once, at dispatch, as a distance regret. Its time is not charged again as travel.
- Starvation minutes are disjoint from truck-waiting minutes by construction (an idle shovel has an empty queue,
  and a queued truck always has a busy or broken shovel in front of it).

**Shaping vs objective.** The payload term is the actual objective. The waiting and starvation terms shape
credit assignment: a dispatch's effect on throughput only shows up after a full haul cycle, and the shaping terms
provide signal within that cycle.

**Information in rewards.** The reward for a step is the realised outcome of the interval that follows the action.
That is standard in an MDP. It does not leak future information into the observation, and nothing the reward
depends on is hidden from the agent's inputs in a way that would make learning impossible.

**Reward hacking and how it is guarded against.**

| Possible exploit | Guard |
|---|---|
| Submitting invalid actions for free rewards or free decisions | The mask removes them from exploration and from the greedy choice. A rejected action costs −5 and changes nothing. Repeats are capped. |
| Dispatching to a far shovel to dodge queues, to avoid the regret term | This is a real trade-off that the reward is meant to expose, not an exploit. Queue time and throughput both enter the reward. |
| Idling shovels to avoid waiting | Shovel starvation is penalised continuously, so a shovel left unfed loses reward. |
| Making many zero-duration decisions at one instant to get more Q-updates | Zero-duration steps earn zero reward, and the time-based metrics stay correct. |
| Gaming the time limit by stalling near the horizon | The remaining-shift feature is observed, and the shift ends at the horizon. There is nothing to gain from stalling. |
| Gaming breakdowns (for example, by avoiding travel) | Breakdowns cost −1 each. The agent can reduce exposure only by shorter trips, which the payload and waiting terms already reward. |

## 7. Termination and truncation

The shift ends when the clock reaches the horizon. That is a **time-limit truncation**: `truncated = True` and
`terminated = False`. The default MDP has no genuine terminal state (no absorbing failure), so `terminated` is
always `False`.

The learner still handles terminal transitions correctly. `terminated` zeroes the bootstrap term. A test
(`tests/test_dqn_agent.py::test_genuine_terminal_transition_does_not_bootstrap`) covers that path, so a future
absorbing state would work without changes to the learner.

**Truncation policy.** When the horizon is reached, the environment returns the state at that instant along with
its action mask. Any decisions still pending at that instant (idle trucks) are visible in the mask. The learner
bootstraps from `max_a Q(s_T, a)` over those valid actions, because the value of continuing is not zero when
trucks are still waiting. If no decision is pending (every truck is mid-cycle), there is no action-value to
bootstrap from, so the target uses 0. The remaining-shift feature is 0 at that point, so the network can learn that
states near the end of a shift are worth little. The only effect is a small pessimistic bias on the last few
decisions of each shift, which is documented here rather than hidden.
