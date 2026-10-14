# M3 research — reading list for the Meta-RL gate

Companion to [`module3.md`](module3.md). It collects the papers behind each design choice in
the Meta-RL pipeline (PPO, GRU, RL², and the MAML alternative we did not use), with one line on
*why each paper matters to M3* and *where it shows up in the code*.

> **How this list was built.** Each entry was looked up on the web on 2026-10-15 and the
> bibliographic details (authors, arXiv id, year) come from the arXiv/venue records returned.
> The "what it says" lines paraphrase the abstracts. Anything marked **(abstract only)** means
> only the abstract was checked, not the full text — read the paper before quoting detailed
> results or equations in the thesis. Keep this file the *only* place the reference list lives.

---

## 1. PPO — the training algorithm

| | |
|---|---|
| **Schulman, Wolski, Dhariwal, Radford, Klimov (2017).** *Proximal Policy Optimization Algorithms.* arXiv:1707.06347 | <https://arxiv.org/abs/1707.06347> |

**What it says (abstract).** A policy-gradient method that alternates between collecting data by
interacting with the environment and optimising a "surrogate" objective, and — unlike earlier
methods — reuses each batch for several epochs of minibatch updates. Aims to keep the benefits
of trust-region methods (TRPO) with a much simpler implementation.

**Why M3 uses it.** A learned gate is trained from noisy, partly random cyclone episodes; PPO's
clipped update stops one unlucky batch from wrecking the policy. It replaces the plain
REINFORCE used in the Week-7 toy demo.
**In the code.** `src/module3/train/ppo.py` — `ppo_update` (clipped objective), `train_gate`.

| | |
|---|---|
| **Schulman, Moritz, Levine, Jordan, Abbeel (2015/ICLR 2016).** *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv:1506.02438 | <https://arxiv.org/abs/1506.02438> |

**What it says (abstract).** Introduces GAE, a family of policy-gradient estimators with two
parameters (γ, λ) that uses a value function to cut variance at the price of a little bias; can
be read as a discounted sum of temporal-difference residuals.

**Why M3 uses it.** PPO needs an "advantage" per step (was this choice better than expected?).
GAE gives a low-variance version of it, which matters with 80-step episodes and sparse
damage events.
**In the code.** `train/ppo.py` — `compute_gae` (unit-tested against a hand calculation).

| | |
|---|---|
| **Huang, Dossa, Raffin, Kanervisto, Wang (ICLR Blog Track).** *The 37 Implementation Details of Proximal Policy Optimization.* | <https://github.com/vwxyzjn/ppo-implementation-details> |

**Why it is here.** PPO's headline idea is simple; its results depend on many small details
(advantage normalisation, gradient clipping, entropy bonus, …). This is the checklist we followed
for those details. Year is inconsistent across sources (2022/2023) — cite the venue page you use.

| | |
|---|---|
| **Engstrom et al. (ICLR 2020).** *Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO.* arXiv:2005.12729 | <https://arxiv.org/pdf/2005.12729> |

**Why it is here.** Argues that code-level optimisations explain much of PPO's gain over TRPO.
Use it to justify reporting *multiple seeds* and exact hyper-parameters rather than a single run.

---

## 2. GRU — the memory inside the gate

| | |
|---|---|
| **Cho, van Merriënboer, Gulcehre, Bahdanau, Bougares, Schwenk, Bengio (2014).** *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation.* arXiv:1406.1078 | <https://arxiv.org/abs/1406.1078> |

**What it says (abstract).** Proposes an encoder–decoder pair of recurrent networks for
translation; the encoder compresses a sequence into a fixed vector. (This is the paper that
introduced the gated recurrent unit; the search snippets did not state that explicitly — confirm
in the full text before citing the GRU definition.) **(abstract only)**

**Why M3 uses it.** The gate must remember what it has seen so far in a storm (e.g. "uncertainty
has been climbing for five steps"), because one reading of `u` is noisy.
**In the code.** `src/module3/policy/recurrent_gate.py` — `nn.GRU` inside `RecurrentGate`.

| | |
|---|---|
| **Chung, Gulcehre, Cho, Bengio (2014).** *Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.* arXiv:1412.3555 | <https://arxiv.org/abs/1412.3555> |

**What it says (abstract).** Compares gated units (LSTM, GRU) with plain tanh recurrent units on
music and speech modelling: gated units beat tanh units, and GRU is about as good as LSTM.
**(abstract only)**

**Why it is here.** Justifies GRU over LSTM: comparable quality with fewer parameters, which keeps
the exported gate small enough for the Go hot path.

| | |
|---|---|
| **Hausknecht & Stone (2015).** *Deep Recurrent Q-Learning for Partially Observable MDPs.* arXiv:1507.06527 | <https://arxiv.org/abs/1507.06527> |

**What it says (abstract).** Adds a recurrent layer to DQN so the agent sees one frame at a time yet
matches DQN, and degrades more gracefully when observations become partial.

**Why it is here.** Early evidence that recurrence is a workable answer to partial observability —
exactly our situation (M2's `u` and `observed_fraction` are noisy, partial views of the grid).

| | |
|---|---|
| **Ni, Eysenbach, Salakhutdinov (ICML 2022).** *Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs.* arXiv:2110.05038 | <https://arxiv.org/abs/2110.05038> |

**What it says (abstract).** Careful architecture and hyper-parameter choices let plain recurrent
model-free RL match or beat several specialised POMDP/meta-RL methods on most of the benchmarks
tested (18 of 21 environments reported). Code: <https://github.com/twni2016/pomdp-baselines>.

**Why M3 uses it.** This is the strongest single citation for our design choice: a recurrent
policy trained with a standard RL algorithm is a *strong baseline*, so we do not need a heavier
meta-learner. Also the argument for keeping the architecture simple.

---

## 3. RL² — how a GRU becomes "Meta-RL"

| | |
|---|---|
| **Duan, Schulman, Chen, Bartlett, Sutskever, Abbeel (2016).** *RL²: Fast Reinforcement Learning via Slow Reinforcement Learning.* arXiv:1611.02779 | <https://arxiv.org/abs/1611.02779> |

**What it says (abstract).** Instead of hand-designing a fast RL algorithm, represent it as a
recurrent network whose weights are learned by an ordinary ("slow") RL method. The network's
activations hold the working state of the fast learner for the current task. Tested on random
bandits and finite MDPs (near human-designed algorithms with optimality guarantees) and on
vision-based navigation.

**Why M3 uses it.** This is the justification for the word *Meta* in "Meta-RL gate": train one
recurrent policy across a *distribution of cyclones*; at run time its memory infers which kind of
storm this is and adapts — with no gradient step in the loop, just a forward pass.
**In the code.** `RecurrentGate` feeds the previous action, previous reward and a trial-done flag
back in as inputs (that is the RL² recipe); `train/ppo.py::collect_tasks` samples one storm per
"task" and can run several trials of the same storm (`policy.trials_per_task`) carrying memory.

| | |
|---|---|
| **Wang, Kurth-Nelson, Tirumala, Soyer, Leibo, Munos, Blundell, Kumaran, Botvinick (2016).** *Learning to reinforcement learn.* arXiv:1611.05763 | <https://arxiv.org/abs/1611.05763> |

**Status.** Independently proposed the same idea at the same time (a recurrent network trained
with RL whose dynamics implement a second, faster RL algorithm). The search returned only the
reference record, not the abstract — **read the paper before describing its content in the
thesis.** Cite alongside Duan et al. as the concurrent work.

| | |
|---|---|
| **Beck, Vuorio, Liu, Xiong, Zintgraf, Finn, Whiteson.** *A Survey of Meta-Reinforcement Learning* (arXiv:2301.08028; v4 retitled *A Tutorial on Meta-Reinforcement Learning*, Foundations and Trends in ML, 2025) | <https://arxiv.org/abs/2301.08028> |

**What it says (abstract).** Organises meta-RL by whether there is a task distribution and how
much learning budget each task gets; covers algorithms, applications and open problems. Cite
whichever version title you actually read (the arXiv id covers all versions).

**Why it is here.** The map of the field: places RL² (memory-based) and MAML (gradient-based) side
by side, which is what the "why not MAML" argument in `module3.md` leans on.

---

## 4. MAML — the alternative we did not use (kept as an option)

| | |
|---|---|
| **Finn, Abbeel, Levine (ICML 2017).** *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.* arXiv:1703.03400 | <https://arxiv.org/abs/1703.03400> |

**What it says (abstract).** A meta-learning method for any model trained by gradient descent.
It trains the *initial parameters* so that a few gradient updates on a new task's data give strong
performance; shown on few-shot classification, regression, and fine-tuning of policy-gradient RL
policies.

**Why it is NOT the default here.** See `module3.md` § "Why RL² and not MAML". In one line: MAML
adapts by *running gradient descent on new-task data at deployment*, and the gate gets one
80-step storm with no spare episodes to adapt on and a millisecond-scale latency budget, whereas
RL² adapts with a forward pass.
**Where it could plug in.** `policy.meta_method: maml` is reserved in `pipeline.yaml` and raises
`NotImplementedError` today. See `module3.md` for when it would be the right tool.

---

## 5. Why the gate pays to think — deliberation cost

| | |
|---|---|
| **Harb, Bacon, Klissarov, Precup (AAAI 2018).** *When Waiting is not an Option: Learning Options with a Deliberation Cost.* arXiv:1709.04571 | <https://arxiv.org/abs/1709.04571> |

**What it says (abstract).** Frames good temporal abstractions through bounded rationality: a
*deliberation cost* is charged when the agent switches to a new option and is subtracted from the
base reward; learned with a Lagrangian-relaxation objective. Reported gains in performance and
interpretability on Atari.

**Why M3 uses it.** The same idea in a different setting: our reward subtracts a deliberation cost
(and a latency cost) every time the gate calls System 2. (This was already in your locked
literature; listed here so all RL references are in one file.)

---

## 6. Related work in power-system resilience (context for the paper)

These came from a single web search and have **not** been read in full; they are starting points
for the related-work section, not verified claims.

| Reference | Why it is relevant |
|---|---|
| Wu et al. (2025), *Expert Systems with Applications* — cooperative multi-agent deep RL energy management under typhoon-driven demand uncertainty, IEEE 33-bus (arXiv:2601.12657). <https://arxiv.org/abs/2601.12657> | Closest example of DRL for microgrid operation under extreme weather. |
| Kumar & De (2025), *Renewable and Sustainable Energy Reviews* 222 — review of deep RL for power-system resilience. <https://ideas.repec.org/a/eee/rensus/v222y2025ics1364032125006240.html> | Survey to position M3 against. |
| A 2026 dissertation on parameterised-DQN restoration of critical loads in an islanded microgrid (IEEE 123-node, 17 prioritised loads). | Same priority-weighted-load idea as our reward, applied to restoration. Find the exact citation before using. |

**Gap to claim (to be confirmed by a proper literature pass).** None of the above separates *when
to deliberate* from *what to do*; M3's contribution is the cost-aware gate between a reactive
path and a deliberative one, verified by an external physics firewall.

---

## Still to read / cite properly

- Wang et al. 2016 (full text) — see §3.
- "Finding the Time to Think" — already in the locked literature; add the full citation here.
- Real best-track / wind data sources for the historical cyclones (Burevi, Ditwah) so the
  placeholder profiles in `scenario/cyclone.py` can be replaced.
