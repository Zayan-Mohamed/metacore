# Module 2 — Agentic Epistemic Uncertainty Quantification (AUQ) Engine — Starter

A runnable, self-contained starter for **Module 2** (Duwaragie K., J26-DS-317).
It trains an Evidential Deep Learning head on **synthetic** island grid-states and shows
epistemic uncertainty **u = K/S** staying low on normal states and rising to 1.0 on
cyclone (out-of-distribution) states — the core thesis of the module — with **zero
dependency on Module 1's data model**. Drop-in location in the repo:
`services/learned/module2_auq_engine/`.

## Verified results (`python run_demo.py`)
| Metric | Value | Target |
|---|---|---|
| ID 3-class accuracy | 0.969 | — |
| mean epistemic u (ID / OOD) | 0.099 / 1.000 | ID low, OOD high |
| AUROC (u, OOD vs ID) | 1.000 | ≥ 0.90 |
| AUPR (OOD) | 0.997 | high |
| FPR95 | 0.001 | low |
| ECE (calibration) | 0.043 | near 0 |
| competence-drop trigger (ID / OOD) | 0.05 / 1.00 | catch OOD, few false alarms |

## Run
```bash
pip install -r requirements.txt
python run_demo.py
```

## Files
- `synthetic_data.py` — **mock M1 state generator** (ID normal + OOD cyclone). Replace with the real M1→M2 adapter when it lands.
- `edl.py` — EDL head, `u = K/S`, Dirichlet KL, EDL loss, OOD-aware evidence regulariser.
- `trigger.py` — competence-drop trigger (calibrated threshold + hysteresis).
- `evaluate.py` — AUROC / AUPR / FPR95 / ECE (NumPy).
- `contract.py` + `M2_TO_M3_CONTRACT.md` — **M2→M3 message + mock stream for Saabir**.
- `run_demo.py` — end-to-end prototype; writes `sample_m2_to_m3.jsonl`.
- `config.yaml` — K, features, training and trigger settings (retune without code changes).

## Two things worth understanding
1. **Why OOD-aware regularisation is in the loss.** Plain EDL extrapolates *confidently* on far-OOD tabular inputs. The `ood_reg_weight` term drives evidence → 0 on far proxy points so cyclone states read u ≈ 1. This is standard practice and is your defensible design choice.
2. **Uncertainty only flags novelty in features the model uses.** The risk label depends on wind/rain too, so the net learns to attend to them; otherwise it would ignore the cyclone dimensions.

## What is mocked vs real
- **Mock now:** the input states (stand-in for M1's shared ID/OOD scenario library) and the K safety classes' feature ranges.
- **Real later:** swap `synthetic_data` for M1's `{embedding + feature vector + timestamp + quality flags}` contract; feed M4's rejection traces back in (feedback loop); export to ONNX for the real-time path.

## Independent-work roadmap (while the data model is being set up)
1. **Define the K safety classes** for real (here: safe / stressed / critical) — the load-bearing decision.
2. **Publish `M2_TO_M3_CONTRACT.md` + the mock stream to Saabir** — unblocks his gate immediately.
3. **Agree the mock M1→M2 input shape with Zayan** so your adapter matches when his output lands.
4. **Harden the EDL core:** calibration (ECE, reliability diagram), epistemic/aleatoric split, latency profiling, ONNX export.
5. **Grow the synthetic scenario set** toward Burevi/Ditwah-like profiles for your paper's OOD evaluation.
