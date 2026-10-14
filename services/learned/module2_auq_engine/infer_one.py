"""One-shot M2 scoring for the dashboard: two knobs in, one JSON object out.

    python infer_one.py --novelty 0.85 --observed-fraction 0.30

This is the real engine, not a simulation. It reuses the module's own functions rather
than reimplementing the maths:

  synthetic_data.sample_states_id / sample_states_ood   the ID and OOD draws
  infer.OnnxAUQ                                         the exported evidence graph
  edl.EDLNet / baselines.train_edl                      torch fallback when onnxruntime
                                                        is absent (the `onnx` extra)
  edl.uncertainty / uncertainty_quality                 u = K/S and the quality discount
  trigger.CompetenceDropTrigger                         the two-condition trigger + reason
  baselines.score_softmax / score_mc_dropout            the comparison baselines

`novelty` interpolates the query state from an in-distribution exemplar toward an
out-of-distribution (cyclone) one, so the slider sweeps a real path through feature space
instead of a synthetic curve. `observed_fraction` is passed straight through as M1's
QualityMask ratio would be.

Everything is seeded from config.yaml, so the same two inputs always produce the same
JSON. The fixture (normalisation, baseline models, calibrated value threshold) is trained
once and cached in .dashboard_cache/ — the first call pays for it, later calls do not.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from baselines import (
    DROPOUT_P,
    OOD_PROXY_SIGMA,
    score_mc_dropout,
    score_softmax,
    train_edl,
    train_softmax,
)
from edl import uncertainty as edl_uncertainty
from edl import uncertainty_quality
from state_contract import stack_features
from synthetic_data import Normalizer, sample_states_id, sample_states_ood
from trigger import CompetenceDropTrigger

HERE = Path(__file__).parent
CACHE_DIR = HERE / ".dashboard_cache"
# Bump when anything below changes shape, so a stale cache is never silently reused.
FIXTURE_VERSION = 1

# Smaller than benchmark.py's 3000/800: this fixture exists to give the dashboard a
# normaliser, two baseline models and a calibrated threshold, not to produce the paper's
# table. benchmark.py remains the full-scale run.
N_TRAIN = 1200
N_OOD = 300
N_CALIBRATE = 400

# One-shot scoring has no history, so a debounce of 2 could never fire. The hysteresis in
# config.yaml is for the streaming path; here the trigger must answer on a single sample.
ONESHOT_HYSTERESIS = 1

# novelty = 1 lands this many ID standard deviations toward the cyclone regime. Reusing
# the module's far-OOD proxy spread rather than inventing a constant — see _query_state.
MAX_SIGMA = OOD_PROXY_SIGMA

LATENCY_REPEATS = 20


def _load_config() -> dict:
    with open(HERE / "config.yaml") as fh:
        return yaml.safe_load(fh)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_fixture(cfg: dict) -> dict:
    """Normaliser, ID/OOD exemplars and the two baseline models. Cached on disk."""
    seed = int(cfg["seed"])
    epochs = int(cfg["train"]["epochs"])
    cache_path = CACHE_DIR / f"fixture-v{FIXTURE_VERSION}-s{seed}-e{epochs}.pt"

    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    _seed_everything(seed)
    rng = np.random.default_rng(seed)

    train_states, y_train = sample_states_id(N_TRAIN, rng, blackout_rate=0.0)
    ood_states, _ = sample_states_ood(N_OOD, rng)

    x_train_raw = stack_features(train_states)
    x_ood_raw = stack_features(ood_states)

    normalizer = Normalizer().fit(x_train_raw)
    x_train = normalizer(x_train_raw)

    softmax_model = train_softmax(x_train, y_train, epochs, seed=seed)
    mc_model = train_softmax(x_train, y_train, epochs, dropout=DROPOUT_P, seed=seed)

    fixture = {
        "version": FIXTURE_VERSION,
        "seed": seed,
        "epochs": epochs,
        # Mean exemplars rather than single draws: the interpolation should sweep between
        # the centres of the two regimes, not between two arbitrary samples.
        "id_exemplar": x_train_raw.mean(0),
        "ood_exemplar": x_ood_raw.mean(0),
        "calibration_raw": x_train_raw[:N_CALIBRATE],
        "normalizer_mu": normalizer.mu,
        "normalizer_sd": normalizer.sd,
        "softmax_state": softmax_model.state_dict(),
        "mc_state": mc_model.state_dict(),
        "n_classes": int(y_train.max()) + 1,
        "n_features": int(x_train_raw.shape[1]),
    }

    CACHE_DIR.mkdir(exist_ok=True)
    torch.save(fixture, cache_path)
    return fixture


def _restore_baselines(fixture: dict):
    from baselines import MLP

    d_in, k = fixture["n_features"], fixture["n_classes"]
    softmax_model = MLP(d_in, k)
    softmax_model.load_state_dict(fixture["softmax_state"])
    softmax_model.eval()

    mc_model = MLP(d_in, k, dropout=DROPOUT_P)
    mc_model.load_state_dict(fixture["mc_state"])
    mc_model.eval()
    return softmax_model, mc_model


class _TorchEngine:
    """Fallback evidence source when the `onnx` extra is not installed.

    Trains the same EDL head baselines.py trains, on the same seeded data, and applies the
    fixture's normalisation. Cached alongside the fixture.
    """

    backend = "torch"

    def __init__(self, model, mu, sd):
        self._model = model
        self._mu = mu
        self._sd = sd

    @classmethod
    def load(cls, cfg: dict, fixture: dict):
        seed = int(cfg["seed"])
        epochs = int(cfg["train"]["epochs"])
        cache_path = CACHE_DIR / f"edl-v{FIXTURE_VERSION}-s{seed}-e{epochs}.pt"

        from edl import EDLNet

        model = EDLNet(fixture["n_features"], fixture["n_classes"])
        if cache_path.exists():
            model.load_state_dict(torch.load(cache_path, weights_only=True))
        else:
            _seed_everything(seed)
            rng = np.random.default_rng(seed)
            states, y = sample_states_id(N_TRAIN, rng, blackout_rate=0.0)
            x_raw = stack_features(states)
            x = ((x_raw - fixture["normalizer_mu"]) / fixture["normalizer_sd"]).astype(np.float32)
            model = train_edl(x, y, epochs, ood_reg=cfg["train"]["ood_reg_weight"], seed=seed)
            CACHE_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), cache_path)
        model.eval()
        return cls(model, fixture["normalizer_mu"], fixture["normalizer_sd"])

    def evidence(self, features_raw: np.ndarray) -> np.ndarray:
        x = ((np.asarray(features_raw, dtype=np.float32).reshape(1, -1) - self._mu) / self._sd)
        with torch.no_grad():
            return self._model(torch.as_tensor(x.astype(np.float32))).numpy()


class _OnnxEngine:
    """The exported evidence graph — the same path the real-time gate would use."""

    backend = "onnx"

    def __init__(self, auq):
        self._auq = auq

    @classmethod
    def load(cls):
        from infer import OnnxAUQ

        # value_threshold is calibrated below; pass a placeholder so load() does not need one.
        return cls(OnnxAUQ.load(value_threshold=float("inf")))

    def evidence(self, features_raw: np.ndarray) -> np.ndarray:
        return self._auq.evidence(features_raw)


def _make_engine(cfg: dict, fixture: dict):
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return _TorchEngine.load(cfg, fixture)
    if not (HERE / "edl.onnx").exists():
        return _TorchEngine.load(cfg, fixture)
    return _OnnxEngine.load()


def _query_state(fixture: dict, novelty: float) -> np.ndarray:
    """Walk from the in-distribution centre toward the cyclone regime.

    Measured in in-distribution standard deviations, not as a raw-space lerp between the
    two centroids. The cyclone centroid sits ~20 sigma per feature away, so a plain lerp
    puts the entire ID -> OOD transition inside the first 20% of the slider and leaves the
    rest saturated at u = 1.

    novelty = 1 lands at OOD_PROXY_SIGMA, the same spread baselines.py uses for the far-OOD
    proxy the evidence regulariser is trained against — so the slider sweeps exactly the
    range the model was regularised over.
    """
    mu, sd = fixture["normalizer_mu"], fixture["normalizer_sd"]
    id_std = (fixture["id_exemplar"] - mu) / sd
    ood_std = (fixture["ood_exemplar"] - mu) / sd

    direction = ood_std - id_std
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return fixture["id_exemplar"]
    unit = direction / norm

    # novelty * MAX_SIGMA expressed as a per-feature RMS displacement.
    displacement = novelty * MAX_SIGMA * np.sqrt(len(unit))
    return (id_std + unit * displacement) * sd + mu


def _value_u(evidence: np.ndarray) -> float:
    """Plain, quality-blind u = K/S, via the module's own torch implementation."""
    u, _, _ = edl_uncertainty(torch.as_tensor(evidence))
    return float(u.item())


def _calibrate_threshold(engine, fixture: dict, false_alarm_rate: float, of_floor: float) -> float:
    """Fit the value threshold on plain u over in-distribution states (see trigger.py)."""
    id_u = np.array([_value_u(engine.evidence(row)) for row in fixture["calibration_raw"]])
    return CompetenceDropTrigger.calibrate(
        id_u, false_alarm_rate, of_floor, ONESHOT_HYSTERESIS
    ).vthr


def score(novelty: float, observed_fraction: float) -> dict:
    cfg = _load_config()
    _seed_everything(int(cfg["seed"]))

    fixture = _build_fixture(cfg)
    engine = _make_engine(cfg, fixture)

    of_floor = float(cfg["trigger"]["observed_fraction_floor"])
    value_threshold = _calibrate_threshold(
        engine, fixture, float(cfg["trigger"]["value_false_alarm_rate"]), of_floor
    )

    n = float(np.clip(novelty, 0.0, 1.0))
    f = float(np.clip(observed_fraction, 0.0, 1.0))
    query_raw = _query_state(fixture, n)

    evidence = engine.evidence(query_raw)
    evidence_t = torch.as_tensor(evidence)

    u = _value_u(evidence)
    u_q = float(uncertainty_quality(evidence_t, f).item())
    # S = sum(alpha) = sum(evidence) + K — the total evidence mass the page reports.
    total_evidence = float(evidence.sum()) + evidence.shape[1]

    fired, reason = CompetenceDropTrigger(
        value_threshold, of_floor, ONESHOT_HYSTERESIS
    ).update(u, f)

    # Real single-pass latency for the evidence call, best-of after a warm-up.
    engine.evidence(query_raw)
    best = float("inf")
    for _ in range(LATENCY_REPEATS):
        start = time.perf_counter()
        engine.evidence(query_raw)
        best = min(best, time.perf_counter() - start)
    latency_ms = best * 1000.0

    softmax_model, mc_model = _restore_baselines(fixture)
    query_norm = (
        (query_raw - fixture["normalizer_mu"]) / fixture["normalizer_sd"]
    ).astype(np.float32).reshape(1, -1)
    softmax_score, _ = score_softmax(softmax_model, query_norm)
    mc_score, _ = score_mc_dropout(mc_model, query_norm, seed=int(cfg["seed"]))

    return {
        "u": u,
        "u_q": u_q,
        "evidence": total_evidence,
        "observed_fraction": f,
        "trigger": bool(fired),
        "reason": reason,
        "latency_ms": latency_ms,
        "baselines": {
            "softmax": float(softmax_score[0]),
            "mc_dropout": float(mc_score[0]),
            "edl": u_q,
        },
        "backend": engine.backend,
        "value_threshold": value_threshold,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="One-shot M2 uncertainty scoring.")
    parser.add_argument("--novelty", type=float, required=True, help="0..1 distribution shift")
    parser.add_argument(
        "--observed-fraction", type=float, required=True, help="0..1 share of features measured"
    )
    args = parser.parse_args()
    print(json.dumps(score(args.novelty, args.observed_fraction)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
