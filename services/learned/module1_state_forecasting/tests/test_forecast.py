"""The forecaster's guarantees: the ones a number in the report silently depends on.

A leaked split, a label that peeks at the future, or a graph edge between islands would each
inflate every result without failing anything else. These tests make those failures loud.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from module1.graph.build import build_graph, check_invariants  # noqa: E402
from module1.models.stgnn import EMBEDDING_DIM, LSTM1D, STGNN, persistence  # noqa: E402
from module1.train import dataset as ds  # noqa: E402
from module1.train import metrics as m  # noqa: E402
from module1.train import target as tgt  # noqa: E402

PROCESSED = Path(__file__).resolve().parents[4] / "data" / "processed"
needs_artifacts = pytest.mark.skipif(
    not (PROCESSED / "island_load_hourly.csv").exists(),
    reason="calibration artifacts absent -- run `task data`",
)


# ------------------------------------------------------------ graph

def test_graph_is_symmetric_connected_and_has_no_inter_island_edges() -> None:
    g = build_graph()
    check_invariants(g)
    assert g.n == 10
    assert g.node_ids[0] == "Eluvaitivu/bus"


def test_normalized_adjacency_is_the_kipf_welling_form() -> None:
    a_hat = build_graph().normalized()
    assert np.allclose(a_hat, a_hat.T)
    assert np.all(np.linalg.eigvalsh(a_hat) <= 1.0 + 1e-9)


def test_an_inter_island_edge_is_rejected() -> None:
    g = build_graph()
    bad = g.adjacency.copy()
    bad[0, 4] = bad[4, 0] = 1.0  # Eluvaitivu bus -- Analaitivu bus
    with pytest.raises(ValueError, match="inter-island"):
        check_invariants(type(g)(g.node_ids, g.node_island, g.node_asset, bad))


# ------------------------------------------------------------ target rules

def test_hazard_is_zero_below_median_and_saturates_at_the_top() -> None:
    ref = {"wind_50m_ms": (5.0, 15.0), "precip_mm_hr": (0.0, 50.0)}
    h = tgt.hazard(np.array([4.0, 15.0, 30.0]), np.array([0.0, 0.0, 0.0]), ref)
    assert h.tolist() == [0.0, 1.0, 1.0]


def test_either_hazard_alone_raises_exposure() -> None:
    ref = {"wind_50m_ms": (0.0, 10.0), "precip_mm_hr": (0.0, 10.0)}
    calm = tgt.hazard(np.array([0.0]), np.array([0.0]), ref)
    rain_only = tgt.hazard(np.array([0.0]), np.array([5.0]), ref)
    both = tgt.hazard(np.array([5.0]), np.array([5.0]), ref)
    assert calm[0] == 0.0 < rain_only[0] < both[0] <= 1.0


def test_only_hybrid_assets_inherit_the_ledger_decay() -> None:
    decay = np.array([0.0, 0.9])
    assert tgt.fragility("bess", decay).tolist() == [0.5, 0.9]
    assert tgt.fragility("diesel", decay).tolist() == [0.4, 0.4]


def test_every_weighting_sums_to_one() -> None:
    for w in tgt.WEIGHTINGS.values():
        assert sum(w) == pytest.approx(1.0)


# ------------------------------------------------------------ data and split

@needs_artifacts
def test_splits_are_disjoint_ordered_and_hold_out_both_real_events() -> None:
    d = ds.build()
    tr, va, te = (d.split_index[k] for k in ("train", "val", "test"))
    assert tr[1] <= va[0] and va[1] <= te[0]
    test_stamps = d.timestamps[te[0]:te[1]]
    assert "2025-11-28T13:00" in test_stamps          # Cyclone Ditwah peak wind
    assert test_stamps[0][:7] == "2025-10"           # first month of the hybrid decay


@needs_artifacts
def test_windows_never_cross_a_split_boundary() -> None:
    d = ds.build()
    for split in ("train", "val", "test"):
        a, b = d.split_index[split]
        _, _, origins = ds.windows(d, split, "equal", 24, 6)
        assert origins.min() - 23 >= a
        assert origins.max() + 6 < b


@needs_artifacts
def test_fitted_scales_come_from_training_rows_only() -> None:
    """Changing the test period must not move the hazard bounds or the load peak."""
    d = ds.build()
    a, b = d.split_index["train"]
    wind = np.array([float(r) for r in _column("Nainativu", "wind_50m_ms")])[a:b]
    assert d.fitted["hazard_ref"]["Nainativu"]["wind_50m_ms"][1] == pytest.approx(
        float(np.percentile(wind, 99.9)))


@needs_artifacts
def test_the_label_uses_last_months_ledger_not_this_months() -> None:
    """The decay is visible in the Eluvaitivu battery's fragility a month late, by design."""
    d = ds.build()
    j = d.graph.node_ids.index("Eluvaitivu/bess")
    frag = d.components["fragility"][:, j]
    oct_ = frag[d.timestamps == "2025-10-15T12:00"][0]
    dec_ = frag[d.timestamps == "2025-12-15T12:00"][0]
    assert oct_ == pytest.approx(tgt.BASE_FRAGILITY["bess"])  # September was healthy
    assert dec_ > 0.8                                          # November had collapsed


@needs_artifacts
def test_feature_tensor_is_finite_and_shaped_to_the_graph() -> None:
    d = ds.build()
    assert d.x.shape == (len(d.timestamps), d.graph.n, len(ds.FEATURES))
    assert np.isfinite(d.x).all()


def _column(island: str, col: str) -> list[str]:
    import csv
    path = PROCESSED.parent / "raw" / "nasa_power" / f"{island}_hourly.csv"
    with path.open(newline="") as fh:
        return [r[col] for r in csv.DictReader(fh)]


# ------------------------------------------------------------ models

def _toy(b: int = 3, seq_len: int = 24, n: int = 10, f: int = len(ds.FEATURES)):
    g = torch.Generator().manual_seed(0)
    return torch.rand(b, seq_len, n, f, generator=g)


def test_embedding_width_is_the_contract_pin() -> None:
    """The node embedding M2 consumes is sized by module1_state_v1.json, not by this module."""
    from metacore_contracts.state_schema import EMBEDDING_DIM as PIN
    from module1.models import stgnn
    from module1.train.run import CONFIG
    assert stgnn.EMBEDDING_DIM is PIN
    assert CONFIG["hidden"] == PIN
    a_hat = torch.tensor(build_graph().normalized(), dtype=torch.float32)
    model = STGNN(len(ds.FEATURES), 6, a_hat, gcn_layers=1, residual=True)
    assert model.embed(_toy(b=1)).shape[-1] == PIN


def test_stgnn_shapes_bounds_and_embedding_width() -> None:
    a_hat = torch.tensor(build_graph().normalized(), dtype=torch.float32)
    model = STGNN(len(ds.FEATURES), 6, a_hat)
    x = _toy()
    y = model(x)
    assert y.shape == (3, 10, 6)
    assert torch.all((y >= 0) & (y <= 1))
    assert model.embed(x).shape == (3, 10, EMBEDDING_DIM)


def test_stgnn_is_deterministic_under_a_fixed_seed() -> None:
    a_hat = torch.tensor(build_graph().normalized(), dtype=torch.float32)
    outs = []
    for _ in range(2):
        torch.manual_seed(7)
        outs.append(STGNN(len(ds.FEATURES), 6, a_hat)(_toy()))
    assert torch.equal(outs[0], outs[1])


def test_residual_head_starts_from_the_last_observed_score_and_stays_in_range() -> None:
    a_hat = torch.tensor(build_graph().normalized(), dtype=torch.float32)
    torch.manual_seed(0)
    model = STGNN(len(ds.FEATURES), 6, a_hat, gcn_layers=1, residual=True)
    torch.nn.init.zeros_(model.head.weight)
    torch.nn.init.zeros_(model.head.bias)
    x = _toy()
    y = model(x)
    assert torch.allclose(y, x[:, -1, :, -1].unsqueeze(-1).expand_as(y))
    assert model.gcn2 is None


def test_graph_free_variant_cannot_mix_islands_or_nodes() -> None:
    """With message passing off, a node's forecast depends only on its own inputs."""
    a_hat = torch.tensor(build_graph().normalized(), dtype=torch.float32)
    torch.manual_seed(0)
    model = STGNN(len(ds.FEATURES), 6, a_hat, use_graph=False).eval()
    x = _toy(b=1)
    x2 = x.clone()
    x2[:, :, 5] += 1.0  # perturb one node
    with torch.no_grad():
        d = (model(x) - model(x2)).abs().sum(dim=-1)[0]
    assert d[5] > 0 and torch.all(d[torch.arange(10) != 5] == 0)


def test_residual_lstm_starts_from_the_last_score_like_the_stgnn() -> None:
    """The fair baseline: same output parameterisation as ST-GNN v0.1, so the comparison is
    about architecture, not about how the head is parameterised."""
    torch.manual_seed(0)
    model = LSTM1D(6, residual=True).eval()
    torch.nn.init.zeros_(model.head.weight)
    torch.nn.init.zeros_(model.head.bias)
    x = _toy()
    with torch.no_grad():
        y = model(x)
    assert torch.allclose(y, x[:, -1, :, -1].unsqueeze(-1).expand_as(y))


def test_persistence_and_lstm_read_only_the_history_column() -> None:
    x = _toy()
    p = persistence(x, 6)
    assert torch.equal(p[..., 0], x[:, -1, :, -1])
    torch.manual_seed(0)
    model = LSTM1D(6).eval()
    x2 = x.clone()
    x2[..., :-1] = 0.0  # wipe every other feature
    with torch.no_grad():
        assert torch.equal(model(x), model(x2))


# ------------------------------------------------------------ inference

@needs_artifacts
def test_inference_reports_split_ranking_and_keeps_scores_in_range() -> None:
    from module1.infer.forecast import forecast
    d = ds.build()
    a_hat = torch.tensor(d.graph.normalized(), dtype=torch.float32)
    torch.manual_seed(0)
    model = STGNN(len(ds.FEATURES), 6, a_hat, gcn_layers=1, residual=True).eval()
    out = forecast(model, d, "2025-11-28T08:00", "equal")
    assert out["split"] == "test"
    assert len(out["nodes"]) == d.graph.n
    assert all(0.0 <= z <= 1.0 for n in out["nodes"] for z in n["forecast"])
    assert out["nodes"][0]["history"][-1] == out["nodes"][0]["current"]
    assert forecast(model, d, "2024-06-15T18:00", "equal")["split"] == "train"


@needs_artifacts
def test_inference_refuses_hours_without_history_or_outside_the_record() -> None:
    from module1.infer.forecast import forecast
    d = ds.build()
    a_hat = torch.tensor(d.graph.normalized(), dtype=torch.float32)
    model = STGNN(len(ds.FEATURES), 6, a_hat, gcn_layers=1, residual=True).eval()
    with pytest.raises(ValueError, match="history"):
        forecast(model, d, "2024-01-01T05:00", "equal")
    with pytest.raises(ValueError, match="not an hour"):
        forecast(model, d, "2027-01-01T00:00", "equal")


def test_missing_checkpoint_names_the_command_that_makes_it(tmp_path) -> None:
    from module1.infer.forecast import CheckpointMissing, load_model
    with pytest.raises(CheckpointMissing, match="module1.train.run"):
        load_model(None, "equal", tmp_path)


# ------------------------------------------------------------ metrics

def test_event_detection_counts() -> None:
    true = np.array([[[0.9]], [[0.1]], [[0.9]]])
    pred = np.array([[[0.9]], [[0.9]], [[0.1]]])
    ev = m.event_detection(pred, true, 0.5)
    assert ev["precision"] == [0.5] and ev["recall"] == [0.5]


def test_onset_lead_time_finds_the_earliest_warning() -> None:
    v = np.zeros((10, 1))
    v[6:, 0] = 1.0                       # onset at t0 = 6
    origins = np.arange(0, 9)
    pred = np.zeros((9, 1, 3))
    pred[3, 0, 2] = 1.0                  # issued at 3, horizon 3 -> covers t0
    out = m.onset_lead_time(pred, v, origins, 0.5)
    assert out == {"onsets": 1, "onset_recall": 1.0, "mean_lead_h": 3.0}


def test_mase_is_one_for_a_forecast_as_good_as_the_training_naive() -> None:
    v_train = np.array([[0.0], [0.1], [0.0], [0.1]])   # one-step naive MAE = 0.1
    scale = m.naive_scale(v_train)
    true = np.zeros((4, 1, 2))
    out = m.mase(true + 0.1, true, scale)
    assert scale == pytest.approx(0.1)
    assert out["mase"] == pytest.approx(1.0) and out["mase_by_h"] == pytest.approx([1.0, 1.0])


def test_diebold_mariano_sign_and_strength() -> None:
    rng = np.random.default_rng(0)
    true = rng.random((300, 4, 6))
    good = true + rng.normal(0, 0.01, true.shape)
    bad = true + rng.normal(0, 0.10, true.shape)
    better = m.diebold_mariano(good, bad, true)
    assert better["dm"] < 0 and better["p_value"] < 1e-6
    assert m.diebold_mariano(bad, good, true)["dm"] == pytest.approx(-better["dm"])
    assert m.diebold_mariano(good, good, true)["mean_loss_diff"] == 0.0


@needs_artifacts
def test_ditwah_window_lies_in_the_test_quarter_and_is_stormy() -> None:
    from module1.train.run import EVENT_WINDOWS, regimes

    data = ds.build()
    a, b = data.split_index["test"]
    origins = np.arange(a, b - 1)
    masks = regimes(data, origins)
    start, end = EVENT_WINDOWS["ditwah"]
    assert ds.SPLITS["test"][0] <= start and end <= "2026"
    assert masks["ditwah"].sum() == 144                       # 25-30 Nov, hourly
    assert masks["storm"][masks["ditwah"]].mean() > 0.9       # the cyclone is in the data
    assert not (masks["normal"] & masks["storm"]).any()
