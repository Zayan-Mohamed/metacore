"""Evaluation metrics for the vulnerability forecaster (proposal Table 2).

All functions take ``pred`` and ``true`` shaped (B, N, H): forecast origin, node, horizon.
"""

from __future__ import annotations

import math

import numpy as np


def error_by_horizon(pred: np.ndarray, true: np.ndarray) -> dict[str, list[float]]:
    err = pred - true
    return {
        "rmse": [float(np.sqrt(np.mean(err[..., h] ** 2))) for h in range(err.shape[-1])],
        "mae": [float(np.mean(np.abs(err[..., h]))) for h in range(err.shape[-1])],
    }


def naive_scale(v_train: np.ndarray) -> float:
    """MASE denominator: in-sample MAE of the one-step naive forecast on the training split
    (Hyndman and Koehler, 2006). ``v_train`` is the (T, N) label series."""
    return float(np.mean(np.abs(np.diff(v_train, axis=0))))


def mase(pred: np.ndarray, true: np.ndarray, scale: float) -> dict:
    """Mean absolute scaled error, overall and per horizon. Below 1 beats the one-step naive
    forecast's training error; multi-step horizons are harder, so persistence itself exceeds 1."""
    err = np.abs(pred - true) / scale
    return {"mase": float(err.mean()),
            "mase_by_h": [float(err[..., h].mean()) for h in range(err.shape[-1])]}


def diebold_mariano(pred_a: np.ndarray, pred_b: np.ndarray, true: np.ndarray) -> dict:
    """Is model A's squared error lower than B's by more than chance? Diebold and Mariano (1995)
    on the per-origin loss differential, with the Harvey, Leybourne and Newbold (1997)
    small-sample correction.

    Origins are hourly and forecasts overlap, so the loss is autocorrelated: the long-run variance
    uses a Bartlett (Newey-West) kernel with lag max(H - 1, 4 (T/100)^(2/9)). The p-value is
    two-sided from the normal; with T in the hundreds the t reference differs in the third decimal.
    Negative ``dm`` means A is better.
    """
    d = (((pred_a - true) ** 2).mean(axis=(1, 2)) - ((pred_b - true) ** 2).mean(axis=(1, 2)))
    t = len(d)
    lag = max(pred_a.shape[-1] - 1, int(4 * (t / 100) ** (2 / 9)))
    dc = d - d.mean()
    lrv = float(dc @ dc) / t
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * float(dc[k:] @ dc[:-k]) / t
    if lrv <= 0:  # identical forecasts: no difference to test
        return {"dm": 0.0, "p_value": 1.0, "mean_loss_diff": float(d.mean()), "n": t, "lag": lag}
    stat = d.mean() / math.sqrt(lrv / t)
    h = pred_a.shape[-1]
    stat *= math.sqrt((t + 1 - 2 * h + h * (h - 1) / t) / t)
    return {"dm": float(stat), "p_value": float(math.erfc(abs(stat) / math.sqrt(2))),
            "mean_loss_diff": float(d.mean()), "n": t, "lag": lag}


def event_detection(pred: np.ndarray, true: np.ndarray, tau: float) -> dict[str, list[float]]:
    """Per-horizon precision / recall / F1 for 'node is in a high-vulnerability state'."""
    out: dict[str, list[float]] = {"precision": [], "recall": [], "f1": [], "positives": []}
    for h in range(pred.shape[-1]):
        p, t = pred[..., h] > tau, true[..., h] > tau
        tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out["precision"].append(prec)
        out["recall"].append(rec)
        out["f1"].append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        out["positives"].append(int(t.sum()))
    return out


def onset_lead_time(pred: np.ndarray, v: np.ndarray, origins: np.ndarray, tau: float) -> dict:
    """For every onset (node crosses tau upward at hour t0), the largest horizon h whose forecast,
    issued at t0 - h, already exceeded tau for t0. Zero means the onset was missed.

    ``v`` is the full (T, N) label series; ``origins`` maps each forecast row to its issue hour.
    """
    horizon = pred.shape[-1]
    row_of = {int(t): i for i, t in enumerate(origins)}
    leads = []
    for n in range(v.shape[1]):
        for t0 in range(1, v.shape[0]):
            if not (v[t0, n] > tau >= v[t0 - 1, n]):
                continue
            if any((t0 - h) not in row_of for h in range(1, horizon + 1)):
                continue  # onset not fully covered by test-split forecasts
            lead = 0
            for h in range(horizon, 0, -1):
                if pred[row_of[t0 - h], n, h - 1] > tau:
                    lead = h
                    break
            leads.append(lead)
    leads_a = np.array(leads)
    caught = leads_a[leads_a > 0]
    return {
        "onsets": int(len(leads_a)),
        "onset_recall": float((leads_a > 0).mean()) if len(leads_a) else 0.0,
        "mean_lead_h": float(caught.mean()) if len(caught) else 0.0,
    }


def spatial_hit_rate(pred: np.ndarray, true: np.ndarray, tau: float, k: int, h: int) -> dict:
    """At origins where some node is truly above tau at horizon h: is the truly most vulnerable
    node among the model's top-k?"""
    hits, total = 0, 0
    for i in range(pred.shape[0]):
        t = true[i, :, h]
        if t.max() <= tau:
            continue
        total += 1
        top_k = np.argsort(-pred[i, :, h])[:k]
        hits += int(int(np.argmax(t)) in top_k)
    return {"hit_rate": hits / total if total else 0.0, "cases": total}
