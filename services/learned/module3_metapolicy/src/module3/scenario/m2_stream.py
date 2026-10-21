"""Loader for Module 2's already-published mock stream (sample_m2_to_m3.jsonl).

Does not import M2's contract.py - it parses the JSON dicts directly. `jitter` adds small noise
to a published row so the cyclone env can draw an arbitrary-length stream (not new M2 data).

Consumes m2-out/0.3 (current on m2/auq-engine). Older 0.1/0.2 mocks still load;
missing trigger_reason / observed_fraction are derived or defaulted.
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSIONS_ACCEPTED = {"m2-out/0.1", "m2-out/0.2", "m2-out/0.3"}
SCHEMA_VERSION_EXPECTED = "m2-out/0.3"
TRIGGER_REASONS = ("none", "value", "sensing", "both")

# Mirrors module2_auq_engine/config.yaml's trigger.observed_fraction_floor (0.35, keyed
# off the real 28-feature nominal of 12/28=0.4286 -- see M2_TO_M3_CONTRACT.md). Keep in
# sync with M2's config; a stale value here silently mislabels the boundary band.
OF_FLOOR = 0.35

# Prefer a local vendored copy so this starter runs on main before M2 merges;
# fall back to the sibling Module 2 path when that branch is present.
_MODULE_ROOT = Path(__file__).resolve().parents[3]  # .../module3_metapolicy
_LOCAL_SAMPLE = _MODULE_ROOT / "sample_m2_to_m3.jsonl"
_SIBLING_SAMPLE = _MODULE_ROOT.parent / "module2_auq_engine" / "sample_m2_to_m3.jsonl"
_DEFAULT_PATH = _LOCAL_SAMPLE if _LOCAL_SAMPLE.is_file() else _SIBLING_SAMPLE


def load_jsonl(path=None) -> list[dict]:
    """Read M2→M3 JSONL; assert schema_version and required gating fields."""
    p = Path(path) if path else _DEFAULT_PATH
    records: list[dict] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ver = rec.get("schema_version")
            if ver not in SCHEMA_VERSIONS_ACCEPTED:
                raise ValueError(
                    f"unexpected schema_version {ver!r}; "
                    f"accepted {sorted(SCHEMA_VERSIONS_ACCEPTED)}"
                )
            for key in ("epistemic_uncertainty", "competence_drop",
                        "state_class", "class_probabilities"):
                if key not in rec:
                    raise ValueError(f"M2 record missing required field {key!r}")
            # Normalise additive 0.2/0.3 fields so every record is gate-ready.
            if "observed_fraction" not in rec:
                rec["observed_fraction"] = 1.0
            if "trigger_reason" not in rec:
                of = float(rec["observed_fraction"])
                if of < OF_FLOOR:
                    rec["trigger_reason"] = "sensing"
                elif bool(rec["competence_drop"]):
                    rec["trigger_reason"] = "value"
                else:
                    rec["trigger_reason"] = "none"
            records.append(rec)
    if not records:
        raise ValueError(f"no records in {p}")
    return records


def by_reason(records: list[dict]) -> dict[str, list[dict]]:
    buckets = {r: [] for r in TRIGGER_REASONS}
    for rec in records:
        reason = str(rec.get("trigger_reason", "none"))
        if reason not in buckets:
            reason = "none"
        buckets[reason].append(rec)
    none_or_all = buckets["none"] or records
    for r in ("none", "value", "sensing"):
        if not buckets[r]:
            buckets[r] = none_or_all
    # Sample file may omit ``both``; derive from value rows + sensing-floor of.
    if not buckets["both"]:
        src = [r for r in records if str(r.get("trigger_reason")) == "value"] or records
        derived = []
        for r in src:
            b = dict(r)
            b["trigger_reason"] = "both"
            b["competence_drop"] = True
            b["observed_fraction"] = min(float(r.get("observed_fraction", 0.5)), 0.35)
            derived.append(b)
        buckets["both"] = derived
    return buckets


def jitter(base: dict, rng) -> dict:
    u = float(base["epistemic_uncertainty"]) + float(rng.normal(0, 0.03))
    u = float(max(0.0, min(1.0, u)))
    probs = [float(p) for p in base["class_probabilities"]]
    probs = [max(1e-6, p + float(rng.normal(0, 0.02))) for p in probs]
    s = sum(probs)
    probs = [p / s for p in probs]
    state_class = int(max(range(len(probs)), key=lambda i: probs[i]))

    reason = str(base.get("trigger_reason", "none"))
    of = float(base.get("observed_fraction", 1.0))
    of = float(np_clip(of + float(rng.normal(0, 0.02)), 0.0, 1.0))

    # Preserve axis identity from the published mock; only the value axis
    # re-derives competence_drop from jittered u.
    if reason == "sensing":
        competence_drop = True
        of = min(of, OF_FLOOR - 0.01)  # stay under the sensing floor
    elif reason == "both":
        competence_drop = True
        of = min(of, OF_FLOOR - 0.01)
    elif reason == "value":
        competence_drop = bool(u > 0.45)
        if not competence_drop:
            reason = "none"
    else:
        competence_drop = False
        reason = "none"

    return {
        "timestamp": float(base.get("timestamp", 0.0)),
        "epistemic_uncertainty": u,
        "aleatoric_proxy": float(base.get("aleatoric_proxy", 0.0)),
        "competence_drop": competence_drop,
        "trigger_reason": reason,
        "state_class": state_class,
        "class_probabilities": probs,
        "observed_fraction": of,
        "schema_version": str(base.get("schema_version", SCHEMA_VERSION_EXPECTED)),
    }


def np_clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x
