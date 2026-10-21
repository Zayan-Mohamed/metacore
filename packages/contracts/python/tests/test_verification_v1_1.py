"""verification.proto v1.1 is additive: a v1.0 reader still parses a v1.1 verdict.

The v1.0 descriptor is frozen in fixtures/verification_v1_0.binpb (taken from the committed
v1.0 stub). It is loaded into its own descriptor pool, beside the unchanged common/module1/
module3 files, so the two versions of the same package never collide.
"""

from __future__ import annotations

from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from metacore_contracts import common_pb2, module1_pb2, module3_pb2, verification_pb2

FIXTURE = Path(__file__).parent / "fixtures" / "verification_v1_0.binpb"


def _add_with_deps(pool: descriptor_pool.DescriptorPool, file_desc, seen: set[str]) -> None:
    if file_desc.name in seen:
        return
    for dep in file_desc.dependencies:
        _add_with_deps(pool, dep, seen)
    proto = descriptor_pb2.FileDescriptorProto()
    file_desc.CopyToProto(proto)
    pool.Add(proto)
    seen.add(file_desc.name)


def _v1_0_verdict_class():
    pool = descriptor_pool.DescriptorPool()
    seen: set[str] = set()
    for dep in (common_pb2, module1_pb2, module3_pb2):
        _add_with_deps(pool, dep.DESCRIPTOR, seen)
    pool.Add(descriptor_pb2.FileDescriptorProto.FromString(FIXTURE.read_bytes()))
    desc = pool.FindMessageTypeByName("metacore.verification.v1.VerificationVerdict")
    return message_factory.GetMessageClass(desc)


def _v1_1_verdict() -> verification_pb2.VerificationVerdict:
    v = verification_pb2.VerificationVerdict(
        action_id="bess-export-40",
        decision=verification_pb2.DECISION_UNVERIFIED,
        solve_latency_ms=130.5,
        spec_version="safety-spec-v1.0",
        network_model_version="northern-islands-v1",
        limit_profile="METACORE-STRICT-v1",
        unverified_reasons=["risk bound straddles the target"],
    )
    v.violations.add(type=verification_pb2.VIOLATION_TYPE_GENERATION_INADEQUATE,
                     element_id="ELU_PS", measured=141.0, limit=130.0, margin_fraction=0.0846)
    c = v.certificate
    c.decision = verification_pb2.DECISION_UNVERIFIED
    c.n_scenarios, c.n_max, c.n_failures = 688, 688, 3
    c.lower_bound, c.upper_bound = 0.00028, 0.01885
    c.seed = 20251128
    c.failures_by_island["Eluvaitivu"] = 3
    v.attribution.add(player="Eluvaitivu", island="Eluvaitivu", buses=["ELU_F1", "ELU_F2"],
                      share=0.00436, fraction_of_risk=1.0, alone=0.00436)
    return v


def test_v1_1_round_trips() -> None:
    v = _v1_1_verdict()
    again = verification_pb2.VerificationVerdict.FromString(v.SerializeToString())
    assert again == v
    assert again.certificate.failures_by_island["Eluvaitivu"] == 3
    assert again.attribution[0].player == "Eluvaitivu"


def test_a_v1_0_reader_parses_a_v1_1_verdict() -> None:
    old_cls = _v1_0_verdict_class()
    old = old_cls.FromString(_v1_1_verdict().SerializeToString())
    assert old.action_id == "bess-export-40"
    assert old.solve_latency_ms == 130.5
    assert old.network_model_version == "northern-islands-v1"
    # UNVERIFIED is unknown to v1.0: proto3 keeps the number, and 3 is not APPROVE (1)
    assert old.decision == 3
    assert old.decision != verification_pb2.DECISION_APPROVE
    assert old.violations[0].element_id == "ELU_PS"
    # The v1.1 fields ride along untouched and survive a v1.0 re-serialization
    resent = verification_pb2.VerificationVerdict.FromString(old.SerializeToString())
    assert resent.certificate.n_failures == 3


def test_v1_0_numbering_is_unchanged() -> None:
    fields = {f.name: f.number
              for f in verification_pb2.VerificationVerdict.DESCRIPTOR.fields}
    assert {k: fields[k] for k in ("envelope", "action_id", "decision", "violations",
                                   "solve_latency_ms", "spec_version",
                                   "network_model_version")} == {
        "envelope": 1, "action_id": 2, "decision": 3, "violations": 4,
        "solve_latency_ms": 5, "spec_version": 6, "network_model_version": 7}
    assert verification_pb2.DECISION_APPROVE == 1 and verification_pb2.DECISION_REJECT == 2
    assert verification_pb2.DECISION_UNVERIFIED == 3
