"""M2 -> M3 output contract (see master plan register). Saabir's gate (M3 FR1)
builds against mock_stream() until this module's real output is live."""
from dataclasses import dataclass, asdict
import json, time

SCHEMA_VERSION = "m2-out/0.1"

@dataclass
class M2Output:
    timestamp: float
    epistemic_uncertainty: float      # u = K/S in [0,1]  (the gating signal)
    aleatoric_proxy: float            # predictive entropy
    competence_drop: bool             # trigger for M3 to escalate to System 2
    state_class: int                  # argmax safety class (0 safe..K-1 critical)
    class_probabilities: list         # Dirichlet mean p over the K classes
    schema_version: str = SCHEMA_VERSION
    def to_json(self): return json.dumps(asdict(self))

def build_output(u, p, ent, competence_drop):
    return M2Output(timestamp=time.time(),
                    epistemic_uncertainty=float(u),
                    aleatoric_proxy=float(ent),
                    competence_drop=bool(competence_drop),
                    state_class=int(p.argmax()),
                    class_probabilities=[float(v) for v in p])
