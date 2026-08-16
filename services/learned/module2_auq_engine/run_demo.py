"""End-to-end Module 2 prototype (the Week-7 demo), on synthetic data only.
Trains EDL -> shows u low on normal / high on cyclone -> reports OOD metrics ->
calibrates the competence-drop trigger -> emits sample M2->M3 messages."""
import numpy as np, torch, yaml, json
from synthetic_data import sample_id, sample_ood, Normalizer, D
from edl import EDLNet, uncertainty, edl_mse_loss, kl_to_uniform
from trigger import CompetenceDropTrigger
from evaluate import auroc, aupr, fpr95, ece
from contract import build_output

cfg = yaml.safe_load(open("config.yaml"))
import random
random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
torch.use_deterministic_algorithms(True)
K = cfg["k_classes"]; tr = cfg["train"]
rng = np.random.default_rng(cfg["seed"]); torch.manual_seed(cfg["seed"])

Xtr,ytr = sample_id(3000,rng); Xte,yte = sample_id(1000,rng); Xood = sample_ood(800,rng)
nz = Normalizer().fit(Xtr)
Xtr_,Xte_,Xood_ = nz(Xtr), nz(Xte), nz(Xood)

m = EDLNet(D,K); opt = torch.optim.Adam(m.parameters(), tr["lr"], weight_decay=tr["weight_decay"])
Xt = torch.tensor(Xtr_); yt = torch.tensor(ytr); N=len(Xt)
for ep in range(tr["epochs"]):
    perm = torch.randperm(N)
    for i in range(0,N,tr["batch_size"]):
        idx = perm[i:i+tr["batch_size"]]; xb = Xt[idx]
        xo = xb + torch.randn_like(xb)*tr["ood_proxy_sigma"]          # far-OOD proxy
        loss = edl_mse_loss(m(xb),yt[idx],ep,tr["kl_anneal_epochs"]) \
               + tr["ood_reg_weight"]*kl_to_uniform(m(xo)+1.0).mean()
        opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    u_id,p_id,e_id = uncertainty(m(torch.tensor(Xte_)))
    u_ood,p_ood,e_ood = uncertainty(m(torch.tensor(Xood_)))
u_id,p_id,e_id = u_id.numpy(),p_id.numpy(),e_id.numpy()
u_ood = u_ood.numpy()
acc = (p_id.argmax(1)==yte).mean()

print("="*56)
print("MODULE 2 - AUQ ENGINE - PROTOTYPE RESULTS (synthetic)")
print("="*56)
print(f"ID 3-class accuracy         : {acc:.3f}")
print(f"mean epistemic u  ID / OOD  : {u_id.mean():.3f} / {u_ood.mean():.3f}")
print(f"AUROC (u, OOD vs ID)        : {auroc(u_ood,u_id):.3f}   (target >= 0.90)")
print(f"AUPR  (OOD)                 : {aupr(u_ood,u_id):.3f}")
print(f"FPR95                       : {fpr95(u_ood,u_id):.3f}   (lower better)")
print(f"ECE   (ID calibration)      : {ece(p_id,yte):.3f}   (lower better)")

trig = CompetenceDropTrigger.calibrate(u_id, cfg["trigger"]["false_alarm_rate"],
                                       cfg["trigger"]["hysteresis"])
print(f"competence-drop threshold   : {trig.thr:.3f}")
print(f"trigger fires  ID / OOD     : {(u_id>trig.thr).mean():.3f} / {(u_ood>trig.thr).mean():.3f}")

# --- emit a few M2->M3 messages (the mock stream Saabir consumes now) ---
print("\nSample M2 -> M3 contract messages (mixed normal + cyclone):")
stream = list(zip(u_id[:2],p_id[:2])) + list(zip(u_ood[:2],p_ood.numpy()[:2]))
t2 = CompetenceDropTrigger(trig.thr, 1)
with open("sample_m2_to_m3.jsonl","w") as f:
    for u,p in stream:
        msg = build_output(u,p,0.0,t2.update(u))
        f.write(msg.to_json()+"\n"); print("  "+msg.to_json())
print("\n(sample_m2_to_m3.jsonl written)")
