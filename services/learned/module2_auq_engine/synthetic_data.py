"""Stand-in for Module 1's shared ID/OOD scenario library, so Module 2 can be
built and evaluated BEFORE M1's real state representation exists (see plan NFR4).
Replace sample_* with the real M1 adapter once the M1->M2 contract is live."""
import numpy as np

FEATURES = ["voltage_pu","load_factor","freq_dev_hz","wind_ms",
            "rainfall_mm","solar_wm2","gen_margin","temp_c"]
D = len(FEATURES)

def sample_id(n, rng):
    """Normal-operation island states + a 3-class safety label (safe/stressed/critical)."""
    x = np.zeros((n, D), np.float32)
    x[:,0] = np.clip(rng.normal(1.0,0.02,n),0.94,1.06)   # voltage (pu)
    x[:,1] = rng.uniform(0.3,0.9,n)                       # load factor
    x[:,2] = rng.normal(0,0.05,n)                         # frequency deviation (Hz)
    x[:,3] = rng.uniform(0,12,n)                          # wind (m/s)
    x[:,4] = rng.exponential(2,n)                         # rainfall (mm)
    x[:,5] = rng.uniform(0,900,n)                         # solar (W/m2)
    x[:,6] = rng.uniform(0.1,0.6,n)                       # generation margin
    x[:,7] = rng.normal(28,3,n)                           # temperature (C)
    # physically-motivated risk -> safety class (wind & rain DO matter, so the
    # model learns to use them; otherwise it would ignore the cyclone features)
    risk = (0.32*x[:,1] + 0.26*np.abs(x[:,0]-1.0)*10 + 0.12*np.abs(x[:,2])*10
            + 0.18*(x[:,3]/12) + 0.12*np.clip(x[:,4]/10,0,1))
    q = np.quantile(risk,[0.5,0.83])
    y = np.digitize(risk,q).astype(np.int64)              # 0 safe, 1 stressed, 2 critical
    return x, y

def sample_ood(n, rng):
    """Cyclone / extreme states, far outside the training distribution (unlabelled)."""
    x = np.zeros((n, D), np.float32)
    x[:,0] = np.clip(rng.normal(1.0,0.09,n),0.80,1.20)
    x[:,1] = rng.uniform(0.85,1.15,n)
    x[:,2] = rng.normal(0,0.4,n)
    x[:,3] = rng.uniform(25,45,n)                         # cyclonic wind
    x[:,4] = rng.uniform(40,120,n)                        # extreme rainfall
    x[:,5] = rng.uniform(0,300,n)
    x[:,6] = rng.uniform(-0.1,0.1,n)                      # generation deficit
    x[:,7] = rng.normal(30,4,n)
    return x

class Normalizer:
    def fit(self, x):  self.mu = x.mean(0); self.sd = x.std(0)+1e-6; return self
    def __call__(self, x): return ((x-self.mu)/self.sd).astype(np.float32)
