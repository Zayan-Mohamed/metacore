"""Standard OOD-detection metrics (Hendrycks & Gimpel 2017) + ECE (Guo 2017),
in NumPy so there is no sklearn dependency. score = epistemic u; positive = OOD."""
import numpy as np

def auroc(pos, neg):
    s = np.concatenate([pos,neg]); order = np.argsort(s)
    ranks = np.empty_like(order,float); ranks[order] = np.arange(1,len(s)+1)
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

def aupr(pos, neg):
    s = np.concatenate([pos,neg]); lab = np.concatenate([np.ones_like(pos),np.zeros_like(neg)])
    idx = np.argsort(-s); lab = lab[idx]
    tp = np.cumsum(lab); fp = np.cumsum(1-lab)
    prec = tp/(tp+fp); rec = tp/lab.sum()
    o = np.argsort(rec); return float(np.trapezoid(prec[o], rec[o]))

def fpr95(pos, neg):
    thr = np.quantile(pos, 0.05)         # 95% of OOD above this
    return float((neg >= thr).mean())

def ece(probs, labels, n_bins=10):
    conf = probs.max(1); pred = probs.argmax(1); acc = (pred==labels)
    bins = np.linspace(0,1,n_bins+1); e = 0.0
    for i in range(n_bins):
        m = (conf>bins[i]) & (conf<=bins[i+1])
        if m.sum()>0: e += m.mean()*abs(acc[m].mean()-conf[m].mean())
    return float(e)
