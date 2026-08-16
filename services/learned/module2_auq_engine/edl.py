"""Evidential Deep Learning head (Sensoy et al., NeurIPS 2018) with OOD-aware
evidence regularisation. Outputs Dirichlet evidence -> epistemic u = K/S."""
import torch, torch.nn as nn, torch.nn.functional as F

class EDLNet(nn.Module):
    def __init__(self, d_in, k, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in,hidden), nn.ReLU(),
                                 nn.Linear(hidden,hidden), nn.ReLU(),
                                 nn.Linear(hidden,k))
    def forward(self, x):
        return F.softplus(self.net(x))            # non-negative evidence

def dirichlet(evidence):
    alpha = evidence + 1.0
    S = alpha.sum(1, keepdim=True)
    return alpha, S

def uncertainty(evidence):
    """Return epistemic u = K/S, class probabilities p, and predictive entropy."""
    alpha, S = dirichlet(evidence)
    K = evidence.shape[1]
    p = alpha / S
    u = (K / S).squeeze(1)                        # epistemic vacuity in [0,1]
    ent = -(p*torch.log(p+1e-9)).sum(1)           # aleatoric proxy
    return u, p, ent

def kl_to_uniform(alpha):
    """KL( Dir(alpha) || Dir(1) ) -- pushing this to 0 drives evidence -> 0."""
    K = alpha.shape[1]; ones = torch.ones((1,K))
    S = alpha.sum(1, keepdim=True)
    t1 = (torch.lgamma(S) - torch.lgamma(alpha).sum(1,keepdim=True)
          - (torch.lgamma(ones.sum()) - torch.lgamma(ones).sum()))
    t2 = ((alpha-ones)*(torch.digamma(alpha)-torch.digamma(S))).sum(1,keepdim=True)
    return (t1+t2).squeeze(1)

def edl_mse_loss(evidence, y, epoch, kl_anneal_epochs):
    """Bayes-risk MSE loss with annealed KL on misleading evidence."""
    alpha, S = dirichlet(evidence); K = evidence.shape[1]
    p = alpha/S; yh = torch.eye(K)[y]
    err = ((yh-p)**2).sum(1)
    var = (p*(1-p)/(S+1)).sum(1)
    lam = min(1.0, epoch/float(kl_anneal_epochs))
    alpha_tilde = yh + (1-yh)*alpha               # keep evidence for the true class
    return (err + var + lam*kl_to_uniform(alpha_tilde)).mean()
