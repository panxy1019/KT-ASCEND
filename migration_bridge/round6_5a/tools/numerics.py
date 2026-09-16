"""CPU-only diagnostic metrics; no serving or accelerator imports."""
import hashlib
import torch


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def metrics(a,b):
    a,b=a.float(),b.float();d=a-b
    return dict(exact=torch.equal(a,b),max_abs=d.abs().max().item(),
        relative_l2=(d.norm()/b.norm().clamp_min(1e-30)).item(),
        cosine=torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0).item(),
        finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
