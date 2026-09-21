"""Checks that micro-batch 2 x accum 3 gives the same update as batch 6, and that
the teacher EMA now covers BatchNorm buffers.   Run:  python tests/test_accumulation.py"""
import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from trainer import train_one_epoch, ema_update_teacher


class Tiny(torch.nn.Module):
    def __init__(self, bn=False):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.LayerNorm(16), torch.nn.ReLU())
        self.bn = torch.nn.BatchNorm1d(16) if bn else torch.nn.Identity()
        self.omni_heads = torch.nn.ModuleList([torch.nn.Linear(16, 3)])

    def forward(self, x, head_n):
        f = self.bn(self.body(x))
        return f, self.omni_heads[head_n](f)


def run(bs, accum):
    torch.manual_seed(0)
    x1, x2, y = torch.randn(10, 8), torch.randn(10, 8), torch.randint(0, 2, (10, 3)).float()
    loader = DataLoader(TensorDataset(x1, x2, y), batch_size=bs, shuffle=False)
    torch.manual_seed(1)
    model = Tiny()
    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad = False
    opt = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
    sched = np.array([0.95, 0.97, 0.99])
    for ep in range(2):
        train_one_epoch(model, 0, "toy", loader, "cpu", torch.nn.BCEWithLogitsLoss(), opt, ep,
                        "epoch", teacher, sched, ep, accum_steps=accum, print_freq=1000)
    return model, teacher


a, ta = run(6, 1)
b, tb = run(2, 3)
d = max((p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters()))
dt = max((p - q).abs().max().item() for p, q in zip(ta.parameters(), tb.parameters()))
print("max |student diff| = {:.2e}   max |teacher diff| = {:.2e}".format(d, dt))
assert d < 1e-5 and dt < 1e-5, "accumulation is not equivalent"

s, t = Tiny(bn=True), Tiny(bn=True)
s.bn.running_mean.fill_(1.0)
ema_update_teacher(s, t, np.array([0.9]), 0)
assert torch.allclose(t.bn.running_mean, torch.full((16,), 0.1)), "BN buffers not averaged"
print("BN running_mean after one EMA step:", round(t.bn.running_mean[0].item(), 4), "(expected 0.1)")
print("ALL TESTS PASSED")
