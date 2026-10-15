import torch

from training.master_weights import MasterAdamW


def _step(opt, param):
    param.grad = torch.ones_like(param)
    opt.step()
    opt.zero_grad(set_to_none=True)


def test_fp32_master_weights():
    param = torch.nn.Parameter(torch.tensor([0.0063], dtype=torch.bfloat16))
    start = param.detach().clone()
    # One bf16 spacing at 0.0063 is 2**-15 (~3.1e-5); Adam's first steps are ~lr = 1e-5.
    opt = MasterAdamW([param], lr=1e-5, weight_decay=0.0)
    master = opt.masters[0]
    assert master.dtype == torch.float32 and master.item() == start.float().item()

    _step(opt, param)
    once = master.item()
    assert once < start.float().item()
    assert torch.equal(param.detach(), start)
    assert param.grad is None

    _step(opt, param)
    assert master.item() < once
    assert param.item() < start.item()
    assert param.dtype == torch.bfloat16

    other = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.bfloat16))
    restored = MasterAdamW([other], lr=1e-5, weight_decay=0.0)
    restored.load_state_dict(opt.state_dict())
    assert torch.equal(restored.masters[0], master)
    assert torch.equal(other.detach(), param.detach())
    assert restored.adamw.state_dict()["state"][0]["step"] == 2
