import torch

from training.checkpoints import load_checkpoint
from training.master_weights import MasterAdamW, MasterOptimizer, fp32_forward


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

    saved = opt.state_dict()
    legacy = MasterAdamW([other], lr=1e-5, weight_decay=0.0)
    legacy.load_state_dict({"masters": saved["masters"], "adamw": saved["optimizer"]})
    assert legacy.adamw.state_dict()["state"][0]["step"] == 2


def test_sgd_master_accumulates_sub_spacing_steps():
    param = torch.nn.Parameter(torch.tensor([0.0063], dtype=torch.bfloat16))
    start = param.detach().clone()
    # lr * grad = 1e-5, under half the bf16 spacing (2**-15) at 0.0063.
    opt = MasterOptimizer([param], torch.optim.SGD, lr=1e-5)
    master = opt.masters[0]

    _step(opt, param)
    assert master.item() < start.float().item()
    assert torch.equal(param.detach(), start)

    for _ in range(3):
        _step(opt, param)
    assert param.item() < start.item()

    other = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.bfloat16))
    restored = MasterOptimizer([other], torch.optim.SGD, lr=1e-5)
    restored.load_state_dict(opt.state_dict())
    assert torch.equal(restored.masters[0], master)
    assert torch.equal(other.detach(), param.detach())


def test_fp32_forward_steps_the_parameters_with_no_copy(tmp_path):
    model = torch.nn.Linear(2, 2).to(torch.bfloat16)
    fp32_forward(model)
    assert all(p.dtype == torch.float32 for p in model.parameters())

    # A value bf16 cannot hold, so a bf16 round trip on load would change it.
    saved = {"weight": torch.full((2, 2), 1.0 + 2**-12)}
    torch.save(saved, tmp_path / "trainable.pt")
    load_checkpoint(model, tmp_path)
    assert torch.equal(model.weight.detach(), saved["weight"])

    opt = MasterOptimizer([model.weight], torch.optim.SGD, lr=2**-12)
    assert opt.masters[0] is model.weight
    ptr = model.weight.data_ptr()
    _step(opt, model.weight)
    assert model.weight.data_ptr() == ptr
    assert torch.equal(model.weight.detach(), torch.ones(2, 2))

    other = torch.nn.Linear(2, 2)
    restored = MasterOptimizer([other.weight], torch.optim.SGD, lr=2**-12)
    restored.load_state_dict(opt.state_dict())
    assert torch.equal(other.weight.detach(), model.weight.detach())
