import functools
import torch, sys, tempfile
from pathlib import Path
sys.path.insert(0, '..')
import models.mamba2_2_7b_memory.model as M
M.causal_conv1d_update = None
M.D_MODEL = 32; M.N_LAYER = 6; M.NHEADS = 8; M.HEADDIM = 8; M.D_STATE = 8  # nheads = (expand=2 * d_model) / headdim = 64/8 = 8
M.READ_LAYER = 3; M.INJECTED_LAYERS = (2, 4); M.MEM_DIM = M.D_MODEL; M.MEM_HIDDEN = 4*M.D_MODEL
M.QUANTIZE_LORA_BASE = False  # skip bitsandbytes for this tiny smoke test
M._TitansFrontEnd = functools.partial(M._TitansFrontEnd, d_model=32, mem_dim=32, mem_hidden=128)
M._GatedDeltaInjection = functools.partial(M._GatedDeltaInjection, mem_dim=32, r=8, nheads=8, headdim=8, d_state=8)

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from lora import apply_lora

import train_memory as TM
TM.CKPT_DIR = Path(tempfile.mkdtemp())

device = 'cuda'
torch.manual_seed(0)
cfg = MambaConfig(d_model=M.D_MODEL, n_layer=M.N_LAYER, vocab_size=50, ssm_cfg=dict(layer='Mamba2', headdim=M.HEADDIM, ngroups=1, d_state=M.D_STATE))
mamba = MambaLMHeadModel(cfg, device=device, dtype=torch.float32)
mamba = apply_lora(mamba, ['in_proj', 'out_proj'], 4, 8.0, 0.0)
model = M.Model(mamba).to(device)
trainable = [p for p in model.parameters() if p.requires_grad]
print('trainable params:', sum(p.numel() for p in trainable), flush=True)

ids = torch.randint(0, 50, (37,), device=device)
loss_sum, weight_sum = TM.process_example(model, ids, chunk_len=10, eos_weight=2.0, backward_scale=1.0)
print('loss_sum, weight_sum:', loss_sum, weight_sum, flush=True)

lora_grads = memory_grads = 0
for name, p in model.named_parameters():
    if not p.requires_grad or p.grad is None or p.grad.abs().max() == 0:
        continue
    if "lora_A" in name or "lora_B" in name:
        lora_grads += 1
    else:
        memory_grads += 1
print('lora_grads:', lora_grads, 'memory_grads:', memory_grads, flush=True)
print("--- debug: per-param grad status ---")
for name, p in model.named_parameters():
    if p.requires_grad and ("front_end" in name or "injections" in name):
        g = p.grad
        status = "None" if g is None else f"max={g.abs().max().item():.3e}"
        print(f"  {name:50s} grad={status}")
assert lora_grads > 0 and memory_grads > 0

optimizer = torch.optim.AdamW(trainable, lr=1e-4)
torch.nn.utils.clip_grad_norm_(trainable, 1.0)
optimizer.step()
optimizer.zero_grad()
print('optimizer step OK', flush=True)

path = TM.save_checkpoint(model, optimizer, step=1, epoch=0, example_idx=0, lora_rank=4, lora_alpha=8.0)
print('saved checkpoint at', path, flush=True)

# Fresh model, load the checkpoint back
mamba2 = MambaLMHeadModel(cfg, device=device, dtype=torch.float32)
mamba2 = apply_lora(mamba2, ['in_proj', 'out_proj'], 4, 8.0, 0.0)
model2 = M.Model(mamba2).to(device)
TM.load_checkpoint(model2, path)
print('load_checkpoint OK', flush=True)

el = TM.eval_loss(model, [ids.cpu()], device, chunk_len=10, max_len=1000)
print('eval_loss:', el, flush=True)
print('ALL OK')

print("--- debug: per-param grad status ---")
for name, p in model.named_parameters():
    if p.requires_grad and ("front_end" in name or "injections" in name):
        g = p.grad
        status = "None" if g is None else f"max={g.abs().max().item():.3e}"
        print(f"  {name:50s} requires_grad={p.requires_grad} grad={status}")
