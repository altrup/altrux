from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

print("Loading state-spaces/mamba2-370m ...")
model = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-370m")
backbone = model.backbone

print(f"\n--- Layer count ---")
print(f"len(backbone.layers) = {len(backbone.layers)}")

print(f"\n--- Block[0] type ---")
print(type(backbone.layers[0]))
print(backbone.layers[0])

print(f"\n--- Mixer type + key attrs ---")
mixer = backbone.layers[0].mixer
print(f"mixer class: {mixer.__class__.__name__}")
for attr in ("d_model", "d_state", "d_conv", "expand", "headdim", "ngroups"):
    print(f"  {attr}: {getattr(mixer, attr, '<missing>')}")

print(f"\n--- Embedding ---")
print(backbone.embedding)

print(f"\n--- norm_f ---")
print(backbone.norm_f)

print(f"\n--- lm_head ---")
print(model.lm_head)

print(f"\n--- State dict keys (layer 0 + lm_head) ---")
sd = model.state_dict()
for k in sd:
    if "layers.0" in k or k.startswith("lm_head"):
        print(f"  {k}: {tuple(sd[k].shape)}")
