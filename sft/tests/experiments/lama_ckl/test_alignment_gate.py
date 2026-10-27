import json
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from experiments.lama_ckl.diagnostics.alignment_gate import (
    LRS,
    alpha_rule,
    assemble,
    bf16_change_fraction,
    bf16_spacing,
    cell_constants,
    fit_h,
    kappa_rule,
    load_gate_dreams,
    permutation_test,
    score,
    score_terms,
    target_mask,
)

MARKERS = {"user_id": 1, "asst_id": 2, "eos_id": 0, "eoc_id": 3}


def test_alpha_is_the_largest_passing_rung_and_an_edge_is_a_failing_top():
    lrs = LRS[:4]
    predicted = [1.0, 3.0, 10.0, 30.0]
    alpha, edge = alpha_rule(lrs, [0.9, 2.0, 4.0, 1.0], predicted)
    assert (alpha, edge) == (lrs[1], True)
    alpha, edge = alpha_rule(lrs, [0.9, 2.0, 6.0, 20.0], predicted)
    assert (alpha, edge) == (lrs[3], False)
    alpha, _ = alpha_rule(lrs, [0.1, 0.2, 0.3, 0.4], predicted)
    assert alpha is None


def test_alpha_rule_needs_a_positive_prediction():
    alpha, _ = alpha_rule(LRS[:2], [0.0, 0.0], [-1.0, -3.0])
    assert alpha is None


def test_cell_alpha_is_the_min_over_passing_rows_and_c_the_max_norm():
    alpha, c = cell_constants([3e-4, None, 1e-4, 1e-3], [2.0, 9.0, 5.0, 1.0])
    assert alpha == 1e-4
    assert c == 9.0
    assert cell_constants([None], [1.0])[0] is None


def test_permutation_rank_counts_labellings_at_or_above_the_observed():
    drops = [10.0, 9.0, 8.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    statistic, rank, p = permutation_test(drops, [0, 1, 2])
    assert statistic == pytest.approx(9.0 - 1.0)
    assert (rank, p) == (1, 1 / 84)
    _, rank, _ = permutation_test(drops, [3, 4, 5])
    assert rank == 84


def test_permutation_top_four_boundary():
    drops = [float(v) for v in range(9)]
    ranks = {}
    from itertools import combinations

    for labelling in combinations(range(9), 3):
        ranks[labelling] = permutation_test(drops, list(labelling))[1]
    ordered = sorted(ranks.values())
    assert ordered[0] == 1 and len(ranks) == 84
    assert ranks[(6, 7, 8)] == 1
    assert ranks[(5, 7, 8)] == 2


def test_h_fit_recovers_known_curvature():
    h_true, alpha_grad, norm2 = 4.0, 2.0, 3.0
    xs, ys = [], []
    for lr in LRS[:5]:
        drop = lr * alpha_grad - 0.5 * h_true * lr**2 * norm2 + 1e-9
        xs.append(-0.5 * lr**2 * norm2)
        ys.append(drop - lr * alpha_grad)
    h, intercept = fit_h(xs, ys)
    assert h == pytest.approx(h_true)
    assert intercept == pytest.approx(1e-9, abs=1e-12)


def test_kappa_rule_outcomes():
    assert kappa_rule(2.0, 1e-4, edge_found=True) == (2e-4, 2.0, "fit")
    assert kappa_rule(2.0, 1e-4, edge_found=False) == (2e-4, 2.0, "fit")
    assert kappa_rule(-1.0, 1e-4, edge_found=False) == (0.0, -1.0, "no-edge-zero")
    assert kappa_rule(-1.0, 1e-4, edge_found=True)[2] == "stop"
    kappa, h, status = kappa_rule(-1.0, 1e-4, edge_found=True, last_good_h=3.0)
    assert (kappa, h, status) == (pytest.approx(3e-4), 3.0, "reused")


def _scalars(grads, g_new, g_old):
    stacked = torch.tensor(grads, dtype=torch.float64)
    return {
        "gram": stacked @ stacked.T,
        "new": stacked @ torch.tensor(g_new, dtype=torch.float64),
        "old": stacked @ torch.tensor(g_old, dtype=torch.float64).T,
    }


def test_score_terms_on_hand_built_vectors():
    s = _scalars([[1.0, 0.0], [0.0, 1.0]], [2.0, 2.0], [[1.0, -1.0], [-1.0, -1.0]])
    learn, quad, hinge, clip = score_terms([0, 1], **s, alpha=0.5, kappa=2.0, c=10.0)
    assert clip == 1.0
    # G = (0.5, 0.5): <G, g_new> = 2, |G|^2 = 0.5
    assert learn == pytest.approx(0.5 * 2.0)
    assert quad == pytest.approx(0.5 * 0.5 * 2.0 * 0.5)
    # <G, g_old^0> = 0 (no fire), <G, g_old^1> = -1 (fires)
    assert hinge == pytest.approx(0.5 * 1.0)
    assert score([0, 1], **s, alpha=0.5, kappa=2.0, c=10.0) == pytest.approx(1.0 - 0.25 - 0.5)


def test_hinge_silent_on_positive_reference_alignment():
    s = _scalars([[1.0, 0.0]], [1.0, 0.0], [[1.0, 0.0], [0.0, 5.0]])
    assert score_terms([0], **s, alpha=1.0, kappa=0.0, c=10.0)[2] == 0.0


def test_clip_scales_g_before_every_term():
    s = _scalars([[3.0, 4.0]], [1.0, 0.0], [[-1.0, 0.0]])
    learn, quad, hinge, clip = score_terms([0], **s, alpha=1.0, kappa=1.0, c=1.0)
    assert clip == pytest.approx(0.2)
    assert learn == pytest.approx(3.0 * 0.2)
    assert quad == pytest.approx(0.5 * 1.0)
    assert hinge == pytest.approx(3.0 * 0.2)


def test_greedy_assembly_picks_the_best_batch():
    s = _scalars([[1.0, 0.0], [0.2, 0.0], [0.9, 0.0], [0.0, 0.0]], [1.0, 0.0], [[0.0, 0.0]])
    batch, value = assemble(range(4), 2, **s, alpha=1.0, kappa=0.0, c=100.0)
    assert sorted(batch) == [0, 2]
    assert value == pytest.approx(0.95)


def test_assembly_refuses_a_pool_smaller_than_n():
    s = _scalars([[1.0, 0.0]], [1.0, 0.0], [[0.0, 0.0]])
    assert assemble([0], 2, **s, alpha=1.0, kappa=0.0, c=1.0) == (None, None)


def test_swap_pass_improves_a_greedy_trap():
    # Dream 0 scores best alone, so greedy takes it first; the quadratic term
    # makes the pair {1, 2} (opposite off-axis parts cancel) the better batch.
    s = _scalars([[1.2, 0.0], [1.0, 1.0], [1.0, -1.0]], [1.0, 0.0], [[0.0, 0.0]])
    kwargs = {"alpha": 1.0, "kappa": 0.5, "c": 100.0}
    assert max(range(3), key=lambda i: score([i], **s, **kwargs)) == 0
    batch, value = assemble(range(3), 2, **s, **kwargs)
    assert sorted(batch) == [1, 2]
    assert value == pytest.approx(0.75)
    assert value > score([0, 1], **s, **kwargs)


def test_bf16_spacing_and_change_fraction():
    x = torch.tensor([1.0, 0.5, 3.0, -1.0])
    assert bf16_spacing(x).tolist() == [2**-7, 2**-8, 2**-6, 2**-7]
    theta = torch.ones(4)
    delta = torch.tensor([2**-6, 2**-10, 0.0, -(2**-6)])
    assert bf16_change_fraction(theta, delta) == pytest.approx(0.5)


def test_target_mask_covers_assistant_content_only():
    ids = [1, 10, 11, 2, 20, 21, 0, 3]
    assert target_mask(ids, **MARKERS) == [False, False, False, False, True, True, False, False]


def _neox():
    from models.common import build_tokenizer

    try:
        return build_tokenizer(
            SimpleNamespace(
                TOKENIZER_ID="EleutherAI/gpt-neox-20b",
                SPECIAL_TOKENS=["[USER]", "[ASSISTANT]", "<|endofconversation|>"],
            )
        )
    except OSError:
        pytest.skip("gpt-neox tokenizer not in the HF cache")


def test_target_mask_on_a_rendered_row():
    from preparation.conversations import format_conversation

    tokenizer = _neox()
    messages = [
        {"role": "user", "content": "Who built the bridge?"},
        {"role": "assistant", "content": "Gustave Eiffel"},
    ]
    ids, _, _ = format_conversation(
        messages, tokenizer, 10**9, "[USER]", "[ASSISTANT]", "<|endofconversation|>"
    )
    ids_of = tokenizer.convert_tokens_to_ids
    mask = target_mask(
        ids,
        user_id=ids_of("[USER]"),
        asst_id=ids_of("[ASSISTANT]"),
        eos_id=tokenizer.eos_token_id,
        eoc_id=ids_of("<|endofconversation|>"),
    )
    target = [t for t, m in zip(ids, mask, strict=True) if m]
    assert tokenizer.decode(target) == " Gustave Eiffel"
    assert ids[-2:] == [tokenizer.eos_token_id, ids_of("<|endofconversation|>")]


def test_gate_dreams_schema_is_checked(tmp_path):
    row = {"passage_id": "p1", "question": {"answer": "Paris"}}
    good = [
        {"kind": k, "text": f"{k} {i}"}
        for k in ("revisit", "acknowledgement", "unrelated")
        for i in range(3)
    ]
    path = tmp_path / "gate_dreams.json"
    path.write_text(json.dumps({"p1": good}))
    assert [d["kind"] for d in load_gate_dreams(path, [row])["p1"]][:3] == ["revisit"] * 3
    path.write_text(
        json.dumps({"p1": [*good[:2], {"kind": "revisit", "text": "in paris"}, *good[3:]]})
    )
    with pytest.raises(ValueError):
        load_gate_dreams(path, [row])
    path.write_text(json.dumps({"p1": good[:8]}))
    with pytest.raises(ValueError):
        load_gate_dreams(path, [row])


def _tiny_model():
    import models  # noqa: F401  (installs the selective_scan_cuda stub)
    import models.mamba2_780m.model as m780

    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    from adapters.lora import apply_lora

    torch.manual_seed(0)
    cfg = MambaConfig(
        d_model=64,
        n_layer=4,
        vocab_size=96,
        ssm_cfg={"layer": "Mamba2", "headdim": 16, "d_state": 16, "expand": 2, "ngroups": 1},
        rms_norm=True,
        fused_add_norm=False,
        tie_embeddings=True,
    )
    base = MambaLMHeadModel(cfg, device="cuda", dtype=torch.float32)
    base.marker_token_ids = [94, 95]
    base = apply_lora(base, ["in_proj", "out_proj"], rank=4, alpha=8.0, dropout=0.0)
    model = m780.Model(base).to("cuda")
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name or "marker_delta" in name:
            p.requires_grad_(True)
            params.append(p)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.05)
    return model, params


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="mamba_ssm norms need a GPU")


@gpu
def test_a_gate_step_follows_first_order_and_restores_theta():
    from experiments.lama_ckl.diagnostics.alignment_gate import (
        dream_gradient,
        forced_dreams,
        losses,
        mean_loss_grad,
        stepped_losses,
    )
    from training.master_weights import MasterOptimizer

    model, params = _tiny_model()
    opt = MasterOptimizer(params, torch.optim.SGD, lr=1.0, momentum=0.0, weight_decay=0.0)
    seqs = [([5, 6, 7, 8, 9, 10], [False, False, True, True, True, False])]
    grad = mean_loss_grad(model, seqs, 1)
    theta = [p.detach().clone() for p in params]
    lr = 1e-3
    drop = losses(model, seqs)[0] - stepped_losses(model, opt, grad, lr, seqs)[0]
    assert drop == pytest.approx(lr * float(grad @ grad), rel=0.05)
    assert all(torch.equal(a, p) for a, p in zip(theta, params, strict=True))

    with torch.no_grad():
        _, state = model(torch.tensor([[1, 2, 3]], device="cuda"), state=None)
    dreams = forced_dreams(model, state, [93], [[94, 11, 12, 0], [94, 13, 0, 95, 14, 15]], str)
    assert [len(d.teacher_logits) for d in dreams] == [5, 7]
    ids = {"user_id": 94, "asst_id": 95, "eos_id": 0, "eoc_id": 93}
    g = dream_gradient(model, dreams[1], ids)
    assert g is not None and g.shape == grad.shape and float(g.norm()) > 0


def _torch_norm(module):
    """mamba_ssm's (gated) RMSNorm in plain torch ops, which autograd differentiates twice."""

    def forward(x, z=None):
        before = getattr(module, "norm_before_gate", True)
        if z is not None and not before:
            x = x * F.silu(z)
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + module.eps) * module.weight
        return y * F.silu(z) if z is not None and before else y

    return forward


@gpu
def test_double_backward_check_tells_a_cut_graph_from_a_whole_one():
    from experiments.lama_ckl.diagnostics.alignment_gate import double_backward

    errors = []
    for torch_norms in (False, True):
        torch.manual_seed(1)
        model, params = _tiny_model()
        if torch_norms:
            for module in [
                model.norm_f,
                *(m for layer in model.layers for m in (layer.norm, layer.mixer.norm)),
            ]:
                module.forward = _torch_norm(module)
        ids = [torch.randint(1, 90, (1, n), device="cuda") for n in (6, 8, 6)]
        out = double_backward(model, params, *ids, block=4)
        assert out["peak_bytes"] > 0 and math.isfinite(out["loss_gm"])
        errors.append(out["dpi_relative_error"])
    # The Triton norms' backward is not differentiable, so the second-order path is cut.
    assert errors[0] > 0.5
    assert errors[1] < 1e-2
