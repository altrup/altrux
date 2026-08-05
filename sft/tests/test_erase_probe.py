"""CPU tests for erase_probe's pure pieces: the rank-1 erase math and the
capture bookkeeping."""

import torch

from erase_probe import deflate, group_by_layer, rank1_erase, state_top_dirs


def read(ssm_state: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhpn,bn->bhp", ssm_state.float(), c.float())


def test_full_erase_zeroes_the_same_read():
    torch.manual_seed(0)
    s = torch.randn(1, 4, 8, 16)
    c = torch.randn(1, 16)
    erased = rank1_erase(s, c, gamma=1.0)
    assert read(erased, c).abs().max().item() < 1e-5


def test_partial_erase_attenuates_by_one_minus_gamma():
    torch.manual_seed(1)
    s = torch.randn(1, 4, 8, 16)
    c = torch.randn(1, 16)
    erased = rank1_erase(s, c, gamma=0.25)
    torch.testing.assert_close(read(erased, c), 0.75 * read(s, c), rtol=1e-5, atol=1e-5)


def test_orthogonal_read_is_untouched():
    torch.manual_seed(2)
    s = torch.randn(1, 4, 8, 16)
    c = torch.randn(1, 16)
    other = torch.randn(1, 16)
    other = other - (other * c).sum() / (c * c).sum() * c
    erased = rank1_erase(s, c, gamma=1.0)
    torch.testing.assert_close(read(erased, other), read(s, other), rtol=1e-5, atol=1e-5)


def test_erase_preserves_dtype_and_shape():
    s = torch.randn(1, 4, 8, 16).to(torch.bfloat16)
    c = torch.randn(1, 16).to(torch.bfloat16)
    erased = rank1_erase(s, c, gamma=0.5)
    assert erased.dtype == s.dtype and erased.shape == s.shape


def test_zero_query_is_a_noop():
    s = torch.randn(1, 4, 8, 16)
    c = torch.zeros(1, 16)
    torch.testing.assert_close(rank1_erase(s, c, gamma=1.0), s)


def test_deflate_removes_basis_components():
    torch.manual_seed(3)
    basis = torch.linalg.qr(torch.randn(16, 2))[0].T  # (2, 16) orthonormal rows
    c = torch.randn(1, 16)
    out = deflate(c, basis)
    assert torch.einsum("bn,kn->bk", out, basis).abs().max().item() < 1e-5


def test_erase_along_deflated_direction_spares_basis_reads():
    torch.manual_seed(4)
    s = torch.randn(1, 4, 8, 16)
    basis = torch.linalg.qr(torch.randn(16, 1))[0].T
    c = torch.randn(1, 16)
    erased = rank1_erase(s, deflate(c, basis), gamma=1.0)
    torch.testing.assert_close(read(erased, basis[:1]), read(s, basis[:1]), rtol=1e-4, atol=1e-5)


def test_state_top_dirs_finds_the_dominant_key():
    torch.manual_seed(5)
    key = torch.nn.functional.normalize(torch.randn(16), dim=0)
    vals = torch.randn(4, 8, 1)
    s = (vals * key).unsqueeze(0) + 0.01 * torch.randn(1, 4, 8, 16)
    top = state_top_dirs(s, k=1)
    assert torch.nn.functional.cosine_similarity(top[0], key, dim=0).abs().item() > 0.99


def test_group_by_layer_orders_capture_token_major():
    # _mixer_step appends one C per layer per token: [t0l0, t0l1, t1l0, t1l1]
    flat = ["t0l0", "t0l1", "t1l0", "t1l1"]
    assert group_by_layer(flat, n_layers=2) == [["t0l0", "t0l1"], ["t1l0", "t1l1"]]


def test_group_by_layer_rejects_partial_tokens():
    try:
        group_by_layer(["a", "b", "c"], n_layers=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError on a capture not divisible by n_layers")
