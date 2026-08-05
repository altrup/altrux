"""CPU tests for erase_probe's pure pieces: the rank-1 erase math and the
capture bookkeeping."""

import torch

from erase_probe import group_by_layer, rank1_erase


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
