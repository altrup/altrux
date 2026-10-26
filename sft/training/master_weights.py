"""An optimizer over fp32 master copies of low-precision trainable parameters."""

from collections.abc import Iterable

import torch


class MasterOptimizer:
    """Steps fp32 copies with `optimizer_class` and casts them into the parameters.

    A step smaller than one bf16 spacing rounds away on a bf16 parameter;
    on the fp32 copy it accumulates until the cast moves the parameter.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        optimizer_class: type[torch.optim.Optimizer],
        **kwargs: float,
    ) -> None:
        self.params = list(params)
        # An fp32 parameter is its own master: no copy, no cast-back.
        self.masters = [
            param
            if param.dtype == torch.float32
            else param.detach().float().clone().requires_grad_()
            for param in self.params
        ]
        self.optimizer = optimizer_class(self.masters, **kwargs)

    def step(self) -> None:
        for param, master in self._copies():
            master.grad = None if param.grad is None else param.grad.float()
        self.optimizer.step()
        self._cast_back()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none)
        for param in self.params:
            if param.grad is not None:
                param.grad = None if set_to_none else param.grad.zero_()

    def state_dict(self) -> dict[str, object]:
        return {
            "masters": [master.detach() for master in self.masters],
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        with torch.no_grad():
            for master, saved in zip(self.masters, state["masters"], strict=True):
                master.copy_(saved)
        # "adamw" is the key optimizer.pt files written before the generalisation carry.
        self.optimizer.load_state_dict(
            state["optimizer"] if "optimizer" in state else state["adamw"]
        )
        self._cast_back()

    def _copies(self) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
        return [(p, m) for p, m in zip(self.params, self.masters, strict=True) if m is not p]

    def _cast_back(self) -> None:
        with torch.no_grad():
            for param, master in self._copies():
                param.copy_(master)


class MasterAdamW(MasterOptimizer):
    def __init__(
        self, params: Iterable[torch.nn.Parameter], *, lr: float, weight_decay: float
    ) -> None:
        super().__init__(params, torch.optim.AdamW, lr=lr, weight_decay=weight_decay)
        self.adamw = self.optimizer


def fp32_forward(model: torch.nn.Module) -> None:
    """Cast the whole model to fp32 in place; the bf16 storage is freed, not kept.

    Call it before `load_checkpoint`, so a `trainable.pt` loads straight into fp32.
    """
    model.float()
