"""AdamW over fp32 master copies of bf16 trainable parameters."""

from collections.abc import Iterable

import torch


class MasterAdamW:
    """AdamW that steps fp32 copies and casts them into the bf16 parameters.

    A step smaller than one bf16 spacing rounds away on a bf16 parameter;
    on the fp32 copy it accumulates until the cast moves the parameter.
    """

    def __init__(
        self, params: Iterable[torch.nn.Parameter], *, lr: float, weight_decay: float
    ) -> None:
        self.params = list(params)
        self.masters = [param.detach().float().clone().requires_grad_() for param in self.params]
        self.adamw = torch.optim.AdamW(self.masters, lr=lr, weight_decay=weight_decay)

    def step(self) -> None:
        for param, master in zip(self.params, self.masters, strict=True):
            master.grad = None if param.grad is None else param.grad.float()
        self.adamw.step()
        self._cast_back()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.adamw.zero_grad(set_to_none)
        for param in self.params:
            if param.grad is not None:
                param.grad = None if set_to_none else param.grad.zero_()

    def state_dict(self) -> dict[str, object]:
        return {
            "masters": [master.detach() for master in self.masters],
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        with torch.no_grad():
            for master, saved in zip(self.masters, state["masters"], strict=True):
                master.copy_(saved)
        self.adamw.load_state_dict(state["adamw"])
        self._cast_back()

    def _cast_back(self) -> None:
        with torch.no_grad():
            for param, master in zip(self.params, self.masters, strict=True):
                param.copy_(master)
