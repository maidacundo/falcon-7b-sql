#!/usr/bin/env python3
from torch import nn

class MaskedCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    Custom CrossEntropyLoss that allows masking.
    Loss is computer only on non-masked labels.
    Masked labels are intended to be the tokens that are not part of the SQL query.
    
    """
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean"):
        super().__init__(weight, size_average, ignore_index, reduce, "none")
        self._reduction = reduction

    def forward(self, input, target, mask=None):
        input = input.view(-1, input.size(-1))
        target = target.view(-1)

        if mask is not None:
            mask = mask.view(-1).bool()
            input = input[mask]
            target = target[mask]

        size = target.numel()

        loss = super().forward(input, target)

        if self._reduction == "none":
            return loss
        return loss.sum() / (size + 1e-8)