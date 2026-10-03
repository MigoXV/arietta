from torch import nn
from torch.nn import functional as F


class ChoiceCriterion(nn.Module):
    """FP32 hard/soft cross entropy; padding options never enter the distribution."""

    def forward(self, logits, targets, mask=None):
        logits = logits.float()
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e4)
        if targets.ndim == 1:
            return F.cross_entropy(logits, targets)
        return -(targets.float() * F.log_softmax(logits, dim=-1)).sum(-1).mean()
