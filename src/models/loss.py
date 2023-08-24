import torch
import torch.nn as nn
from torch.nn import functional as F


def get_loss_module(config):

    task = config['task']

    if (task == "imputation") or (task == "transduction"):
        return MaskedMSELoss(reduction='none')  # outputs loss for each batch element

    if task == "classification":
        return NoFussCrossEntropyLoss(reduction='none')  # outputs loss for each batch sample

    if task == "regression":
        return nn.MSELoss(reduction='none')  # outputs loss for each batch sample

    if task == "simmtm":
        return nn.MSELoss(reduction='mean')
    else:
        raise ValueError("Loss module for task '{}' does not exist".format(task))


def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))



def contrastive_loss(R, B):
    M = R.shape[0]//B
    mask = torch.eye(B, device=R.device).repeat_interleave(M,dim=0).repeat_interleave(M,dim=1)
    # mask = torch.eye(B, device=R.device).repeat_interleave(M,dim=0).repeat_interleave(M,dim=1) - torch.eye(R.shape[0], device=R.device)
    # print("mask", mask, mask.shape)
    denom = R * (torch.ones_like(R) - torch.eye(R.shape[0], device=R.device))
    # denom = (denom * (torch.ones_like(mask) - mask)).sum(-1).unsqueeze(-1)
    denom = R.sum(-1).unsqueeze(-1)
    # print("con loss:", torch.isnan(R).any(), torch.isnan(denom).any())
    loss = torch.log(R/denom)
    
    # print("constraint", mask.shape, mask, loss.shape)
    loss = (loss * (mask - torch.eye(R.shape[0], device=R.device))).sum(1)/(M-1) # except no masked unit
    loss = loss.mean(0)
    
    return -loss



class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)
