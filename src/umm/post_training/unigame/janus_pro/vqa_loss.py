import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class VQALoss(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        answer_weight: float = 1.0,
        alignment_weight: float = 0.0,  
        reduction: str = "mean",
        align_margin: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.answer_weight = answer_weight
        self.alignment_weight = alignment_weight
        self.reduction = reduction
        self.align_margin = align_margin

    def forward(self, logits: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B, V = logits.shape
        y = batch["answers"]  
        assert y.shape == (B, V) and y.dtype.is_floating_point

        with torch.autocast('cuda', enabled=False):
            logits_fp32 = logits.float()
            y_fp32 = batch["answers"].to(logits_fp32.dtype)
            logq = F.log_softmax(logits_fp32, dim=-1)
            ce = -(y_fp32 * logq).sum(dim=-1)

        if self.reduction == "mean":
            ce = ce.mean()
        elif self.reduction == "sum":
            ce = ce.sum() 

        if ("image_features" in batch) and ("text_features" in batch):
            imgf = F.normalize(batch["image_features"], dim=-1)
            txtf = F.normalize(batch["text_features"], dim=-1)
            t = torch.ones(B, device=logits.device)
            align = F.cosine_embedding_loss(imgf, txtf, t, margin=self.align_margin, reduction=self.reduction)
        else:
            align = logits.new_tensor(0.0)

        total = self.answer_weight * ce + self.alignment_weight * align
        return {"ce_loss": ce, "alignment_loss": align, "total_loss": total}

