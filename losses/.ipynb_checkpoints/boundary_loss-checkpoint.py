import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryLoss(nn.Module):
    """
    简化边界一致性正则：
    - 对预测概率图与真值掩码分别计算梯度幅值（Sobel）。
    - 以 L1 距离约束边界形状的一致性。
    可与 Dice/CE 并联。
    """
    def __init__(self, in_ch: int = 1, reduction: str = 'mean'):
        super().__init__()
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer('kx', kx.view(1, 1, 3, 3))
        self.register_buffer('ky', ky.view(1, 1, 3, 3))
        self.reduction = reduction

    def _grad_mag(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,H,W]
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        g = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return g

    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        # 多类情况下仅对前景聚合，也可以对每类求和
        probs = torch.softmax(logits, dim=1)
        if probs.size(1) > 1:
            probs_fg = probs[:, 1:, ...].sum(dim=1, keepdim=True)
        else:
            probs_fg = torch.sigmoid(logits)

        
        # 统一标签形状到 [B,1,H,W]
        if masks.dim() == 3:                           # [B,H,W]  整型标签
            masks_fg = (masks > 0).float().unsqueeze(1)
        elif masks.dim() == 4 and masks.size(1) == 1:  # [B,1,H,W]  二值/概率
            masks_fg = (masks > 0).float()
        elif masks.dim() == 4:                         # [B,C,H,W]  one-hot 或 logits式标签
            masks_fg = (masks[:, 1:, ...] > 0).float().sum(dim=1, keepdim=True).clamp(0, 1)
        else:
            raise ValueError(f"Unexpected masks shape {masks.shape}")

        gp = self._grad_mag(probs_fg)
        gm = self._grad_mag(masks_fg)
        loss = (gp - gm).abs()
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss