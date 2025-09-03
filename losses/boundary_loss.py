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

    def _ensure_onehot(self, masks: torch.Tensor, C: int) -> torch.Tensor:
        # masks: [B,H,W] (long) or [B,1,H,W] or [B,C,H,W]
        if masks.dim() == 3:
            return F.one_hot(masks.long(), num_classes=C).permute(0,3,1,2).float()
        if masks.dim() == 4 and masks.size(1) == 1:
            # 二值前景（0/1）
            bg = (masks <= 0).float()
            fg = (masks > 0).float()
            return torch.cat([bg, fg], dim=1)
        return masks.float()  # [B,C,H,W]


    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        probs = torch.softmax(logits, dim=1) if C > 1 else torch.sigmoid(logits)
        if C == 1:
            probs = torch.cat([1 - probs, probs], dim=1); C = 2
        masks_1h = self._ensure_onehot(masks, C)

        # 忽略背景，逐类边界一致性求平均
        losses = []
        for c in range(1, C):
            gp = self._grad_mag(probs[:, c:c+1])
            gm = self._grad_mag(masks_1h[:, c:c+1])
            losses.append((gp - gm).abs().mean())
        return torch.stack(losses).mean() if losses else torch.tensor(0., device=logits.device)