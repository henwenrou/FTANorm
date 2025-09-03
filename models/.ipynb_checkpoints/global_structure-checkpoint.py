import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalStructureEncoder(nn.Module):
    """
    从原始图像提取全局结构上下文：强度统计 + 边缘/纹理 + 低分辨率布局。
    输出 context 向量供 FTANorm 调制。
    """
    def __init__(self, in_ch: int = 1, context_dim: int = 64):
        super().__init__()
        self.in_ch = in_ch
        self.context_dim = context_dim

        # Sobel 卷积核（固定参数）
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer('sobel_kx', kx.view(1, 1, 3, 3).repeat(in_ch, 1, 1, 1))
        self.register_buffer('sobel_ky', ky.view(1, 1, 3, 3).repeat(in_ch, 1, 1, 1))

        # 轻量 CNN 编码器
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),  # 捕获低频布局
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8 + 7, 128),  # 2048 + 4(stats) + 3(hist) = 2055
            nn.ReLU(inplace=True),
            nn.Linear(128, context_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        # 强度统计（每通道均值与标准差）
        mean = x.mean(dim=[2, 3])  # [B,C]
        std = x.std(dim=[2, 3]) + 1e-6
        intensity_stats = torch.stack([mean.mean(dim=1), std.mean(dim=1)], dim=1)  # [B,2]

        # 梯度幅值
        gx = F.conv2d(x, self.sobel_kx, padding=1, groups=self.in_ch)
        gy = F.conv2d(x, self.sobel_ky, padding=1, groups=self.in_ch)
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        edge_mean = grad_mag.mean(dim=[1, 2, 3])  # [B]
        edge_std = grad_mag.std(dim=[1, 2, 3])
        edge_stats = torch.stack([edge_mean, edge_std], dim=1)  # [B,2]

        # 低分辨率布局特征
        low = self.backbone(x)  # [B,32,8,8]
        low_flat = low.view(B, -1)  # [B,32*8*8]

        # 聚合
        stats = torch.cat([intensity_stats, edge_stats], dim=1)  # [B,4]
        # 附加一个简单直方图-like 统计：将强度归一化并做 4-bin 直方图近似
        xn = (x - mean.view(B, C, 1, 1)) / (std.view(B, C, 1, 1) + 1e-6)
        bins = torch.tensor([-1.5, -0.5, 0.5, 1.5], device=x.device)
        # 计算每个 bin 的占比（粗略软直方图）
        hist = []
        for i in range(3):
            left = bins[i]
            right = bins[i+1]
            mask = (xn >= left) & (xn < right)
            hist.append(mask.float().mean(dim=[1, 2, 3]))
        hist = torch.stack(hist, dim=1)  # [B,3]

        feat = torch.cat([low_flat, stats, hist], dim=1)  # [B, 32*64 + 4 + 3]
        ctx = self.head(feat)
        return ctx
