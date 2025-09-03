
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class FTANorm2d(nn.Module):
    """
    Feature-Topology-Aware Normalization（简化实现）
    - 主体：InstanceNorm2d 去均值/方差。
    - 条件：来自 GlobalStructureEncoder 的上下文向量 c ∈ R^{d}。
    - 方式：FiLM 风格调制 y = IN(x) * (1 + γ(c)) + β(c)。
    - 退化：无上下文时，回退到可学习仿射参数 γ0、β0。
    """
    def __init__(self, num_features: int, context_dim: int = 64, eps: float = 1e-5,
                 momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.context_dim = context_dim
        self.affine_fallback = affine

        # 以 InstanceNorm 为核，不跟踪 running stats，避免域偏移。
        self.inorm = nn.InstanceNorm2d(num_features, affine=False, eps=eps, momentum=momentum, track_running_stats=False)

        # 条件映射：c -> [γ, β]，每层独立一套小 MLP，避免跨层干扰。
        hidden = max(128, num_features * 2)
        self.cond = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_features * 2)
        )

        if self.affine_fallback:
            self.gamma0 = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.beta0 = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.register_parameter('gamma0', None)
            self.register_parameter('beta0', None)

        # 前向时由外部注入，形状 [B, context_dim]
        self._gs_context: Optional[torch.Tensor] = None

    def set_context(self, ctx: Optional[torch.Tensor]):
        self._gs_context = ctx

    def forward(self, x):
        y = self.inorm(x)
        c = self._gs_context
        # 形状不匹配则忽略旧 ctx
        if c is not None and (c.dim()!=2 or c.size(0)!=x.size(0)):
            c = None
        if c is not None:
            params = self.cond(c)
            gamma, beta = params.chunk(2, dim=1)
            gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
            beta  = torch.tanh(beta ).unsqueeze(-1).unsqueeze(-1)
            y = y * (1.0 + gamma) + beta
        elif self.affine_fallback:
            y = y * self.gamma0 + self.beta0
        return y


# ==============================
# file: models/global_structure.py
# ==============================
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
            nn.Linear(32 * 8 * 8 + 8, 128),
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


# ==============================
# file: models/convert_bn_to_ftanorm.py
# ==============================
import copy
import torch
import torch.nn as nn
from .ftanorm import FTANorm2d


def convert_bn_to_ftanorm(module: nn.Module, context_dim: int = 64) -> nn.Module:
    """
    递归地将模型中的 BatchNorm2d / InstanceNorm2d 替换为 FTANorm2d。
    尽量迁移 BN 的仿射参数到 FTANorm 的回退仿射。
    """
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            num_features = child.num_features
            ftan = FTANorm2d(num_features=num_features, context_dim=context_dim, affine=True)
            # 迁移 BN 的 weight/bias 到 affine fallback
            with torch.no_grad():
                if hasattr(ftan, 'gamma0') and child.weight is not None:
                    ftan.gamma0.data = child.wei