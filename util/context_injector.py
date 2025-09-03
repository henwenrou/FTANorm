import torch
import torch.nn as nn
from typing import Optional
from models.ftanorm import FTANorm2d


def inject_global_context(module: nn.Module, ctx: Optional[torch.Tensor]):
    """将上下文向量注入到所有 FTANorm2d 层。ctx: [B, D] 或 None。"""
    for m in module.modules():
        if isinstance(m, FTANorm2d):
            m.set_context(ctx)