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
                    ftan.gamma0.data = child.weight.view(1, -1, 1, 1).data.clone()
                if hasattr(ftan, 'beta0') and child.bias is not None:
                    ftan.beta0.data = child.bias.view(1, -1, 1, 1).data.clone()
            setattr(module, name, ftan)
        else:
            convert_bn_to_ftanorm(child, context_dim=context_dim)
    return module