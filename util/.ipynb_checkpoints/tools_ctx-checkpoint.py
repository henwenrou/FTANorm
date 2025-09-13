# tools_ctx.py
import torch
from models.ftanorm import FTANorm2d  # 需保证有 set_context(self, ctx)

def inject_global_context(module: torch.nn.Module, ctx):
    for m in module.modules():
        if isinstance(m, FTANorm2d):
            m.set_context(ctx)

def clear_global_context(module: torch.nn.Module):
    for m in module.modules():
        if isinstance(m, FTANorm2d):
            m.set_context(None)