import torch
import torch.nn as nn
from typing import Optional
from models.ftanorm import FTANorm2d

def inject_global_context(module, ctx):
    for m in module.modules():
        if isinstance(m, FTANorm2d):
            m.set_context(ctx)

def clear_global_context(module):
    for m in module.modules():
        if isinstance(m, FTANorm2d):
            m.set_context(None)