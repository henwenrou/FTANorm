import torch
from torch import nn
from models.global_structure import GlobalStructureEncoder
from models.convert_bn_to_ftanorm import convert_bn_to_ftanorm
from utils.context_injector import inject_global_context
from losses.boundary_loss import BoundaryLoss

class FTANormTrainerAdapter:
    def __init__(self, model: nn.Module, in_ch: int = 1, context_dim: int = 64, boundary_lambda: float = 0.2):
        self.model = convert_bn_to_ftanorm(model, context_dim=context_dim)
        self.gse = GlobalStructureEncoder(in_ch=in_ch, context_dim=context_dim)
        self.boundary_loss = BoundaryLoss()
        self.boundary_lambda = boundary_lambda

    def to(self, device):
        self.model = self.model.to(device)
        self.gse = self.gse.to(device)
        return self

    def train_step(self, batch, seg_criterion: nn.Module, optimizer: torch.optim.Optimizer):
        images, masks = batch  # 按你的 DataLoader 输出调整
        images = images.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        ctx = self.gse(images)  # [B,D]
        inject_global_context(self.model, ctx)

        logits = self.model(images)
        loss_seg = seg_criterion(logits, masks)
        loss_bnd = self.boundary_loss(logits, masks)
        loss = loss_seg + self.boundary_lambda * loss_bnd

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return {
            'loss': float(loss.item()),
            'loss_seg': float(loss_seg.item()),
            'loss_bnd': float(loss_bnd.item())
        }
