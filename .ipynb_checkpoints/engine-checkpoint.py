# engine.py
from typing import Iterable
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import util.misc as utils
from util.misc import MetricLogger
import functools
from tqdm import tqdm
import torch.nn.functional as F
from monai.metrics import compute_meandice
from torch.autograd import Variable
from dataloaders.saliency_balancing_fusion import get_SBF_map
from utils.context_injector import inject_global_context, clear_global_context

print = functools.partial(print, flush=True)

def train_warm_up(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, learning_rate:float, warmup_iteration: int = 1500):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    print_freq = 10
    cur_iteration=0
    while True:
        for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, 'WarmUp with max iteration: {}'.format(warmup_iteration))):
            for k,v in samples.items():
                if isinstance(samples[k],torch.Tensor):
                    samples[k]=v.to(device)
            cur_iteration+=1
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = cur_iteration/warmup_iteration*learning_rate * param_group["lr_scale"]

            img=samples['images']
            lbl=samples['labels']
            pred = model(img)
            loss_dict = criterion.get_loss(pred,lbl)
            losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            if cur_iteration>=warmup_iteration:
                print(f'WarnUp End with Iteration {cur_iteration} and current lr is {optimizer.param_groups[0]["lr"]}.')
                return cur_iteration
        metric_logger.synchronize_between_processes()

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch,
                    max_norm=0, scaler=None,
                    context_provider=None,
                    boundary_loss=None,
                    boundary_lambda: float = 0.0):

    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 20

    for samples in metric_logger.log_every(data_loader, print_freq, header):
        # 原仓库是 dict
        images = samples['images'].to(device, non_blocking=True)
        masks  = samples['labels'].to(device, non_blocking=True)

        # 注入上下文
        ctx = context_provider(images) if context_provider is not None else None
        inject_global_context(model, ctx)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss_dict = criterion.get_loss(logits, masks)
                loss_seg = sum(loss_dict[k] * criterion.weight_dict[k]
                               for k in loss_dict.keys() if k in criterion.weight_dict)
                if boundary_loss is not None and boundary_lambda > 0:
                    loss_bnd = boundary_loss(logits, masks)
                    loss = loss_seg + boundary_lambda * loss_bnd
                else:
                    loss_bnd = torch.tensor(0.0, device=device)
                    loss = loss_seg
        else:
            logits = model(images)
            loss_dict = criterion.get_loss(logits, masks)
            loss_seg = sum(loss_dict[k] * criterion.weight_dict[k]
                           for k in loss_dict.keys() if k in criterion.weight_dict)
            if boundary_loss is not None and boundary_lambda > 0:
                loss_bnd = boundary_loss(logits, masks)
                loss = loss_seg + boundary_lambda * loss_bnd
            else:
                loss_bnd = torch.tensor(0.0, device=device)
                loss = loss_seg

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        metric_logger.update(loss=float(loss.item()),
                             loss_seg=float(loss_seg.item()),
                             loss_bnd=float(loss_bnd.item()))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_SBF(model, criterion, data_loader, optimizer, device,
                        epoch, cur_iteration, max_iteration=-1, config=None, visdir=None,
                        context_provider=None, boundary_loss=None, boundary_lambda: float=0.0):

    model.train()
    criterion.train()
    clear_global_context(model)  # 避免沿用上次 ctx
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    visual_freq = 500
    for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for k, v in samples.items():
            if isinstance(samples[k], torch.Tensor):
                samples[k] = v.to(device)

        GLA_img = samples['images']
        LLA_img = samples['aug_images']
        lbl = samples['labels']
        if cur_iteration % visual_freq == 0:
            visual_dict={}
            visual_dict['GLA']=GLA_img.detach().cpu().numpy()[0,0]
            visual_dict['LLA']=LLA_img.detach().cpu().numpy()[0,0]
            visual_dict['GT']=lbl.detach().cpu().numpy()[0]
        else:
            visual_dict=None
            
        if context_provider is not None:
            ctx = context_provider(GLA_img)
            inject_global_context(model, ctx)

        input_var = Variable(GLA_img, requires_grad=True)

        optimizer.zero_grad()
        logits = model(input_var)
        loss_dict = criterion.get_loss(logits, lbl)
        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
        if boundary_loss is not None and boundary_lambda > 0:
            loss_bnd = boundary_loss(logits, lbl)
            losses = losses + boundary_lambda * loss_bnd
        losses.backward(retain_graph=True)

        # saliency
        gradient = torch.sqrt(torch.mean(input_var.grad ** 2, dim=1, keepdim=True)).detach()

        saliency=get_SBF_map(gradient,config.grid_size)

        mixed_img = GLA_img.detach() * saliency + LLA_img * (1 - saliency)
        if visual_dict is not None:
            visual_dict['GLA_pred']=torch.argmax(logits,1).cpu().numpy()[0]

        if visual_dict is not None:
            visual_dict['GLA_saliency']= saliency.detach().cpu().numpy()[0,0]

        mixed_img = GLA_img.detach() * saliency + LLA_img * (1 - saliency)
        if visual_dict is not None:
            visual_dict['SBF']= mixed_img.detach().cpu().numpy()[0,0]
            
        if context_provider is not None:
            ctx2 = context_provider(mixed_img)
            inject_global_context(model, ctx2)
            
        aug_var = Variable(mixed_img, requires_grad=False)  # 这里不再需要二次反传 saliency
        if context_provider is not None:
            ctx_sbf = context_provider(mixed_img)
            inject_global_context(model, ctx_sbf)

        aug_logits = model(aug_var)
        aug_loss_dict = criterion.get_loss(aug_logits, lbl)
        aug_losses = sum(aug_loss_dict[k] * criterion.weight_dict[k] for k in aug_loss_dict.keys() if k in criterion.weight_dict)

        if boundary_loss is not None and boundary_lambda > 0:
            aug_loss_bnd = boundary_loss(aug_logits, lbl)
            aug_losses = aug_losses + boundary_lambda * aug_loss_bnd

        aug_losses.backward()

        if visual_dict is not None:
            visual_dict['SBF_pred'] = torch.argmax(aug_logits, 1).cpu().numpy()[0]

        optimizer.step()

        all_loss_dict={}
        for k in loss_dict.keys():
            if k not in criterion.weight_dict:continue
            all_loss_dict[k]=loss_dict[k]
            all_loss_dict[k+'_aug']=aug_loss_dict[k]
        if boundary_loss is not None and boundary_lambda > 0:
            all_loss_dict['bnd'] = loss_bnd.detach()
            all_loss_dict['bnd_aug'] = aug_loss_bnd.detach()
        metric_logger.update(**all_loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


        if cur_iteration>=max_iteration and max_iteration>0:
            break

        if visdir is not None and cur_iteration%visual_freq==0:
            fs=int(len(visual_dict)**0.5)+1
            for idx, k in enumerate(visual_dict.keys()):
                plt.subplot(fs,fs,idx+1)
                plt.title(k)
                plt.axis('off')
                if k not in ['GT','GLA_pred','SBF_pred']:
                    plt.imshow(visual_dict[k], cmap='gray')
                else:
                    plt.imshow(visual_dict[k], vmin=0, vmax=4)
            plt.tight_layout()
            plt.savefig(f'{visdir}/{cur_iteration}.png')
            plt.close()
        cur_iteration+=1

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return cur_iteration


@torch.no_grad()
def evaluate(model, criterion, data_loader, device,
             context_provider=None):  # NEW：验证时也可注入上下文（或设为 None 走回退仿射）
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for samples in metric_logger.log_every(data_loader, 50, header):
        images = samples['images'].to(device, non_blocking=True)
        masks  = samples['labels'].to(device, non_blocking=True)

        ctx = context_provider(images) if context_provider is not None else None
        inject_global_context(model, ctx)

        logits = model(images)
        loss_dict = criterion.get_loss(logits, masks)
        loss = sum(loss_dict[k] * criterion.weight_dict[k]
                   for k in loss_dict if k in criterion.weight_dict)
        metric_logger.update(loss=float(loss.item()))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
@torch.no_grad()
def prediction_wrapper(model, test_loader, epoch, label_name, mode='base', save_prediction=False):
    model.eval()
    device = next(model.parameters()).device

    import torch
    def _as_bool(x):  # 接受 bool/int/str/0-d张量
        if isinstance(x, bool): return x
        if isinstance(x, (int, float)): return bool(x)
        if isinstance(x, str): return x.lower() in ('true','1','t','y','yes')
        if torch.is_tensor(x): return bool(x.item())
        return bool(x)

    def _as_int(x):
        if isinstance(x, (int, float)): return int(x)
        if torch.is_tensor(x): return int(x.item())
        return int(x)

    out_prediction_list, state = {}, {}

    from tqdm import tqdm
    for batch in tqdm(test_loader, total=len(test_loader)):
        imgs   = batch['images'].to(device)          # [B,C,H,W]
        gths   = batch['labels'].to(device)          # [B,1,H,W] 或 [B,H,W]
        sids   = batch['scan_id']                    # list[str] 或张量
        nfs    = batch['nframe']                     # list[int]/张量或标量
        is_st  = batch.get('is_start', None)         # [B] 或缺省
        is_ed  = batch.get('is_end', None)           # [B] 或缺省

        B, _, H, W = imgs.shape
        for b in range(B):
            # 逐样本取值
            sid = sids[b] if isinstance(sids, (list, tuple)) else (sids[b] if torch.is_tensor(sids) else sids)
            sid = str(sid)
            nf_b = nfs[b] if isinstance(nfs, (list, tuple)) else (nfs[b] if torch.is_tensor(nfs) and nfs.dim()==1 else nfs)
            nf_b = _as_int(nf_b)
            st_b = True if is_st is None else _as_bool(is_st[b] if isinstance(is_st,(list,tuple)) or (torch.is_tensor(is_st) and is_st.dim()==1) else is_st)
            ed_b = False if is_ed is None else _as_bool(is_ed[b] if isinstance(is_ed,(list,tuple)) or (torch.is_tensor(is_ed) and is_ed.dim()==1) else is_ed)

            # 初始化该 scan
            if st_b and sid not in state:
                state[sid] = {
                    'slice_idx': 0,
                    'pred': torch.zeros((nf_b, H, W), device=device),
                    'gth' : torch.zeros((nf_b, H, W), device=device)
                }
                out_prediction_list.setdefault(sid, {})

            # 前向
            x = imgs[b:b+1]
            y = gths[b:b+1]
            logit = model(x)
            pred = torch.argmax(logit, dim=1)                 # [1,H,W]
            y2d  = y[:,0] if y.dim()==4 and y.size(1)==1 else y.squeeze(0) if y.dim()==3 else y
            y2d  = y2d.long()

            st = state[sid]
            idx = st['slice_idx']
            # 防御：若 nf_b 低估，扩容
            if idx >= st['pred'].size(0):
                new_n = max(idx+1, st['pred'].size(0)*2)
                st['pred'] = torch.cat([st['pred'], torch.zeros((new_n-st['pred'].size(0), H, W), device=device)], 0)
                st['gth']  = torch.cat([st['gth'],  torch.zeros((new_n-st['gth'].size(0),  H, W), device=device)], 0)

            st['pred'][idx] = pred[0]
            st['gth'][idx]  = y2d if y2d.dim()==2 else y2d[0]
            st['slice_idx'] = idx + 1

            if ed_b:
                used = st['slice_idx']
                out_prediction_list[sid]['pred'] = st['pred'][:used]
                out_prediction_list[sid]['gth']  = st['gth'][:used]
                del state[sid]

    print(f"Epoch {epoch} test result on mode {mode} segmentation are shown as follows:")
    error_dict, dsc_table, domain_names = eval_list_wrapper(out_prediction_list, len(label_name), label_name)
    error_dict["mode"] = mode
    if not save_prediction:
        del out_prediction_list
        out_prediction_list = []
    torch.cuda.empty_cache()
    
    return out_prediction_list, dsc_table, error_dict, domain_names

    

def eval_list_wrapper(vol_list, nclass, label_name):
    """
    Evaluatation and arrange predictions
    """
    def convert_to_one_hot2(tensor,num_c):
        return F.one_hot(tensor.long(),num_c).permute((3,0,1,2)).unsqueeze(0)

    out_count = len(vol_list)
    tables_by_domain = {} # tables by domain
    dsc_table = np.ones([ out_count, nclass ]  ) # rows and samples, columns are structures
    idx = 0
    for scan_id, comp in vol_list.items():
        domain, pid = scan_id.split("_")
        if domain not in tables_by_domain.keys():
            tables_by_domain[domain] = {'scores': [],'scan_ids': []}
        pred_ = comp['pred']
        gth_  = comp['gth']
        dices=compute_meandice(y_pred=convert_to_one_hot2(pred_,nclass),y=convert_to_one_hot2(gth_,nclass),include_background=True).cpu().numpy()[0].tolist()

        tables_by_domain[domain]['scores'].append( [_sc for _sc in dices]  )
        tables_by_domain[domain]['scan_ids'].append( scan_id )
        dsc_table[idx, ...] = np.reshape(dices, (-1))
        del pred_
        del gth_
        idx += 1
        torch.cuda.empty_cache()

    # then output the result
    error_dict = {}
    for organ in range(nclass):
        mean_dc = np.mean( dsc_table[:, organ] )
        std_dc  = np.std(  dsc_table[:, organ] )
        print("Organ {} with dice: mean: {:06.5f}, std: {:06.5f}".format(label_name[organ], mean_dc, std_dc))
        error_dict[label_name[organ]] = mean_dc
    print("Overall std dice by sample {:06.5f}".format(dsc_table[:, 1:].std()))
    print("Overall mean dice by sample {:06.5f}".format( dsc_table[:,1:].mean())) # background is noted as class 0 and therefore not counted
    error_dict['overall'] = dsc_table[:,1:].mean()

    # then deal with table_by_domain issue
    overall_by_domain = []
    domain_names = []
    for domain_name, domain_dict in tables_by_domain.items():
        domain_scores = np.array( tables_by_domain[domain_name]['scores']  )
        domain_mean_score = np.mean(domain_scores[:, 1:])
        error_dict[f'domain_{domain_name}_overall'] = domain_mean_score
        error_dict[f'domain_{domain_name}_table'] = domain_scores
        overall_by_domain.append(domain_mean_score)
        domain_names.append(domain_name)
    print('per domain resutls:', overall_by_domain)
    error_dict['overall_by_domain'] = np.mean(overall_by_domain)

    print("Overall mean dice by domain {:06.5f}".format( error_dict['overall_by_domain'] ) )
    return error_dict, dsc_table, domain_names
