import argparse, os, sys, datetime, importlib
os.environ['KMP_DUPLICATE_LIB_OK']='true'
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "23")
import torch.optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from engine import train_warm_up,evaluate,train_one_epoch,train_one_epoch_SBF,prediction_wrapper
from losses import SetCriterion
import numpy as np
import random
from torch.optim import lr_scheduler

from models.convert_bn_to_ftanorm import convert_bn_to_ftanorm
from models.global_structure import GlobalStructureEncoder
from losses.boundary_loss import BoundaryLoss


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def seed_everything(seed=None):
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    try:
        if seed is None:
            seed = os.environ.get("PL_GLOBAL_SEED", random.randint(min_seed_value, max_seed_value))
        seed = int(seed)
    except (TypeError, ValueError):
        seed = random.randint(min_seed_value, max_seed_value)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f'training seed is {seed}')
    return seed

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    return parser

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class DataModuleFromConfig(torch.nn.Module):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers)

torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    opt = args
    seed=seed_everything(args.seed)
    if args.resume:
        if not os.path.exists(args.resume):
            raise ValueError("Cannot find {}".format(args.resume))
    if args.base:
        cfg_fname = os.path.split(args.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = "_" + cfg_name
    else:
        name=None
        raise ValueError('no config')

    nowname = now +f'_seed{seed}'+ name + args.postfix
    logdir = os.path.join("logs", nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    visdir= os.path.join(logdir, "visuals")
    for d in [logdir, cfgdir, ckptdir,visdir ]:
        os.makedirs(d, exist_ok=True)

    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    OmegaConf.save(config,os.path.join(cfgdir, "{}-project.yaml".format(now)))

    model_config = config.pop("model", OmegaConf.create())
    optimizer_config = config.pop('optimizer', OmegaConf.create())
    SBF_config = config.pop('saliency_balancing_fusion', OmegaConf.create())

    # 实例化模型（你之前注释掉了，必须恢复）
    model = instantiate_from_config(model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = None  # 如需要 AMP 再启用 GradScaler
    # 设备与混精
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = None  # 如需AMP可自行创建 torch.cuda.amp.GradScaler()
    
    context_dim = 64
    boundary_lambda = 0.2
    # 从配置拉 in_ch；没有就默认为 1
    in_ch = int(OmegaConf.select(config, 'data.params.in_ch') or 1)
    
    model = convert_bn_to_ftanorm(model, context_dim=context_dim).to(device)
    gse = GlobalStructureEncoder(in_ch=in_ch, context_dim=context_dim).to(device)
    boundary_loss = BoundaryLoss().to(device)


    def context_provider(images):
        return gse(images)
    
    # 优化器变量统一用 opt（你上面就是 opt）
    optimizer = opt  # 为了和 engine 形参一致，也可以直接把下方调用里的 optimizer 改成 opt

    if torch.cuda.is_available():
        model=model.cuda()

    if getattr(model_config.params, 'base_learning_rate') :
        bs, base_lr = config.data.params.batch_size, optimizer_config.base_learning_rate
        lr = bs * base_lr
    else:
        bs, lr = config.data.params.batch_size, optimizer_config.learning_rate

    if getattr(model_config.params, 'pretrain') :
        param_dicts = model.optim_parameters()
    else:
        param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad], "lr_scale": 1}]

    opt_params = {'lr': lr}
    for k in ['momentum', 'weight_decay']:
        if k in optimizer_config:
            opt_params[k] = optimizer_config[k]

    criterion = SetCriterion()

    print('optimization parameters: ', opt_params)
     # 把 GSE 参数也训练
    param_dicts.append({"params": [p for p in gse.parameters() if p.requires_grad], "lr_scale": 1})
    optimizer = eval(optimizer_config['target'])(param_dicts, **opt_params)

    if optimizer_config.lr_scheduler =='lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + 0 - 50) / float(optimizer_config.max_epoch-50 + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        scheduler=None
        print('We follow the SSDG learning rate schedule by default, you can add your own schedule by yourself')
        raise NotImplementedError

    assert optimizer_config.max_epoch > 0 or optimizer_config.max_iter > 0
    if optimizer_config.max_iter > 0:
        max_epoch=999
        print('detect identified max iteration, set max_epoch to 999')
    else:
        max_epoch= optimizer_config.max_epoch

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    print(len(data.datasets["train"]))
    train_loader=DataLoader(data.datasets["train"], batch_size=data.batch_size,
                          num_workers=data.num_workers, shuffle=True, persistent_workers=True, drop_last=True, pin_memory = True)

    val_loader=DataLoader(data.datasets["validation"], batch_size=data.batch_size,  num_workers=1)

    if data.datasets.get('test') is not None:
        test_loader=DataLoader(data.datasets["test"], batch_size=1, num_workers=1)
        best_test_dice = 0
        test_phase=True
    else:
        test_phase=False

    if getattr(optimizer_config, 'warmup_iter'):
        if optimizer_config.warmup_iter>0:
            train_warm_up(model, criterion, train_loader, opt, torch.device('cuda'), lr, optimizer_config.warmup_iter)
    cur_iter=0
    best_dice=0
    label_name=data.datasets["train"].all_label_names
    for cur_epoch in range(max_epoch):
        iters_per_epoch = len(train_loader)
        
        if SBF_config.usage:
            # 用原来的 SBF 版本，它返回累计迭代数
            cur_iter = train_one_epoch_SBF(
                model, criterion, train_loader, optimizer, device,
                cur_epoch, cur_iter,
                max_iteration=optimizer_config.max_iter,
                config=SBF_config, visdir=visdir
            )
        else:
            _stats = train_one_epoch(
                model, criterion, train_loader, optimizer, device, cur_epoch,
                max_norm=0, scaler=scaler,
                context_provider=context_provider,
                boundary_loss=boundary_loss, boundary_lambda=boundary_lambda,
            )
            cur_iter += iters_per_epoch
        if scheduler is not None:
            scheduler.step()

        # Save Bset model on val
        if (cur_epoch+1)%3==0:
            cur_dice = evaluate(
    model, criterion, val_loader, device,
    context_provider=context_provider,  # 或传 None 走回退仿射
)
            if np.mean(cur_dice)>best_dice:
                best_dice=np.mean(cur_dice)
                for f in os.listdir(ckptdir):
                    if 'val' in f:
                        os.remove(os.path.join(ckptdir,f))
                torch.save({'model': model.state_dict()}, os.path.join(ckptdir,f'val_best_epoch_{cur_epoch}.pth'))

            str=f'Epoch [{cur_epoch}]   '
            for i,d in enumerate(cur_dice):
                str+=f'Class {i}: {d}, '
            str+=f'Validation DICE {np.mean(cur_dice)}/{best_dice}'
            print(str)

        if (cur_epoch+1) % 3 == 0:
            _, dsc_table, err, domains = prediction_wrapper(
                model, val_loader, cur_epoch, label_name, mode='val', save_prediction=False)
            cur_dice = float(err['overall'])  # 使用 overall
            if cur_dice > best_dice:
                best_dice = cur_dice
                for f in os.listdir(ckptdir):
                    if 'val' in f: os.remove(os.path.join(ckptdir, f))
                torch.save({'model': model.state_dict()},
                           os.path.join(ckptdir, f'val_best_epoch_{cur_epoch}.pth'))
            print(f"Epoch [{cur_epoch}]  Val overall Dice {cur_dice:.4f} / best {best_dice:.4f}")

        # Save latest model
        if (cur_epoch+1)%50==0:
            torch.save({'model': model.state_dict()}, os.path.join(ckptdir,'latest.pth'))

        if cur_iter >= optimizer_config.max_iter and optimizer_config.max_iter>0:
            torch.save({'model': model.state_dict()}, os.path.join(ckptdir, 'latest.pth'))
            print(f'End training with iteration {cur_iter}')
            break
