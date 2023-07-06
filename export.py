# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
The main training/evaluation loop
Modified from: https://github.com/facebookresearch/deit
"""
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
import utils

import xcit
model_name = 'xcit_nano_12_p8' #'' #'xcit_tiny_12_p8' 

def get_args_parser():
    parser = argparse.ArgumentParser('XCiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=5, type=int)
    parser.add_argument('--epochs', default=40, type=int)

    # Model parameters
    parser.add_argument('--model', default=f'{model_name}', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.05, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")


    # Dataset parameters
    parser.add_argument('--data-path', default='datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR10', 'CIFAR100', 'IMNET',
                                                                'INAT', 'INAT19', 'CARS', 'FLOWERS',
                                                                'IMNET22k'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default=f'experiments/{model_name}/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--test-freq', default=1, type=int, help='Number of epochs between \
                                                                  validation runs.')

    # parser.add_argument('--full_crop', action='store_true', help='use crop_ratio=1.0 instead of the\
    #                                                               default 0.875 (Used by CaiT).')
    parser.add_argument("--pretrained", default=None, type=str, help='Path to pre-trained checkpoint')
    parser.add_argument('--surgery', default=None, type=str, help='Path to checkpoint to copy the \
                                                                   patch projection from. \
                                                                   Can improve stability for very \
                                                                   large models.')

    return parser

parser = argparse.ArgumentParser('XCiT training and evaluation script', parents=[get_args_parser()])
args = parser.parse_known_args()[0]
if args.output_dir:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

utils.init_distributed_mode(args)
print(args)

device = torch.device(args.device)

# fix the seed for reproducibility
seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)

cudnn.benchmark = True

# dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
from timm.data import create_transform
is_train=False
resize_im = args.input_size > 32

transform = create_transform(
    input_size=args.input_size,
    is_training=True,
    color_jitter=args.color_jitter,
    auto_augment=args.aa,
    interpolation=args.train_interpolation,
    re_prob=args.reprob,
    re_mode=args.remode,
    re_count=args.recount,
)

from torchvision import datasets, transforms
root = os.path.join(args.data_path, 'train' if is_train else 'val')
dataset_train = datasets.ImageFolder(root, transform=transform)
nb_classes = 1000

args.nb_classes = nb_classes


# dataset_val, _ = build_dataset(is_train=False, args=args)
# transform = build_transform(False, args)

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
transform = {}
transform = transforms.Compose(
    [transforms.Resize(args.input_size, interpolation=3),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
root = os.path.join(args.data_path, 'val')
dataset_val = datasets.ImageFolder(root, transform=transform)

num_tasks = utils.get_world_size()
global_rank = utils.get_rank()

sampler_train = RASampler(
    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)

data_loader_train = torch.utils.data.DataLoader(
    dataset_train, sampler=sampler_train,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=True,
)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val, sampler=sampler_val,
    batch_size=int(1.5 * args.batch_size),
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=False
)

mixup_fn = None
mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
if mixup_active:
    mixup_fn = Mixup(
        mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
        prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
        label_smoothing=args.smoothing, num_classes=args.nb_classes)

print(f"Creating model: {args.model}")

model = create_model(
    args.model,
    pretrained=False,
    num_classes=args.nb_classes,
    drop_rate=args.drop,
    drop_path_rate=args.drop_path,
    drop_block_rate=None
)
model.to(device)

# Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
model_ema = ModelEma(
    model,
    decay=args.model_ema_decay,
    device='cpu' if args.model_ema_force_cpu else '',
    resume='')

model_without_ddp = model
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)

linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
args.lr = linear_scaled_lr
optimizer = create_optimizer(args, model_without_ddp)
loss_scaler = NativeScaler()

lr_scheduler, _ = create_scheduler(args, optimizer)

criterion = LabelSmoothingCrossEntropy()

# smoothing is handled with mixup label transform
criterion = SoftTargetCrossEntropy()

teacher_model = None
# wrap the criterion in our custom DistillationLoss, which
# just dispatches to the original criterion if args.distillation_type is 'none'
criterion = DistillationLoss(
    criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
)

output_dir = Path(args.output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

resume_path = os.path.join(output_dir, f"{model_name}_224.pth")

print("Loading from checkpoint ...")
checkpoint = torch.load(resume_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])

args.eval = True
from timm.utils import accuracy

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images, target = batch[0], batch[1]

        images = images.to(device, non_blocking=True)
        # To convert folder sequence numbers to labels.
        target = torch.tensor([int(dataset_val.classes[t]) for t in target]).to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)

        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

test_stats = evaluate(data_loader_val, model, device)
print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

import onnx
from onnxsim import simplify

# Export the model
x = torch.randn(1, 3, args.input_size, args.input_size, requires_grad=False).to(device)
torch.onnx.export(model, 
                  x, 
                  args.model+".onnx", 
                  export_params=True, 
                  opset_version=11,
                  do_constant_folding=True)


onnx_model = onnx.load("xcit_nano_12_p8.onnx")
onnx.checker.check_model(onnx_model)

model_simp, check = simplify(onnx_model)
onnx.save(model_simp, "xcit_nano_12_p8.onnx")