import os
import sys
import shutil
import time
import numpy as np
from optparse import OptionParser
from shutil import copyfile
from tqdm import tqdm

from utils import vararg_callback_bool, vararg_callback_int, get_config
from dataloader import  *

import torch
from engine import omni_engine

sys.setrecursionlimit(40000)


def get_args_parser():
    parser = OptionParser()

    parser.add_option("--GPU", dest="GPU", help="the index of gpu is used", default=None, action="callback",
                      callback=vararg_callback_int)
    parser.add_option("--model", dest="model_name", help="swin_base|swin_large|swin_large_384|swin_large_768|conv_base", default="swin_base", type="string")
    parser.add_option("--init", dest="init",help="Random| ImageNet_1k| ImageNet_21k| SAM| DeiT| BEiT| DINO| MoCo_V3| MoBY | MAE| SimMIM", default="Random", type="string")
    parser.add_option("--pretrained_weights", dest="pretrained_weights", help="Path to the Pretrained model", default=None, type="string")
    parser.add_option("--num_class", dest="num_class", help="number of the classes in the downstream task",
                      default=14, type="int")
    parser.add_option("--data_set", dest="dataset_list", help="ChestXray14|CheXpert|Shenzhen|VinDrCXR|RSNAPneumonia",  action="append")
    parser.add_option("--normalization", dest="normalization", help="how to normalize data (imagenet|chestx-ray)", default="imagenet",
                      type="string")
    parser.add_option("--input_size", dest="crop_size", help="input resolution", default=224, type="int")
    parser.add_option("--img_resize", dest="resize", help="resize image resolution", default=256, type="int")
    parser.add_option("--img_depth", dest="img_depth", help="num of image depth", default=3, type="int")
    parser.add_option("--batch_size", dest="batch_size", help="batch size", default=32, type="int")
    parser.add_option("--epochs", dest="epochs", help="num of epoches", default=200, type="int")
    parser.add_option("--exp_name", dest="exp_name", default="", type="string")
    parser.add_option("--mode", dest="mode", help="train | test", default="train", type="string")
    
    parser.add_option("--ema_mode", dest="ema_mode", default="epoch", help="update teacher model at which time (epoch | iteration)", type="string")
    parser.add_option('--momentum_teacher', default=0.9, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_option("--pretrain_epochs", dest="pretrain_epochs", help="num of omni-pretraining epoches", default=10, type="int")
    parser.add_option("--test_epoch", dest="test_epoch", help="whether test after every epoch", default=1, type="int")                
    parser.add_option("--val_loss_metric", dest="val_loss_metric", help="which validation loss for early stop and model save (average | [dataset])", default="average", type="string")                  
    parser.add_option("--projector_features", dest="projector_features", help="num of projector features", default=None, type="int")
    parser.add_option("--use_mlp", dest="use_mlp", help="whether use mlp for projector", default=False, action="callback",
                      callback=vararg_callback_bool)
    # Optimizer parameters
    parser.add_option('--opt', default='momentum', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_option('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_option('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_option('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_option('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_option('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_option('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_option('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_option('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_option('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_option('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_option('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_option('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_option('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_option('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_option('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_option('--decay-rate', '--dr', type=float, default=0.5, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_option('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')

    parser.add_option("--resume", dest="resume", help="whether latest checkpoint", default=False, action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--workers", dest="workers", help="number of CPU workers", default=8, type="int")
    parser.add_option("--print_freq", dest="print_freq", help="print frequency", default=50, type="int")
    parser.add_option("--test_augment", dest="test_augment", help="whether use test time augmentation",
                      default=True, action="callback", callback=vararg_callback_bool)
    parser.add_option("--anno_percent", dest="anno_percent", help="data percent", default=100, type="int")
    parser.add_option("--device", dest="device", help="cpu|cuda", default="cuda", type="string")
    parser.add_option("--activate", dest="activate", help="Sigmoid", default="Sigmoid", type="string")
    parser.add_option("--uncertain_label", dest="uncertain_label",
                      help="the label assigned to uncertain data (Ones | Zeros | LSR-Ones | LSR-Zeros)",
                      default="LSR-Ones", type="string")
    parser.add_option("--unknown_label", dest="unknown_label", help="the label assigned to unknown data",
                      default=0, type="int")


    # ---------------- PATCH: options added for the MedMNIST replication ----------------
    parser.add_option("--data_root", dest="data_root", help="folder that holds chestmnist/, dermamnist/, ... (.npy)", default="./data", type="string")
    parser.add_option("--output_root", dest="output_root", help="where Models/ and Outputs/ are written", default=".", type="string")
    parser.add_option("--config", dest="config", help="datasets config yaml", default="datasets_config.yaml", type="string")
    parser.add_option("--accum_steps", dest="accum_steps", help="gradient accumulation steps (effective batch = batch_size * accum_steps)", default=1, type="int")
    parser.add_option("--eval_batch_size", dest="eval_batch_size", help="batch size for validation/test (0 = batch_size)", default=0, type="int")
    parser.add_option("--amp", dest="amp", help="float16 mixed precision on CUDA", default=False, action="callback", callback=vararg_callback_bool)
    parser.add_option("--reinit_heads", dest="reinit_heads", help="drop task heads when resuming", default=False, action="callback", callback=vararg_callback_bool)
    parser.add_option("--sched_step_mode", dest="sched_step_mode", help="original: step(val_loss) exactly as released | epoch: step(epoch+1)", default="original", type="string")
    parser.add_option("--early_stop_patience", dest="early_stop_patience", help="stop after N epochs without val improvement (0 = off)", default=0, type="int")
    parser.add_option("--early_stop_min_delta", dest="early_stop_min_delta", help="minimum decrease that counts as improvement", default=0.0, type="float")
    parser.add_option("--early_stop_min_epochs", dest="early_stop_min_epochs", help="never stop before this many epochs", default=0, type="int")
    parser.add_option("--time_limit_hours", dest="time_limit_hours", help="stop cleanly before this wall-clock limit (0 = off)", default=0.0, type="float")
    parser.add_option("--final_eval_reserve_min", dest="final_eval_reserve_min", help="minutes kept free for the final test when a time limit is set", default=45.0, type="float")
    parser.add_option("--keep_periodic_ckpt", dest="keep_periodic_ckpt", help="also keep a checkpoint every test_epoch (about 1 GB each)", default=False, action="callback", callback=vararg_callback_bool)
    parser.add_option("--save_debug_images", dest="save_debug_images", help="dump augmented images every print_freq steps", default=False, action="callback", callback=vararg_callback_bool)
    parser.add_option("--limit_train", dest="limit_train", help="smoke test: use only N random training images per dataset (0 = all)", default=0, type="int")
    parser.add_option("--limit_eval", dest="limit_eval", help="smoke test: use only N random val/test images per dataset (0 = all)", default=0, type="int")
    parser.add_option("--max_train_steps", dest="max_train_steps", help="smoke test: max micro-batches per dataset per epoch (0 = all)", default=0, type="int")
    parser.add_option("--max_eval_steps", dest="max_eval_steps", help="smoke test: max batches per val/test pass (0 = all)", default=0, type="int")
    parser.add_option("--slice_mode", dest="slice_mode", help="3D volumes to one RGB image: center3 | orthogonal", default="center3", type="string")
    parser.add_option("--seed", dest="seed", help="random seed", default=0, type="int")

    (options, args) = parser.parse_args()

    return options


def main(args):
    print(args)
    # PATCH: seed (the original does not seed; kept optional so runs stay comparable)
    if args.seed:
        import random
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    exp_name = args.model_name + "_" + args.exp_name
    model_path = os.path.join(args.output_root, "Models", exp_name)
    output_path = os.path.join(args.output_root, "Outputs", exp_name)

    datasets_config = get_config(args.config)
    for dataset in args.dataset_list:
        assert dataset in list(datasets_config.keys()), dataset
        # PATCH: fill in the data root
        datasets_config[dataset]['data_dir'] = datasets_config[dataset]['data_dir'].replace("<DATA_ROOT>", args.data_root)

    def build(dataset, split_key, augment, limit):
        cfg = datasets_config[dataset]
        kw = dict(images_path=cfg['data_dir'], file_path=cfg[split_key], crop_size=args.crop_size, resize=args.resize, augment=augment)
        if dict_dataloarder[dataset] in (MedMNIST, MedMNIST3D):
            kw.update(task_type=cfg['task_type'], num_class=len(cfg['diseases']), limit=limit)
        if dict_dataloarder[dataset] is MedMNIST3D:
            kw.update(slice_mode=args.slice_mode)
        return dict_dataloarder[dataset](**kw)

    dataset_train_list = []
    dataset_val_list = []
    dataset_test_list = []
    for dataset in args.dataset_list:
        dataset_train_list.append(build(dataset, 'train_list', None, args.limit_train))
        dataset_val_list.append(build(dataset, 'val_list', build_transform_classification(normalize=args.normalization, crop_size=args.crop_size, resize=args.resize, mode="valid"), args.limit_eval))
        dataset_test_list.append(build(dataset, 'test_list', build_transform_classification(normalize=args.normalization, crop_size=args.crop_size, resize=args.resize, mode="test", test_augment=args.test_augment), args.limit_eval))

    return omni_engine(args, model_path, output_path, args.dataset_list, datasets_config, dataset_train_list, dataset_val_list, dataset_test_list)

if __name__ == '__main__':
    args = get_args_parser()
    main(args)

