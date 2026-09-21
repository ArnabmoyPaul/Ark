"""
engine.py for Ark+ on MedMNIST.

Same training loop as Ark_Plus/Pretraining/engine.py: every epoch the student is
trained for one pass on each dataset in turn (cyclic pretraining), the teacher is
updated by EMA after each dataset, then validation loss is computed on every
dataset and the checkpoint is saved.

Additions (all listed in PATCHES.md):
  * resume that actually works (the original crashed on args.reinit_heads) and is
    portable between 1 GPU and 2 GPUs (state dicts saved without "module.")
  * atomic checkpoint writes, so a killed session never leaves a broken file
  * early stopping on the validation metric (patience N epochs)
  * best checkpoint + final test of the best teacher and student
  * history.json / final_results.json for plotting
  * optional time limit so a Kaggle session stops cleanly before the 12 h cut-off
"""
import os
import sys
import json
import time
import copy
import math
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from models import build_omni_model
from utils import metric_AUROC, cosine_scheduler
from trainer import train_one_epoch, test_classification, evaluate, count_bn_buffers

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

sys.setrecursionlimit(40000)


# ----------------------------------------------------------------------------- helpers
def _unwrap(m):
    return m.module if isinstance(m, torch.nn.DataParallel) else m


def _strip_module(state_dict):
    return {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}


def _atomic_save(state, path):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _medmnist_acc(y, p, task_type):
    """Accuracy with the MedMNIST evaluator's definition."""
    y = y.cpu().numpy()
    p = p.cpu().numpy()
    if task_type == "multi-class classification":
        return float(accuracy_score(np.argmax(y, 1), np.argmax(p, 1)))
    if task_type == "binary classification":
        return float(accuracy_score(y[:, 0], (p[:, 0] > 0.5).astype(int)))
    # multi-label: mean over labels of per-label accuracy at threshold 0.5
    return float(np.mean([accuracy_score(y[:, i], (p[:, i] > 0.5).astype(int)) for i in range(y.shape[1])]))


def _test_all(net, dataset_list, datasets_config, loaders, device, use_amp, max_steps=0):
    """AUC exactly as the original engine computes it (metric_AUROC, mean over classes)."""
    out = {}
    for i, dataset in enumerate(dataset_list):
        cfg = datasets_config[dataset]
        diseases = cfg['diseases']
        multiclass = cfg['task_type'] == "multi-class classification"
        y, p = test_classification(net, i, loaders[i], device, multiclass, use_amp=use_amp, max_steps=max_steps)
        if 'test_diseases_name' in cfg:
            idx = [diseases.index(c) for c in cfg['test_diseases_name']]
            y_auc, p_auc, n_cls = y[:, idx], p[:, idx], len(idx)
        else:
            y_auc, p_auc, n_cls = y, p, len(diseases)
        per_class = [float(v) for v in metric_AUROC(y_auc, p_auc, n_cls)]
        out[dataset] = {
            "task_type": cfg['task_type'],
            "n_test": int(y.shape[0]),
            "auc": float(np.nanmean(per_class)) if per_class else float("nan"),
            "auc_per_class": per_class,
            "acc": _medmnist_acc(y, p, cfg['task_type']),
        }
    return out


# ----------------------------------------------------------------------------- engine
def omni_engine(args, model_path, output_path, dataset_list, datasets_config, dataset_train_list, dataset_val_list, dataset_test_list):
    process_start = time.time()
    device = torch.device(args.device)
    cudnn.benchmark = True

    # logs (same folder layout as the original)
    exp = 'Ark_Plus'
    for dataset in dataset_list:
        exp += '_' + dataset
    model_path = os.path.join(model_path, exp)
    model_path = os.path.join(model_path, args.exp_name)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(model_path, "train.log")
    output_file = os.path.join(output_path, exp + "_" + args.exp_name + "_results.txt")
    history_file = os.path.join(model_path, "history.json")
    periodic_test_file = os.path.join(model_path, "periodic_test.json")
    final_file = os.path.join(model_path, "final_results.json")
    status_file = os.path.join(model_path, "status.json")
    save_model_path = os.path.join(model_path, exp)          # latest: <exp>.pth.tar (original name)
    best_model_path = save_model_path + "_best.pth.tar"
    latest_model_path = save_model_path + ".pth.tar"

    # dataloaders
    # PATCH: batch_size is the physical (micro) batch; effective batch = batch_size * accum_steps.
    # Evaluation batch size is separate because it does not affect results (no BatchNorm).
    eval_bs = args.eval_batch_size if args.eval_batch_size > 0 else args.batch_size
    # (no persistent_workers: 4 datasets x 3 splits would keep 12 x workers processes alive)
    dl_kw = dict(num_workers=args.workers, pin_memory=True)
    data_loader_list_train = [DataLoader(dataset=d, batch_size=args.batch_size, shuffle=True, **dl_kw) for d in dataset_train_list]
    data_loader_list_val = [DataLoader(dataset=d, batch_size=eval_bs, shuffle=False, **dl_kw) for d in dataset_val_list]
    data_loader_list_test = [DataLoader(dataset=d, batch_size=max(1, int(eval_bs / 2)), shuffle=False, **dl_kw) for d in dataset_test_list]

    num_classes_list = [len(datasets_config[dataset]['diseases']) for dataset in dataset_list]
    print("num_classes_list:", num_classes_list)
    for name, d in zip(dataset_list, dataset_train_list):
        print("  {:<12s} train={:>6d}".format(name, len(d)))

    # training setups (unchanged: student and teacher are both built from the pretrained weights)
    model = build_omni_model(args, num_classes_list)
    teacher = build_omni_model(args, num_classes_list)
    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        teacher = torch.nn.DataParallel(teacher)
    model.to(device)
    teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.model_name} network. GPUs used: {max(n_gpu, 1) if device.type == 'cuda' else 0}")
    n_bn = count_bn_buffers(model)
    print(f"BatchNorm running-stat buffers in backbone: {n_bn}"
          + ("  (LayerNorm model: BN-EMA has nothing to average)" if n_bn == 0 else "  (BN-EMA active)"))

    # momentum parameter is increased to 1. during training with a cosine schedule (unchanged)
    if args.ema_mode == "epoch":
        momentum_schedule = cosine_scheduler(args.momentum_teacher, 1, args.pretrain_epochs, len(dataset_list))
    elif args.ema_mode == "iteration":
        iters_per_epoch = 0
        for d in data_loader_list_train:
            iters_per_epoch += math.ceil(len(d) / args.accum_steps)  # optimizer steps
        momentum_schedule = cosine_scheduler(args.momentum_teacher, 1, args.pretrain_epochs, iters_per_epoch)
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"AMP (float16): {use_amp} | micro-batch {args.batch_size} x accum {args.accum_steps} = effective batch {args.batch_size * args.accum_steps}")

    # state
    start_epoch = 0
    best_metric = float("inf")
    best_epoch = -1
    no_improve = 0
    stopped_early = False
    history = []
    periodic_test = []

    def _state(epoch, val_loss_list, full=True):
        s = {
            'epoch': epoch,
            'lossMIN': val_loss_list,
            'state_dict': _unwrap(model).state_dict(),
            'teacher': _unwrap(teacher).state_dict(),
            'best_metric': best_metric,
            'best_epoch': best_epoch,
            'no_improve': no_improve,
            'stopped_early': stopped_early,
            'dataset_list': list(dataset_list),
            'args': {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, list, type(None)))},
        }
        if full:
            s['optimizer'] = optimizer.state_dict()
            s['scheduler'] = lr_scheduler.state_dict()
            s['scaler'] = scaler.state_dict() if scaler is not None else None
        return s

    if args.mode == "train" and args.resume:
        resume = latest_model_path
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
            state_dict = _strip_module(checkpoint['state_dict'])
            teacher_state_dict = _strip_module(checkpoint['teacher'])
            if args.reinit_heads:
                for k in list(state_dict.keys()):
                    if k.startswith('omni_heads.'):
                        print(f"Removing key {k} from pretrained checkpoint")
                        del state_dict[k]
            _unwrap(model).load_state_dict(state_dict, strict=not args.reinit_heads)
            _unwrap(teacher).load_state_dict(teacher_state_dict, strict=True)
            if 'scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['scheduler'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if scaler is not None and checkpoint.get('scaler'):
                scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint['epoch'] + 1
            best_metric = checkpoint.get('best_metric', best_metric)
            best_epoch = checkpoint.get('best_epoch', best_epoch)
            no_improve = checkpoint.get('no_improve', 0)
            stopped_early = checkpoint.get('stopped_early', False)
            if os.path.isfile(history_file):
                history = [h for h in json.load(open(history_file)) if h["epoch"] < start_epoch]
            if os.path.isfile(periodic_test_file):
                periodic_test = [h for h in json.load(open(periodic_test_file)) if h["epoch"] < start_epoch]
            print("=> resumed from epoch {:04d}; continuing at epoch {:04d} (best epoch {} , no-improve {}/{})".format(
                checkpoint['epoch'], start_epoch, best_epoch, no_improve, args.early_stop_patience))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}', starting fresh".format(resume))

    with open(log_file, 'a') as log:
        log.write(str(args) + "\n")

    if args.ema_mode == "epoch":
        it = start_epoch * len(dataset_list)
    else:
        it = start_epoch * iters_per_epoch

    status = "running"
    epoch_times = [h["epoch_time_sec"] for h in history]

    if stopped_early:
        print("Early stopping had already triggered in the checkpoint; going straight to final evaluation.")
        epochs_to_run = []
    else:
        epochs_to_run = range(start_epoch, args.pretrain_epochs)

    for epoch in epochs_to_run:
        ep_start = time.time()
        lr_used = optimizer.param_groups[0]['lr']
        m_used = float(momentum_schedule[it])
        train_stats = {}
        for i, data_loader in enumerate(data_loader_list_train):
            criterion = torch.nn.CrossEntropyLoss() if datasets_config[dataset_list[i]]['task_type'] == "multi-class classification" else torch.nn.BCEWithLogitsLoss()
            st = train_one_epoch(model, i, dataset_list[i], data_loader, device, criterion, optimizer, epoch,
                                 args.ema_mode, teacher, momentum_schedule, it,
                                 accum_steps=args.accum_steps, scaler=scaler, max_steps=args.max_train_steps,
                                 print_freq=args.print_freq, save_debug_images=args.save_debug_images,
                                 debug_image_dir=os.path.join(args.output_root, "Models"))
            train_stats[dataset_list[i]] = st
            if args.ema_mode == "epoch":
                it += 1
            else:
                it += math.ceil(st["steps"] / args.accum_steps)

        val_loss_list = []
        for i, dv in enumerate(data_loader_list_val):
            criterion = torch.nn.CrossEntropyLoss() if datasets_config[dataset_list[i]]['task_type'] == "multi-class classification" else torch.nn.BCEWithLogitsLoss()
            val_loss = evaluate(model, i, dv, device, criterion, dataset_list[i], use_amp=use_amp, max_steps=args.max_eval_steps)
            val_loss_list.append(float(val_loss))

        avg_val_loss = float(np.average(val_loss_list))
        if args.val_loss_metric == "average":
            val_loss_metric = avg_val_loss
        else:
            val_loss_metric = val_loss_list[dataset_list.index(args.val_loss_metric)]

        # LR schedule. "original" reproduces the released code exactly: timm's
        # scheduler.step(epoch, metric) is called with the validation loss as the epoch.
        if args.sched_step_mode == "original":
            lr_scheduler.step(val_loss_metric)
        else:
            lr_scheduler.step(epoch + 1, val_loss_metric)

        improved = val_loss_metric < best_metric - args.early_stop_min_delta
        if improved:
            best_metric = val_loss_metric
            best_epoch = epoch
            no_improve = 0
            _atomic_save(_state(epoch, val_loss_list, full=False), best_model_path)
        else:
            no_improve += 1

        if args.early_stop_patience > 0 and epoch + 1 >= args.early_stop_min_epochs and no_improve >= args.early_stop_patience:
            stopped_early = True

        print("Epoch {:04d}: avg_val_loss {:.5f} | metric {:.5f} | best {:.5f} @ {} | no-improve {}/{} | saving model to {}".format(
            epoch, avg_val_loss, val_loss_metric, best_metric, best_epoch, no_improve, args.early_stop_patience, save_model_path))
        _atomic_save(_state(epoch, val_loss_list), latest_model_path)

        with open(log_file, 'a') as log:
            log.write("Epoch {:04d}: avg_val_loss = {:.5f} \n".format(epoch, avg_val_loss))
            log.write("     Datasets  : " + str(dataset_list) + "\n")
            log.write("     Val Losses: " + str(val_loss_list) + "\n")

        # periodic test, as in the original (epoch % test_epoch == 0 or last epoch)
        if args.test_epoch > 0 and (epoch % args.test_epoch == 0 or epoch + 1 == args.pretrain_epochs):
            if args.keep_periodic_ckpt:
                _atomic_save(_state(epoch, val_loss_list), save_model_path + str(epoch) + ".pth.tar")
            s_res = _test_all(model, dataset_list, datasets_config, data_loader_list_test, device, use_amp, args.max_eval_steps)
            t_res = _test_all(teacher, dataset_list, datasets_config, data_loader_list_test, device, use_amp, args.max_eval_steps)
            periodic_test.append({"epoch": epoch, "student": s_res, "teacher": t_res})
            _atomic_json(periodic_test, periodic_test_file)
            with open(output_file, 'a') as writer:
                writer.write("Omni-pretraining stage:\nEpoch {:04d}:\n".format(epoch))
                for i, dataset in enumerate(dataset_list):
                    writer.write("{} Validation Loss = {:.5f}:\n".format(dataset, val_loss_list[i]))
                    writer.write("{}: Student mAUC = {:.4f}, Teacher mAUC = {:.4f}\n".format(dataset, s_res[dataset]["auc"], t_res[dataset]["auc"]))
                    print(">>{}: Student mAUC = {:.4f}, Teacher mAUC = {:.4f}".format(dataset, s_res[dataset]["auc"], t_res[dataset]["auc"]))

        ep_time = time.time() - ep_start
        epoch_times.append(ep_time)
        history.append({
            "epoch": epoch,
            "val_loss": dict(zip(dataset_list, val_loss_list)),
            "avg_val_loss": avg_val_loss,
            "val_metric": float(val_loss_metric),
            "lr": float(lr_used),
            "teacher_momentum": m_used,
            "coff": train_stats[dataset_list[0]]["coff"],
            "train_loss_cls": {k: float(v["loss_cls"]) for k, v in train_stats.items()},
            "train_loss_mse": {k: float(v["loss_mse"]) for k, v in train_stats.items()},
            "improved": bool(improved),
            "no_improve": no_improve,
            "epoch_time_sec": ep_time,
        })
        _atomic_json(history, history_file)
        print("Epoch {:04d} took {:.1f} min".format(epoch, ep_time / 60))

        if stopped_early:
            print("Early stopping: no improvement for {} epochs (best epoch {}).".format(no_improve, best_epoch))
            _atomic_save(_state(epoch, val_loss_list), latest_model_path)
            break

        # PATCH: clean stop before a hard session limit (e.g. Kaggle 12 h)
        if args.time_limit_hours > 0:
            elapsed = time.time() - process_start
            need = np.mean(epoch_times[-3:]) * 1.15 + args.final_eval_reserve_min * 60
            if elapsed + need > args.time_limit_hours * 3600 and epoch + 1 < args.pretrain_epochs:
                status = "paused"
                print("Time limit: {:.1f} h used, next epoch would not fit. Stopping cleanly. "
                      "Run again with --resume to continue from epoch {}.".format(elapsed / 3600, epoch + 1))
                break

    if status == "paused":
        _atomic_json({"status": "paused", "next_epoch": history[-1]["epoch"] + 1 if history else start_epoch,
                      "best_epoch": best_epoch, "best_metric": best_metric}, status_file)
        return {"status": "paused"}

    # ------------------------------------------------------------------ final evaluation
    ckpt_path = best_model_path if os.path.isfile(best_model_path) else latest_model_path
    print("Final evaluation on the test sets using: {}".format(ckpt_path))
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _unwrap(model).load_state_dict(_strip_module(ck['state_dict']))
    _unwrap(teacher).load_state_dict(_strip_module(ck['teacher']))
    eval_epoch = ck['epoch']
    del ck
    s_res = _test_all(model, dataset_list, datasets_config, data_loader_list_test, device, use_amp, args.max_eval_steps)
    t_res = _test_all(teacher, dataset_list, datasets_config, data_loader_list_test, device, use_amp, args.max_eval_steps)
    final = {
        "datasets": list(dataset_list),
        "checkpoint_epoch": int(eval_epoch),
        "best_epoch": int(best_epoch),
        "best_val_metric": float(best_metric),
        "epochs_run": len(history),
        "stopped_early": bool(stopped_early),
        "total_train_time_h": float(sum(h["epoch_time_sec"] for h in history) / 3600),
        "config": {
            "model": args.model_name, "init": args.init, "input_size": args.crop_size, "resize": args.resize,
            "effective_batch": args.batch_size * args.accum_steps, "micro_batch": args.batch_size,
            "accum_steps": args.accum_steps, "lr": args.lr, "opt": args.opt, "warmup_epochs": args.warmup_epochs,
            "pretrain_epochs": args.pretrain_epochs, "momentum_teacher": args.momentum_teacher,
            "projector_features": args.projector_features, "amp": use_amp, "sched_step_mode": args.sched_step_mode,
            "early_stop_patience": args.early_stop_patience, "val_loss_metric": args.val_loss_metric,
            "bn_buffers": n_bn, "smoke_test": bool(args.max_train_steps or args.limit_train),
        },
        "teacher": t_res,
        "student": s_res,
    }
    _atomic_json(final, final_file)
    _atomic_json({"status": "done", "best_epoch": best_epoch}, status_file)
    with open(output_file, 'a') as writer:
        writer.write("Final (checkpoint epoch {}):\n".format(eval_epoch))
        for d in dataset_list:
            writer.write("{}: Student mAUC = {:.4f}, Teacher mAUC = {:.4f}\n".format(d, s_res[d]["auc"], t_res[d]["auc"]))
    print("\nFINAL (teacher, epoch {}):".format(eval_epoch))
    for d in dataset_list:
        print("  {:<12s} AUC {:.4f}  ACC {:.4f}".format(d, t_res[d]["auc"], t_res[d]["acc"]))
    print("Saved: {}".format(final_file))
    return {"status": "done", "final_file": final_file}
