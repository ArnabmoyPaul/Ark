# What changed from `Ark_Plus/Pretraining`

This folder is a copy of the official Ark+ pretraining code with the smallest set of
changes needed to run it on four MedMNIST+ datasets on a small GPU. The original
folder is untouched, so every change can be checked with:

```
git diff --no-index Ark_Plus/Pretraining Ark_Plus/Pretraining_MedMNIST
```

Every changed line in the code is marked `PATCH`.

## Kept exactly as released

| Item | Value |
|---|---|
| Backbone | Swin-Base, patch 4, window 7, 224x224 (`--model swin_base`) |
| Initialisation | ImageNet-22k to 1k Swin-Base weights, same URL as the README |
| Student and teacher | both built from the same pretrained weights; teacher = EMA of student |
| Projector / heads | linear projector to 1376 features, one linear head per dataset |
| Cyclic training | each epoch = one pass over every dataset in turn, teacher EMA after each dataset (`ema_mode=epoch`) |
| Loss | `(1-coff)*task_loss + coff*MSE(student_feat, teacher_feat)`, `coff=(m-0.9)*5` |
| Task losses | BCE for multi-label and binary, CrossEntropy for multi-class |
| Teacher momentum | cosine from 0.9 to 1 over the run |
| Optimiser | SGD, lr 0.3, momentum 0.9, weight decay 0, timm cosine scheduler, 20 warm-up epochs |
| LR stepping | `lr_scheduler.step(val_loss)` exactly as released (see note 1) |
| Augmentation | student: RandomResizedCrop + ShiftScaleRotate + brightness/contrast/gamma; teacher: resized original image |
| Validation / test | Resize 256, CenterCrop 224 / TenCrop 224, ImageNet normalisation |
| Metric | `metric_AUROC`, mean AUC over classes |
| Effective batch | 200, as in the README Swin-Base command |

## Changes

| # | Change | Why | Effect on results |
|---|---|---|---|
| 1 | New `MedMNIST` dataset class and `datasets_config.yaml` entries | the original only ships chest X-ray loaders | none; image pipeline is copied line by line from `ChestXray14` |
| 2 | Gradient accumulation (`--accum_steps`) with size-weighted micro-batch losses | batch 200 does not fit in 8 GB or 16 GB | none: Swin has no BatchNorm, so K micro-batches equal one batch of K x B. Verified by `tests/test_accumulation.py` (difference below 1e-7) |
| 3 | Mixed precision (`--amp True`) | memory and speed on RTX 4060 / T4 | tiny floating point differences; can be turned off |
| 4 | BN-EMA: teacher EMA also averages BatchNorm buffers | requested by Prof. Liang | none for Swin-Base (0 BatchNorm buffers, printed at start). Active if a BatchNorm backbone is used |
| 5 | Teacher forward inside `torch.no_grad()` | memory | none (teacher parameters already had `requires_grad=False`) |
| 6 | Early stopping (`--early_stop_patience`), best checkpoint, final test of the best checkpoint | 50-epoch budget | the original has no early stopping and reports the checkpoints at every `test_epoch` |
| 7 | Resume fixed and made portable | original crashed on `args.reinit_heads`; DataParallel prefixes broke 1 GPU vs 2 GPU checkpoints | none |
| 8 | Atomic checkpoint writes, `--time_limit_hours` clean stop | Kaggle sessions end at 12 h | none |
| 9 | `targets.cuda()` changed to `targets.to(device)` | allows CPU test | none |
| 10 | albumentations imports trimmed, `RandomResizedCrop` signature shim, `ShiftScaleRotate(border_mode=REFLECT_101)` | old names were removed from albumentations; 2.x changed the default border to black | none: keeps the 0.x/1.x behaviour |
| 11 | `wandb.log` only when a wandb run exists | original crashes on recent wandb | none |
| 12 | Separate `--eval_batch_size` | original test batch = batch/2 = 100 images x 10 crops | none (no BatchNorm) |
| 13 | Safety check that the Swin backbone weights really loaded | `strict=False` would hide a wrong checkpoint | none |
| 14 | Debug image dumps off by default (`--save_debug_images True` to enable) | disk and speed | none |
| 15 | `history.json`, `periodic_test.json`, `final_results.json` | plotting | none |

## Notes worth raising

1. **LR schedule in the released code.** `engine.py` calls `lr_scheduler.step(val_loss_metric)`.
   timm's `Scheduler.step(epoch, metric)` takes the epoch as its first argument, so the
   validation loss is used as the epoch. During warm-up timm computes
   `lr = warmup_lr + epoch * (lr - warmup_lr) / warmup_epochs`, so the learning rate
   becomes about `0.015 * val_loss` (for example 0.009 at val loss 0.6) and never
   reaches 0.3. Checked with timm 0.5.4. This folder reproduces that behaviour by
   default (`--sched_step_mode original`); `--sched_step_mode epoch` gives the
   schedule described in the paper.
2. **Epochs.** The README Swin-Base command uses 200 epochs; the Nature paper
   (Methods, Pretraining set-up) reports 50 epochs. 50 is used here.
3. **RetinaMNIST** is ordinal in MedMNIST; it is trained and scored as 5-class
   classification, like the MedMNIST benchmark.
4. **BreastMNIST** uses one output with BCE (like Shenzhen in the original). MedMNIST
   label 1 is "normal/benign", kept as is so AUC matches the MedMNIST evaluator.
