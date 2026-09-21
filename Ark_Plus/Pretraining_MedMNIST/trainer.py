"""
trainer.py for Ark+ on MedMNIST.

Copied from Ark_Plus/Pretraining/trainer.py. The training math is unchanged:
    loss = (1 - coff) * loss_cls + coff * MSE(feat_student, feat_teacher)
    coff = (momentum_schedule[it] - 0.9) * 5
Every change is marked with "PATCH" and listed in PATCHES.md.
"""
from utils import MetricLogger, ProgressLogger, save_image, save_snapshot
import os
import time
import torch
from tqdm import tqdm


# PATCH: the original calls wandb.log() without ever calling wandb.init().
# Recent wandb versions raise an error for that, so only log when a run exists.
def _wandb_log(payload):
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(payload)
    except Exception:
        pass


def _unwrap(m):
    return m.module if isinstance(m, torch.nn.DataParallel) else m


def train_one_epoch(model, use_head_n, dataset, data_loader_train, device, criterion, optimizer, epoch,
                    ema_mode, teacher, momentum_schedule, it,
                    accum_steps=1, scaler=None, max_steps=0, print_freq=50, save_debug_images=False,
                    debug_image_dir="Models"):
    """One pass over one dataset, exactly as in the original.

    PATCH (gradient accumulation): a physical batch of B images with accum_steps=K
    behaves like one optimizer step on B*K images. Each micro-batch loss is weighted
    by (micro-batch size / group size), so the gradient equals the mean over the whole
    group, including the smaller last group. Swin uses LayerNorm (no BatchNorm), so the
    result matches a real batch of B*K up to floating point rounding.

    PATCH (AMP): if scaler is given, forward passes run in float16 autocast.
    """
    batch_time = MetricLogger('Time', ':6.3f')
    losses_cls = MetricLogger('Loss_' + dataset + ' cls', ':.4e')
    losses_mse = MetricLogger('Loss_' + dataset + ' mse', ':.4e')
    progress = ProgressLogger(
        len(data_loader_train),
        [batch_time, losses_cls, losses_mse],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    MSE = torch.nn.MSELoss()
    coff = (momentum_schedule[it] - 0.9) * 5

    # Size of every micro-batch, known in advance (shuffle=True, drop_last=False).
    n = len(data_loader_train.dataset)
    bs = data_loader_train.batch_size
    sizes = [bs] * (n // bs) + ([n % bs] if n % bs else [])
    if max_steps and max_steps > 0:
        sizes = sizes[:max_steps]
    group_total = {}
    for i, s in enumerate(sizes):
        group_total[i // accum_steps] = group_total.get(i // accum_steps, 0) + s
    n_steps = len(sizes)

    use_amp = scaler is not None
    amp_device = "cuda" if str(device).startswith("cuda") else "cpu"

    optimizer.zero_grad(set_to_none=True)
    end = time.time()
    for i, (samples1, samples2, targets) in enumerate(data_loader_train):
        if i >= n_steps:
            break
        samples1 = samples1.float().to(device, non_blocking=True)
        samples2 = samples2.float().to(device, non_blocking=True)
        targets = targets.float().to(device, non_blocking=True)

        with torch.autocast(device_type=amp_device, dtype=torch.float16, enabled=use_amp):
            # PATCH: teacher forward under no_grad. Teacher params already have
            # requires_grad=False in the original, so gradients are identical;
            # this only saves memory.
            with torch.no_grad():
                feat_t, pred_t = teacher(samples2, use_head_n)
            feat_s, pred_s = model(samples1, use_head_n)
            loss_cls = criterion(pred_s.float(), targets)
            loss_const = MSE(feat_s.float(), feat_t.float())
            loss = (1 - coff) * loss_cls + coff * loss_const

        weight = samples1.size(0) / group_total[i // accum_steps]
        scaled = loss * weight
        if use_amp:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        last_in_group = ((i + 1) % accum_steps == 0) or (i + 1 == n_steps)
        if last_in_group:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if ema_mode == "iteration":
                ema_update_teacher(model, teacher, momentum_schedule, it)
                it += 1

        losses_cls.update(loss_cls.item(), samples1.size(0))
        losses_mse.update(loss_const.item(), samples1.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)
            # PATCH: debug image dumps are off by default (they do not affect training).
            if save_debug_images:
                os.makedirs(debug_image_dir, exist_ok=True)
                save_image(samples1[0].float().cpu().numpy().transpose(1, 2, 0), os.path.join(debug_image_dir, "student" + str(i)))
                save_image(samples2[0].float().cpu().numpy().transpose(1, 2, 0), os.path.join(debug_image_dir, "teacher" + str(i)))

    if ema_mode == "epoch":
        ema_update_teacher(model, teacher, momentum_schedule, it)
        it += 1

    _wandb_log({"train_loss_cls_{}".format(dataset): losses_cls.avg})
    _wandb_log({"train_loss_mse_{}".format(dataset): losses_mse.avg})
    return {"loss_cls": losses_cls.avg, "loss_mse": losses_mse.avg, "coff": float(coff), "steps": n_steps}


def ema_update_teacher(model, teacher, momentum_schedule, it):
    """Teacher = EMA of student.

    PATCH (BN-EMA, requested by Prof. Liang): the original loops over parameters()
    only, so BatchNorm running_mean / running_var (which are buffers) were never
    averaged into the teacher. Floating point buffers are now averaged with the same
    momentum; integer buffers (num_batches_tracked) are copied.
    Note: Swin-Base and ConvNeXt-Base use LayerNorm and have no BatchNorm buffers,
    so for those backbones this extra loop changes nothing.
    """
    with torch.no_grad():
        m = momentum_schedule[it]  # momentum parameter
        for param_q, param_k in zip(model.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        for buf_q, buf_k in zip(model.buffers(), teacher.buffers()):
            if buf_k.dtype.is_floating_point:
                buf_k.data.mul_(m).add_((1 - m) * buf_q.detach().data)
            else:
                buf_k.data.copy_(buf_q.data)


def count_bn_buffers(model):
    """How many BatchNorm running-stat buffers the model has (0 for Swin/ConvNeXt)."""
    n = 0
    for mod in _unwrap(model).modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm) and mod.running_mean is not None:
            n += 2
    return n


def evaluate(model, use_head_n, data_loader_val, device, criterion, dataset, use_amp=False, max_steps=0):
    model.eval()
    amp_device = "cuda" if str(device).startswith("cuda") else "cpu"

    with torch.no_grad():
        batch_time = MetricLogger('Time', ':6.3f')
        losses = MetricLogger('Loss', ':.4e')
        progress = ProgressLogger(
            len(data_loader_val),
            [batch_time, losses], prefix='Val_' + dataset + ': ')

        end = time.time()
        for i, (samples, _, targets) in enumerate(data_loader_val):
            if max_steps and i >= max_steps:
                break
            samples, targets = samples.float().to(device), targets.float().to(device)

            with torch.autocast(device_type=amp_device, dtype=torch.float16, enabled=use_amp):
                _, outputs = model(samples, use_head_n)
            loss = criterion(outputs.float(), targets)

            losses.update(loss.item(), samples.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                progress.display(i)

    return losses.avg


def test_classification(model, use_head_n, data_loader_test, device, multiclass=False, use_amp=False, max_steps=0):
    model.eval()
    amp_device = "cuda" if str(device).startswith("cuda") else "cpu"

    y_test = torch.FloatTensor().to(device)
    p_test = torch.FloatTensor().to(device)

    with torch.no_grad():
        for i, (samples, _, targets) in enumerate(tqdm(data_loader_test)):
            if max_steps and i >= max_steps:
                break
            targets = targets.to(device)  # PATCH: was targets.cuda(); .to(device) also works on CPU
            y_test = torch.cat((y_test, targets), 0)

            if len(samples.size()) == 4:
                bs, c, h, w = samples.size()
                n_crops = 1
            elif len(samples.size()) == 5:
                bs, n_crops, c, h, w = samples.size()

            varInput = samples.view(-1, c, h, w).float().to(device)

            with torch.autocast(device_type=amp_device, dtype=torch.float16, enabled=use_amp):
                _, out = model(varInput, use_head_n)
            out = out.float()
            if multiclass:
                out = torch.softmax(out, dim=1)
            else:
                out = torch.sigmoid(out)
            outMean = out.view(bs, n_crops, -1).mean(1)
            p_test = torch.cat((p_test, outMean.data), 0)

    return y_test, p_test
