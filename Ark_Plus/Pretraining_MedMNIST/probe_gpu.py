"""
Find the largest micro-batch that fits on this GPU for Ark+ Swin-Base 224
(student forward+backward + teacher forward + SGD step, same as training),
measure throughput, and project how long the 4-dataset run will take.

    python probe_gpu.py --effective_batch 200 --amp True

Prints one line you can paste into the training command:  --batch_size B --accum_steps K
"""
import time
import json
import types
import argparse
import torch
from models import build_omni_model

# Official MedMNIST split sizes
TRAIN = {"ChestMNIST": 78468, "DermaMNIST": 7007, "RetinaMNIST": 1080, "BreastMNIST": 546}
VAL = {"ChestMNIST": 11219, "DermaMNIST": 1003, "RetinaMNIST": 120, "BreastMNIST": 78}
TEST = {"ChestMNIST": 22433, "DermaMNIST": 2005, "RetinaMNIST": 400, "BreastMNIST": 156}


def divisors_desc(n, cap):
    return [d for d in range(min(n, cap), 0, -1) if n % d == 0]


def step_fn(model, teacher, opt, x, amp, scaler):
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        with torch.no_grad():
            ft, _ = teacher(x, 0)
        fs, ps = model(x, 0)
        loss = ps.float().mean() + torch.nn.functional.mse_loss(fs.float(), ft.float())
    if amp:
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    else:
        loss.backward(); opt.step()
    opt.zero_grad(set_to_none=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--effective_batch", type=int, default=200)
    ap.add_argument("--amp", default="True")
    ap.add_argument("--max_micro", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--test_epoch", type=int, default=10)
    ap.add_argument("--safety", type=float, default=0.85, help="use at most this share of the largest batch that fits")
    a = ap.parse_args()
    amp = a.amp.lower() in ("1", "true", "yes", "t")
    assert torch.cuda.is_available(), "no CUDA GPU visible"
    n_gpu = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
    mem = [torch.cuda.get_device_properties(i).total_memory / 2**30 for i in range(n_gpu)]
    print("GPUs:", ", ".join("{} ({:.1f} GB)".format(n, m) for n, m in zip(names, mem)))

    args = types.SimpleNamespace(model_name="swin_base", projector_features=1376, use_mlp=False, pretrained_weights=None)
    model = build_omni_model(args, [14, 7, 5, 1]).cuda()
    teacher = build_omni_model(args, [14, 7, 5, 1]).cuda()
    for p in teacher.parameters():
        p.requires_grad = False
    if n_gpu > 1:
        model, teacher = torch.nn.DataParallel(model), torch.nn.DataParallel(teacher)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda") if amp else None

    fits = 0
    for b in divisors_desc(a.effective_batch, a.max_micro):
        if n_gpu > 1 and b % n_gpu:
            continue
        try:
            torch.cuda.empty_cache()
            x = torch.randn(b, 3, 224, 224, device="cuda")
            step_fn(model, teacher, opt, x, amp, scaler)
            torch.cuda.synchronize()
            fits = b
            print("  micro-batch {:>3d}: fits".format(b))
            break
        except torch.cuda.OutOfMemoryError:
            print("  micro-batch {:>3d}: out of memory".format(b))
            opt.zero_grad(set_to_none=True)
            del x
            torch.cuda.empty_cache()
    assert fits, "even micro-batch 1 does not fit"

    # pick a safe divisor of the effective batch
    cands = [d for d in divisors_desc(a.effective_batch, fits) if d <= max(1, int(fits * a.safety)) and (n_gpu == 1 or d % n_gpu == 0)]
    b = cands[0] if cands else fits
    k = a.effective_batch // b

    x = torch.randn(b, 3, 224, 224, device="cuda")
    for _ in range(3):
        step_fn(model, teacher, opt, x, amp, scaler)
    torch.cuda.synchronize()
    t0 = time.time(); n = 8
    for _ in range(n):
        step_fn(model, teacher, opt, x, amp, scaler)
    torch.cuda.synchronize()
    train_ips = n * b / (time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**30

    model.eval()
    xe = torch.randn(b * 2, 3, 224, 224, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        model(xe, 0); torch.cuda.synchronize(); t0 = time.time()
        for _ in range(5):
            model(xe, 0)
        torch.cuda.synchronize()
    eval_ips = 5 * xe.shape[0] / (time.time() - t0)

    # Data loading/augmentation is not included, so real time is usually 10-30% higher.
    train_sec = sum(TRAIN.values()) / train_ips
    val_sec = sum(VAL.values()) / eval_ips
    test_sec = 2 * 10 * sum(TEST.values()) / eval_ips  # student+teacher, TenCrop
    n_tests = len(range(0, a.epochs, a.test_epoch)) + 1 if a.test_epoch > 0 else 1
    epoch_h = (train_sec + val_sec) / 3600
    total_h = a.epochs * epoch_h + n_tests * test_sec / 3600
    out = {
        "gpus": names, "amp": amp, "largest_fit": fits, "micro_batch": b, "accum_steps": k,
        "effective_batch": b * k, "peak_mem_gb": round(peak, 2),
        "train_img_per_s": round(train_ips, 1), "eval_img_per_s": round(eval_ips, 1),
        "epoch_hours": round(epoch_h, 2), "one_full_test_hours": round(test_sec / 3600, 2),
        "projected_hours_{}_epochs".format(a.epochs): round(total_h, 1),
    }
    print(json.dumps(out, indent=2))
    print("\nUse:  --batch_size {} --accum_steps {}   (effective batch {})".format(b, k, b * k))
    json.dump(out, open("probe_result.json", "w"), indent=2)


if __name__ == "__main__":
    main()
