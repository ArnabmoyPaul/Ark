"""
Download MedMNIST+ (224x224) and write memory-mappable .npy files for Ark+.

    python prepare_medmnist.py --data_root ./data --datasets chestmnist dermamnist retinamnist breastmnist

Output, per dataset:
    <data_root>/<name>/{train,val,test}_images.npy   uint8, (N,224,224) or (N,224,224,3)
    <data_root>/<name>/{train,val,test}_labels.npy   (N,1) or (N,14)

If automatic download fails (no internet, Zenodo blocked), put <name>_224.npz files
in --npz_dir and run again; they are used directly. On Kaggle you can also attach
a dataset that already contains the .npz files and point --npz_dir at it.
"""
import os
import argparse
import hashlib
import numpy as np

NAMES = ["chestmnist", "dermamnist", "retinamnist", "breastmnist"]
NAMES_3D = ["organmnist3d", "fracturemnist3d", "synapsemnist3d",
            "nodulemnist3d", "adrenalmnist3d", "vesselmnist3d"]


def default_size(name):
    """2D MedMNIST+ ships at 224; the 3D sets only go up to 64."""
    return 64 if name.endswith("3d") else 224


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def get_npz(name, npz_dir, size):
    fname = "{}_{}.npz".format(name, size)
    for d in [npz_dir, os.path.join(npz_dir, name)]:
        p = os.path.join(d, fname)
        if os.path.isfile(p):
            return p
    import medmnist
    from medmnist import INFO
    cls = getattr(medmnist, INFO[name]["python_class"])
    os.makedirs(npz_dir, exist_ok=True)
    cls(split="train", download=True, size=size, root=npz_dir)
    p = os.path.join(npz_dir, fname)
    assert os.path.isfile(p), "download did not produce " + p
    return p


def convert(name, data_root, npz_dir, size=224, check_md5=True):
    out = os.path.join(data_root, name)
    done = all(os.path.isfile(os.path.join(out, "{}_{}.npy".format(s, k)))
               for s in ["train", "val", "test"] for k in ["images", "labels"])
    if done:
        print("[skip] {} already prepared in {}".format(name, out))
        return
    p = get_npz(name, npz_dir, size)
    if check_md5:
        try:
            from medmnist import INFO
            want = INFO[name].get("MD5_{}".format(size))
            if want:
                got = md5(p)
                assert got == want, "MD5 mismatch for {}: {} != {}".format(p, got, want)
                print("[ok] MD5 {}".format(name))
        except ImportError:
            pass
    os.makedirs(out, exist_ok=True)
    z = np.load(p)
    for s in ["train", "val", "test"]:
        imgs = z["{}_images".format(s)]
        labs = z["{}_labels".format(s)]
        np.save(os.path.join(out, "{}_images.npy".format(s)), imgs)
        np.save(os.path.join(out, "{}_labels.npy".format(s)), labs)
        print("  {:<12s} {:<5s} images {} labels {}".format(name, s, imgs.shape, labs.shape))
        del imgs, labs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--npz_dir", default="./data/npz")
    ap.add_argument("--datasets", nargs="+", default=NAMES)
    ap.add_argument("--size", type=int, default=0, help="0 = 224 for 2D sets, 64 for 3D sets")
    ap.add_argument("--no_md5", action="store_true")
    a = ap.parse_args()
    for n in a.datasets:
        n = n.lower()
        convert(n, a.data_root, a.npz_dir, a.size or default_size(n), check_md5=not a.no_md5)
    print("done")
