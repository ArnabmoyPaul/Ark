# Ark+ on four MedMNIST datasets

Official Ark+ pretraining code (`../Pretraining`), adapted to train one Ark+ model
jointly on ChestMNIST, DermaMNIST, RetinaMNIST and BreastMNIST (MedMNIST+ 224x224).
Every change is listed in [PATCHES.md](PATCHES.md).

## 1. Prepare data
```
pip install -r requirements_medmnist.txt
python prepare_medmnist.py --data_root ./data --npz_dir ./data/npz
```

## 2. Pick batch settings for your GPU
```
python probe_gpu.py --effective_batch 200 --amp True
```
Typical result: RTX 4060 8 GB about `--batch_size 10 --accum_steps 20`,
Kaggle 2x T4 about `--batch_size 40 --accum_steps 5`.

## 3. Train (safe to stop and restart at any time)
```
python main_ark.py --data_set ChestMNIST --data_set DermaMNIST --data_set RetinaMNIST --data_set BreastMNIST \
  --opt sgd --warmup-epochs 20 --lr 0.3 --model swin_base --init imagenet \
  --pretrained_weights https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth \
  --momentum_teacher 0.9 --projector_features 1376 \
  --pretrain_epochs 50 --test_epoch 10 \
  --batch_size 10 --accum_steps 20 --amp True \
  --early_stop_patience 5 --resume True \
  --data_root ./data --output_root ./runs --exp_name medmnist4
```
Run the same command again after any interruption: it continues from the last
finished epoch. Add `--time_limit_hours 11.3` on Kaggle.

Smoke test (a few minutes): add `--limit_train 64 --limit_eval 32 --pretrain_epochs 2 --test_epoch 1`.

## 4. Report
```
python report.py --run_dir runs/Models/swin_base_medmnist4/Ark_Plus_ChestMNIST_DermaMNIST_RetinaMNIST_BreastMNIST/medmnist4 --out_dir report
```

## 3D datasets

```
python prepare_medmnist.py --data_root ./data --npz_dir ./data/npz \
  --datasets organmnist3d fracturemnist3d synapsemnist3d     # 64x64x64 is used automatically

python main_ark.py --data_set OrganMNIST3D --data_set FractureMNIST3D --data_set SynapseMNIST3D \
  --opt sgd --warmup-epochs 20 --lr 0.3 --model swin_base --init imagenet \
  --pretrained_weights https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth \
  --momentum_teacher 0.9 --projector_features 1376 \
  --pretrain_epochs 50 --test_epoch 10 --batch_size 10 --accum_steps 20 --amp True \
  --slice_mode center3 --resume True \
  --data_root ./data --output_root ./runs --exp_name medmnist3d
```
The three 3D sets hold 3,228 training volumes in total, about 4% of the 2D run, so an epoch takes roughly a minute
on two T4s. See PATCHES.md for what `--slice_mode` does and why it must be disclosed.

## Tests
`python tests/test_accumulation.py` checks that gradient accumulation matches a full
batch and that the teacher EMA covers BatchNorm buffers.
