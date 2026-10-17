# U-NetMo — concrete damage segmentation

Multi-class RC damage segmentation (U-Net / MobileNetV3-U-Net). The dataset is **not** in this repo; it is downloaded from Roboflow.

## Layout
```
train.py / evaluate.py     entry points (short, readable)
configs/classes.py         class names, 6class / 7class variants, default weights  <- edit classes here
datasets/                  dataset class, augmentations, Roboflow download script
models/                    unet, unet_mobilenet, builder.py (add new models here)
training/                  losses, metrics, train/eval loops, checkpoints, tracker, class weights
runs/                      outputs (git-ignored)
```

## Colab quick start
```python
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/absit23/U-NetMo && cd U-NetMo && pip install -q -r requirements.txt
import os; os.environ["ROBOFLOW_API_KEY"] = "<key>"
!cd U-NetMo && python -m datasets.roboflow_download --workspace WS --project PROJ --version 1 --out /content/dataset
!cd U-NetMo && python train.py --data_dir /content/dataset --variant 7class --class_weights auto --runs_dir "/content/drive/MyDrive/U-Net/runs"
!cd U-NetMo && python evaluate.py --data_dir /content/dataset --checkpoint_path "/content/drive/MyDrive/U-Net/runs/round1/best_unet_mobilenet.pth"
```
Each run folder (`runs/roundN/`) holds `best_*.pth`, `last_*.pth`, `metrics.csv`, loss / metrics / per-class-IoU curves, `confusion_matrix.{csv,png}` and `args.json`. Re-running the same command resumes from `last_*.pth`.

## Notes
- Mask IDs must match the order in `configs/classes.py` (Roboflow sorts class names alphabetically).
- `--variant 6class` drops Efflorescence; the dataset export must match the variant.
- `--class_weights`: omit = manual defaults, `auto` = from training-mask pixel counts, or `w0,w1,...`.
