# ResNet50 FairFace Gender

Du an phan lop nhi phan gioi tinh `Male/Female` tu anh khuon mat trong FairFace padding 0.25 bang PyTorch va torchvision.

Mo hinh chinh la ResNet50 pretrained ImageNet, thay lop phan loai cuoi bang `Linear(2048, 2)` va huan luyen hai giai doan:

1. Feature extraction: chi huan luyen head.
2. Fine-tuning: mo khoa mot phan hoac toan bo backbone voi learning rate nho hon.

## Tinh nang

- Tu dong tach `train`, `val` va `test` theo `gender + race` voi `random_state=42`.
- FairFace CSV val duoc giu lam test chinh thuc, khong dung de chon checkpoint.
- Ho tro ResNet50, ResNet18, VGG16 va MobileNetV2.
- Augmentation, class weights, freeze BatchNorm, early stopping va resume checkpoint.
- Smoke test tren du lieu gia, khong can Internet, GPU hay FairFace that.
- Danh gia accuracy, precision, recall, F1 macro va confusion matrix.
- Phan tich theo race/age, latency, top errors va Grad-CAM.
- Demo Gradio chay CPU, co Haar face detector mac dinh.

## Yeu cau moi truong

- Python >= 3.10.
- Local: dung cho test, smoke test va phan tich nhe. Khong train toan bo dataset tren CPU yeu.
- Kaggle: dung GPU P100/T4 de train that.

## Cau truc thu muc

```text
Deep_Learning/
├── AGENT_GUIDE_ResNet50_FairFace_Code.md  # Dac ta ky thuat goc
├── README.md
├── requirements.txt                        # Dependency local
├── requirements-kaggle.txt                 # Dependency bo sung tren Kaggle
├── .gitignore
├── dataset/                                # Du lieu local, khong commit
│   ├── fairface_label_train.csv
│   ├── fairface_label_val.csv
│   └── fairface-img-margin025-trainval/
│       ├── train/
│       └── val/
├── src/
│   ├── config.py           # Duong dan, seed, batch size, class names
│   ├── utils.py            # Seed, logger, JSON, timer
│   ├── data.py             # Split, Dataset, transforms, DataLoader
│   ├── model.py            # Model factory va freeze/unfreeze
│   ├── experiments.py      # Cau hinh E1 den E9
│   ├── train.py            # CLI huan luyen va checkpoint
│   ├── evaluate.py         # CLI du doan va metrics
│   ├── analysis.py         # Bang so lieu va bieu do
│   └── gradcam.py          # Giai thich ResNet bang Grad-CAM
├── scripts/
│   └── check_duplicates.py # Tim anh trung giua train va test
├── app/
│   └── demo.py             # Gradio demo CPU
├── notebooks/              # Notebook mong cho Kaggle/phan tich
├── tests/                  # Test voi du lieu anh gia
└── artifacts/              # Checkpoint va ket qua local, khong commit
```

## Cai dat local

```bash
cd /home/huynguyen075/Projects/DeepLearning
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Kiem tra code bang du lieu gia:

```bash
python -m pytest -q
```

Lenh tren khong quet toan bo anh FairFace va khong train that. Neu shell dang tro toi executable `pytest` ngoai virtual environment, dung `python -m pytest` nhu tren.

## Cau hinh du lieu

Local, `src/config.py` mac dinh dung:

```text
DATA_DIR = ./dataset
OUT_DIR = ./artifacts
SPLIT_DIR = ./splits
```

Code ho tro ca hai dang:

```text
dataset/fairface_label_train.csv
dataset/train/1.jpg
```

va layout hien tai cua repo:

```text
dataset/fairface_label_train.csv
dataset/fairface-img-margin025-trainval/train/1.jpg
```

CSV bat buoc co cac cot `file`, `age`, `gender`, `race`. Gia tri `gender` phai la `Male` hoac `Female`. Mapping co dinh la `Male=0`, `Female=1`.

## Smoke test local

Smoke test dung du lieu gia do pytest tao tu dong. Co the chay CLI smoke voi du lieu FairFace that, nhung chi lay toi da 48 anh train, 16 anh val va 16 anh test:

```bash
python -m src.train --exp e8_resnet18 --smoke
python -m src.evaluate --exp e8_resnet18 --split val --smoke
```

## Train that tren Kaggle

Khuyen nghi train that tren Kaggle GPU. Attach dataset FairFace vao Notebook, sau do clone repository:

```python
!git clone https://github.com/NoBugNoPain500/Deep_Learning.git /kaggle/working/Deep_Learning
%cd /kaggle/working/Deep_Learning
!pip install -q -r requirements-kaggle.txt
```

Kiem tra ten thu muc dataset:

```python
!find /kaggle/input -maxdepth 3 -type f | head -30
```

Neu dataset nam tai `/kaggle/input/fairface`, code tu nhan dien duong dan:

```python
!python -m src.train --exp e1_main
!python -m src.evaluate --exp e1_main --split val
```

Chi sau khi chon checkpoint bang validation moi chay test:

```python
!python -m src.evaluate --exp e1_main --split test
```

Neu ten dataset khac `fairface`, truyen duong dan truc tiep:

```python
!python -m src.train \
	--exp e1_main \
	--data-dir /kaggle/input/ten-dataset \
	--split-dir /kaggle/working/splits
```

Neu dung pretrained ImageNet va Kaggle tat Internet, can cung cap checkpoint qua `--weights-path`; neu khong, bat Internet de torchvision tai weights trong `build_model`.

## Cac experiment

| Ten              | Cau hinh                                                |
| ---------------- | ------------------------------------------------------- |
| `e1_main`        | ResNet50 pretrained, augmentation, fine-tune `layer3_4` |
| `e2_scratch`     | ResNet50 train tu dau, khong pretrained                 |
| `e3_no_aug`      | ResNet50 khong augmentation                             |
| `e4_freeze`      | Chi train head, khong fine-tune backbone                |
| `e5_ft_l4`       | Fine-tune `layer4`                                      |
| `e6_ft_all`      | Fine-tune toan bo model                                 |
| `e7_vgg16`       | VGG16 pretrained                                        |
| `e8_resnet18`    | ResNet18 pretrained                                     |
| `e9_mobilenetv2` | MobileNetV2 pretrained                                  |

Chay mot experiment:

```bash
python -m src.train --exp e5_ft_l4
```

Chay lai tu checkpoint cu:

```bash
python -m src.train --exp e5_ft_l4 --resume
```

Override epoch de test nhanh:

```bash
python -m src.train --exp e5_ft_l4 --epochs-override 1
```

## Ket qua va checkpoint

Local, ket qua nam trong `artifacts/<experiment>/`. Tren Kaggle, ket qua nam trong `/kaggle/working/<experiment>/`.

```text
<experiment>/
├── best.pth          # Checkpoint dung cho demo/evaluate
├── best_stage1.pth   # Best checkpoint giai doan 1
├── best_stage2.pth   # Best checkpoint giai doan 2
├── last.pth          # Checkpoint phuc vu resume
├── history.json      # Loss/accuracy/F1 theo epoch
├── config.json       # Cau hinh experiment va so tham so
├── val_metrics.json  # Ket qua validation
├── metrics.json      # Ket qua test, duoc bao ve khong ghi de
└── test_preds.csv    # Du doan tung anh test
```

## Evaluate va phan tich

```bash
python -m src.evaluate --exp e1_main --split val
python -m src.evaluate --exp e1_main --split test
```

Test set bi khoa neu `metrics.json` da ton tai. Chi chay lai co chu y:

```bash
python -m src.evaluate --exp e1_main --split test --force
```

Phan tich ket qua da xuat co the thuc hien trong Python:

```python
from src.analysis import compare_experiments, group_accuracy, top_errors
import pandas as pd

preds = pd.read_csv("artifacts/e1_main/test_preds.csv")
print(group_accuracy(preds, by="race"))
print(top_errors(preds, k=24))
print(compare_experiments("artifacts"))
```

## Demo

Sau khi co checkpoint:

```bash
python app/demo.py --ckpt artifacts/e1_main/best.pth
```

Demo chay CPU, preprocess anh theo cung transform validation, thu phat hien khuon mat bang OpenCV Haar va dung anh goc neu khong phat hien duoc mat. Ket qua chi co y nghia hoc thuat.

## Kiem tra anh trung

Lenh nay chi nen chay day du tren Kaggle CPU. Local chi dung `--limit`:

```bash
python scripts/check_duplicates.py --limit 50 --out duplicates.csv
```

## Luu y ve reproducibility va ethics

- Seed mac dinh la `42`.
- Khi co CUDA, `cudnn.benchmark=True`; toc do nhanh hon nhung khong dam bao deterministic tuyet doi.
- Khong dung test set de tune hyperparameter, early stopping hay chon checkpoint.
- Khong commit dataset, checkpoint, `artifacts/`, `splits/` hay token Kaggle.
- FairFace la dataset nghien cuu ve khuon mat; khong dung model cho quyet dinh tu dong ve con nguoi, danh gia danh tinh hay cac ung dung nhay cam.
- Nguon du lieu: FairFace, license CC BY 4.0. Hay doc dieu khoan cua dataset truoc khi phan phoi ket qua.
