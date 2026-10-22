# Chay ResNet50 FairFace tren Google Colab

Tai lieu nay la quy trinh chay chinh cho train GPU. Source code van nam trong `src/`; Colab chi goi CLI, vi vay co the tiep tuc resume va tai artifacts tu Google Drive.

## 1. Tao runtime

Trong Colab chon `Runtime` -> `Change runtime type` -> mot GPU (T4, L4 hoac A100 tuy tai khoan). Kiem tra:

```python
!nvidia-smi
```

## 2. Mount Drive va clone repo

```python
from google.colab import drive
drive.mount("/content/drive")

%cd /content
!git clone <REPOSITORY_URL> DeepLearning
%cd /content/DeepLearning
```

Neu repo da co tren Drive, thay lenh `git clone` bang `%cd` toi thu muc repo do.

## 3. Cai dependency

Colab da co PyTorch va torchvision ban CUDA. Chi cai cac goi bo sung:

```python
!pip install -q -r requirements-colab.txt
!python -c "import torch, torchvision; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Khong dung `requirements-kaggle.txt` cho Colab va khong cai lai PyTorch neu khong co ly do dac biet, vi co the lam mat CUDA runtime.

## 4. Dat dataset

Layout khuyen nghi tren Drive:

```text
/content/drive/MyDrive/fairface/
├── fairface_label_train.csv
├── fairface_label_val.csv
└── fairface-img-margin025-trainval/
    ├── train/
    └── val/
```

CSV phai co `file`, `age`, `gender`, `race`; `gender` chi gom `Male` va `Female`. Kiem tra nhanh:

```python
!find /content/drive/MyDrive/fairface -maxdepth 3 -type f | head -30
```

Neu dataset o vi tri khac, thay `--data-dir` trong cac lenh ben duoi. Khong sua source code.

## 5. Smoke test

Smoke test nen chay truoc train that de kiem tra import, CSV va checkpoint:

```python
%cd /content/DeepLearning
!python -m src.train --exp e8_resnet18 --smoke \
  --data-dir /content/drive/MyDrive/fairface \
  --split-dir /content/drive/MyDrive/fairface-artifacts/splits \
  --out-dir /content/drive/MyDrive/fairface-artifacts
!python -m src.evaluate --exp e8_resnet18 --split val --smoke \
  --data-dir /content/drive/MyDrive/fairface \
  --split-dir /content/drive/MyDrive/fairface-artifacts/splits \
  --out-dir /content/drive/MyDrive/fairface-artifacts
```

Neu smoke fail, doc log loi va sua path/layout truoc khi train e1.

## 6. Train va evaluate

Artifacts nen nam tren Drive de khong mat khi runtime bi reset:

```python
!python -m src.train --exp e1_main \
  --data-dir /content/drive/MyDrive/fairface \
  --split-dir /content/drive/MyDrive/fairface-artifacts/splits \
  --out-dir /content/drive/MyDrive/fairface-artifacts
```

Chi dung validation de chon checkpoint:

```python
!python -m src.evaluate --exp e1_main --split val \
  --data-dir /content/drive/MyDrive/fairface \
  --split-dir /content/drive/MyDrive/fairface-artifacts/splits \
  --out-dir /content/drive/MyDrive/fairface-artifacts
```

Sau khi chot experiment, chay test mot lan:

```python
!python -m src.evaluate --exp e1_main --split test \
  --data-dir /content/drive/MyDrive/fairface \
  --split-dir /content/drive/MyDrive/fairface-artifacts/splits \
  --out-dir /content/drive/MyDrive/fairface-artifacts
```

Lan chay test thu hai se bi chan neu chua them `--force`. Dung `--resume` de tiep tuc mot experiment dang do.

## 7. Chay demo

```python
!python app/demo.py \
  --ckpt /content/drive/MyDrive/fairface-artifacts/e1_main/best.pth \
  --share
```

Colab se in mot public Gradio URL. Ket qua chi co y nghia hoc thuat; khong dung cho quyet dinh ve con nguoi.

## 8. Thu muc ket qua

Sau khi train, thu muc experiment co `best.pth`, `history.json`, `config.json`, `val_metrics.json`, `metrics.json` va cac file du doan. Giu `--data-dir`, `--split-dir` va `--out-dir` dong nhat giua train/evaluate.

## Xu ly loi thuong gap

- `torch.cuda.is_available()` la `False`: kiem tra lai runtime GPU, sau do restart runtime.
- `FileNotFoundError`: dataset khong dung layout; xem lai `--data-dir` va cot `file` trong CSV.
- Het RAM: giam batch trong `src/config.py`, dat `FAIRFACE_NUM_WORKERS=2`, hoac dung experiment `e8_resnet18` de kiem tra pipeline.
- Mat ket qua sau khi reset runtime: luu `--out-dir` tren Drive, khong luu trong `/content`.
