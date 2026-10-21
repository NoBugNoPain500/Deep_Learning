# ResNet50 FairFace Gender

Project phan lop nhi phan `Male/Female` tren FairFace padding 0.25 bang PyTorch.

## Local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

Local smoke test khong quet hoac huan luyen toan bo dataset. Du lieu mac dinh duoc tim trong `dataset/`; co the doi bang `--data-dir` va `--split-dir`.

```bash
python -m src.train --exp e1_main --smoke
python -m src.evaluate --exp e1_main --split val --smoke
python -m src.evaluate --exp e1_main --split test
python app/demo.py --ckpt artifacts/e1_main/best.pth
```

## Kaggle

Sao chep code vao notebook, cai `requirements-kaggle.txt`, sua `DATA_DIR` trong `src/config.py` neu ten dataset khac, sau do chay train/evaluate. Checkpoint va metric nam trong `/kaggle/working`.

## Luu y

Validation duoc tach tu train theo `gender + race` voi seed 42. CSV FairFace val la test chinh thuc va khong duoc dung de chon checkpoint. Khi bat CUDA, cudnn benchmark duoc phep de tang toc nen ket qua khong deterministic tuyet doi.

Nguon du lieu: FairFace, giay phep CC BY 4.0. Ket qua phan lop chi co y nghia hoc thuat va khong nen dung cho quyet dinh ve con nguoi.
