# AGENT GUIDE – Viết code cho dự án ResNet50 phân lớp giới tính (FairFace)

> Tài liệu dành cho **AI coding agent** (Claude Code, Cursor, Copilot Agent…). Có thể đổi tên thành `AGENTS.md` hoặc `CLAUDE.md` ở gốc repo.
> Quy trình chạy Google Colab nằm trong `COLAB_SETUP.md`. Nếu tài liệu mâu thuẫn, file này quyết định về **code**, `COLAB_SETUP.md` quyết định về **quy trình chạy**.

## Prompt mở đầu (dán cho agent)

```
Đọc AGENT_GUIDE_ResNet50_FairFace_Code.md ở gốc repo. Thực hiện lần lượt các milestone M0 → M7.
Sau mỗi milestone: chạy lệnh kiểm tra của milestone đó, báo kết quả, rồi dừng chờ tôi xác nhận trước khi sang milestone kế tiếp.
Máy này yếu, không có GPU: tuyệt đối không chạy huấn luyện thật hay quét toàn bộ dataset ở local.
```

---

## 1. Bối cảnh bài toán

- Phân lớp nhị phân giới tính (Male / Female) từ ảnh khuôn mặt, dataset **FairFace**, bản **padding 0.25** (ảnh 224×224).
- Mô hình chính: **ResNet50 pretrained ImageNet**, thay FC thành `Linear(2048, 2)`, huấn luyện 2 giai đoạn (feature extraction → fine-tuning).
- Nhãn ở CSV: `fairface_label_train.csv`, `fairface_label_val.csv`, cột `file, age, gender, race, service_test`. Cột `file` là đường dẫn tương đối (ví dụ `train/1.jpg`).
- Chia tập: **validation chính thức = test** (10.954 ảnh); tách 10% train làm **val** (stratify theo `gender + race`, `random_state=42`).
- Framework: PyTorch + torchvision, scikit-learn, pandas, matplotlib. Demo bằng Gradio.

## 2. Ràng buộc môi trường (đọc kỹ)

| Môi trường | Tài nguyên | Agent được phép |
| --- | --- | --- |
| **Local** | CPU yếu, **không GPU** | Viết code, chạy unit test và **smoke test ≤ 64 ảnh**, chạy demo 1 ảnh, vẽ biểu đồ từ CSV/JSON |
| **Google Colab** (người dùng tự chạy) | GPU T4/L4/A100 tùy runtime, Google Drive | Agent **không chạy được**; chỉ viết code + notebook mỏng để người dùng chạy |
| **Kaggle** (tùy chọn) | GPU P100/T4, 4 CPU | Agent **không chạy được**; chỉ viết code + notebook mỏng để người dùng chạy |

Quy tắc cứng:

1. **Không** chạy train/eval toàn bộ dữ liệu ở local. Mọi lệnh chạy ở local phải có `--smoke` hoặc dùng dữ liệu giả của test.
2. **Không** quét toàn bộ ~97 nghìn ảnh ở local (hash, thống kê kích thước…). Script quét toàn bộ phải có `--limit N` để test local, chạy đủ trên Kaggle.
3. Code phải **chạy được trên local, Colab và Kaggle mà không sửa tay**: đường dẫn có default theo runtime và có thể override bằng CLI.
4. **Không** dùng tập test để chọn siêu tham số, early stopping hay chọn checkpoint. Chỉ dùng val.
5. **Không** gọi mạng khi `import`. Việc tải trọng số chỉ xảy ra trong `build_model`, và phải có đường lùi `weights_path` cho Kaggle khi tắt Internet.
6. **Không** commit dữ liệu, checkpoint, thư mục `artifacts/`, `data/` (thêm vào `.gitignore`).
7. Logic nằm trong `src/`. Notebook chỉ là lớp mỏng gọi lệnh (tối đa ~10 cell).

## 3. Cấu trúc repo cần tạo

```
resnet50-fairface-gender/
├── AGENTS.md                  # bản sao/đổi tên của file này (tùy chọn)
├── README.md
├── requirements.txt           # dùng cho local (torch bản CPU, xem M0)
├── requirements-kaggle.txt    # chỉ gói còn thiếu trên Kaggle
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py               # set_seed, save_json, logger, timer
│   ├── data.py
│   ├── model.py
│   ├── experiments.py         # định nghĩa E1…E9
│   ├── train.py
│   ├── evaluate.py
│   ├── analysis.py            # nhẹ, chạy local
│   └── gradcam.py
├── scripts/
│   └── check_duplicates.py    # chạy đủ trên Kaggle-CPU
├── app/
│   └── demo.py
├── notebooks/
│   ├── 02_train_kaggle.ipynb
│   ├── 03_experiments_kaggle.ipynb
│   └── 04_analysis_local.ipynb
├── tests/
│   ├── conftest.py            # sinh dataset giả
│   ├── test_data.py
│   ├── test_model.py
│   └── test_smoke.py
├── splits/                    # train_split.csv, val_split.csv, test.csv
└── artifacts/                 # kết quả tải từ Kaggle (không commit)
```

## 4. Quy ước code

- Python ≥ 3.10, type hints cho hàm công khai, docstring ngắn (tiếng Việt hoặc Anh, nhất quán).
- Dùng `pathlib.Path`, không nối chuỗi đường dẫn.
- In log bằng `logging` (INFO), có `flush`; mỗi epoch in: `ep, train_loss/acc, val_loss/acc/f1, lr, thời gian`.
- Mọi tham số nằm trong `config.py` hoặc `experiments.py`, **không** hard-code số ma thuật trong hàm.
- Nhãn thống nhất: `CLASSES = ["Male", "Female"]`, `Male=0`, `Female=1`. Chỉ định nghĩa **một lần** trong `config.py`.
- Seed: `set_seed(42)` cho `random`, `numpy`, `torch`, `torch.cuda`; `DataLoader` dùng `generator` + `worker_init_fn`. Được bật `cudnn.benchmark = True` (không cần deterministic tuyệt đối), nhưng ghi chú trong README.
- Không dùng thư viện lạ khi không cần. Cho phép: `torch, torchvision, numpy, pandas, scikit-learn, matplotlib, Pillow, pytorch-grad-cam, gradio, opencv-python-headless`.

---

## 5. Đặc tả từng module

### 5.1 `src/config.py`

```python
ON_KAGGLE = Path("/kaggle/input").exists()
DATA_DIR   # Kaggle: "/kaggle/input/<ten-dataset>" (có TODO để người dùng sửa) | Local: "data/fairface"
OUT_DIR    # Kaggle: "/kaggle/working"            | Local: "artifacts"
SPLIT_DIR  # Kaggle: OUT_DIR/"splits"             | Local: "splits"   (/kaggle/input là read-only)
NUM_WORKERS  # Kaggle: os.cpu_count() | Local: 0
IMG_SIZE, BATCH, SEED = 224, 32, 42
CLASSES = ["Male", "Female"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
```

### 5.2 `src/utils.py`

`set_seed(seed)`, `seed_worker(worker_id)`, `save_json(obj, path)`, `get_logger(name)`, `Timer` (context manager đo thời gian).

### 5.3 `src/data.py`

| Hàm / lớp | Hành vi bắt buộc |
| --- | --- |
| `load_splits(data_dir, split_dir, seed) -> (df_tr, df_va, df_te)` | Nếu `split_dir/*.csv` đã có thì đọc lại; nếu chưa thì tạo: đọc 2 CSV gốc, `train_test_split(test_size=0.1, random_state=seed, stratify=gender+"_"+race)`, lưu 3 CSV. Trả về DataFrame có thêm cột `label` (0/1). |
| Kiểm tra trong `load_splits` | `assert` tập giá trị `gender` ⊆ {"Male","Female"}; không có `file` nào trùng giữa 3 tập; đủ cột `file, age, gender, race`; kiểm tra tồn tại 5 file ảnh đầu tiên mỗi tập, nếu thiếu → `FileNotFoundError` với gợi ý kiểm tra `DATA_DIR`. |
| `FairFaceGender(df, root, transform)` | `__getitem__` trả `(tensor, label)`; mở ảnh bằng PIL, `ImageOps.exif_transpose`, `.convert("RGB")`. Thuộc tính `classes = CLASSES`. |
| `get_transforms(train: bool, augment: bool = True)` | Train + augment: `Resize → RandomHorizontalFlip → RandomRotation(15) → ColorJitter(brightness=0.15, hue=0.05) → ToTensor → Normalize`. Ngược lại: `Resize → ToTensor → Normalize` (không augmentation). |
| `make_loaders(cfg, smoke=False)` | Trả `train_dl, val_dl, test_dl`. `shuffle=True` chỉ cho train; `pin_memory = torch.cuda.is_available()`; `persistent_workers` khi `num_workers>0`. Nếu `smoke=True`: lấy tối đa 48 ảnh train, 16 val, 16 test (stratify theo nhãn), `batch=8`, `num_workers=0`. |
| `class_weights(df) -> Tensor` | Trọng số nghịch tần suất theo `label`, dùng cho `CrossEntropyLoss(weight=…)` khi người dùng bật `use_class_weights`. |

### 5.4 `src/model.py`

```python
def build_model(arch: str, pretrained: bool = True, num_classes: int = 2,
                weights_path: str | None = None) -> nn.Module: ...
def set_trainable(model, mode: str) -> None: ...
def param_groups(model, mode: str, lr_backbone: float, lr_head: float) -> list[dict]: ...
def freeze_bn(model) -> None: ...
def count_params(model) -> tuple[int, int]: ...   # (tổng, trainable)
```

- `arch` ∈ `{"resnet50","resnet18","vgg16","mobilenet_v2"}`. Thay đúng lớp phân loại cuối: ResNet → `model.fc`; VGG16 → `classifier[6]`; MobileNetV2 → `classifier[1]`.
- `pretrained=True` dùng `weights=…Weights.DEFAULT`; nếu có `weights_path` thì nạp `state_dict` từ file thay vì tải mạng.
- `set_trainable` hỗ trợ `mode` ∈ `{"head","layer4","layer3_4","all"}`. Với ResNet: `head` chỉ train `fc` (khi đó số tham số trainable của ResNet50 **phải bằng 4.098**). Với kiến trúc khác, định nghĩa tương đương và ghi docstring.
- `freeze_bn(model)`: đặt mọi `BatchNorm` về `eval()` và `requires_grad=False`. Vì `model.train()` sẽ bật lại BN, hàm train phải gọi lại `freeze_bn` sau mỗi lần `model.train()` khi cờ `freeze_bn_stage1=True`.

### 5.5 `src/experiments.py`

Dùng `dataclass Experiment`:

```python
@dataclass
class Experiment:
    name: str; arch: str = "resnet50"; pretrained: bool = True; augment: bool = True
    stage1_epochs: int = 8;  stage1_lr: float = 1e-3
    stage2_epochs: int = 15; lr_backbone: float = 1e-5; lr_head: float = 1e-4
    unfreeze: str = "layer3_4"          # head | layer4 | layer3_4 | all
    freeze_bn_stage1: bool = False
    use_class_weights: bool = False
    patience: int = 4
    selection_metric: str = "val_acc"   # hoặc "val_f1"
```

`EXPERIMENTS: dict[str, Experiment]` gồm:

| Tên | Khác biệt so với `e1_main` |
| --- | --- |
| `e1_main` | Mô hình chính (mặc định ở trên) |
| `e2_scratch` | `pretrained=False`, `unfreeze="all"`, bỏ stage 1, train toàn mạng từ đầu với `stage2_epochs = stage1+stage2` và `lr_backbone=lr_head=1e-3` |
| `e3_no_aug` | `augment=False` |
| `e4_freeze` | Chỉ stage 1 (`stage2_epochs=0`) |
| `e5_ft_l4` | `unfreeze="layer4"` |
| `e6_ft_all` | `unfreeze="all"` |
| `e7_vgg16` | `arch="vgg16"` |
| `e8_resnet18` | `arch="resnet18"` |
| `e9_mobilenetv2` | `arch="mobilenet_v2"` |

Tất cả dùng cùng seed, batch size và **cùng số epoch tối đa** (với E2 là tổng stage1+stage2 để công bằng).

### 5.6 `src/train.py`

```python
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=True) -> dict
def fit(model, loaders, optimizer, scheduler, exp, out_dir, tag, resume=False) -> list[dict]
def main()   # CLI
```

- `run_epoch` trả `{"loss", "acc", "f1"}` (F1 macro tính bằng scikit-learn từ dự đoán gom lại). Dùng `torch.autocast` + `GradScaler` **chỉ khi** có CUDA; trên CPU tắt AMP.
- `fit`: mỗi epoch ghi `history` (list dict), lưu `last.pth` (model, optimizer, scheduler, epoch, best_metric, history) và `best_<tag>.pth` theo `exp.selection_metric` **trên val**; early stopping theo `patience`; hỗ trợ `resume` từ `last.pth`.
- Luồng `main` cho `--exp <tên>`:
  1. Xây model, loaders, loss.
  2. **Stage 1** (nếu `stage1_epochs > 0`): `set_trainable("head")`, Adam trên tham số head, `lr=stage1_lr`.
  3. Nạp lại `best_stage1.pth`.
  4. **Stage 2** (nếu `stage2_epochs > 0`): `set_trainable(exp.unfreeze)`, `param_groups`, `CosineAnnealingLR(T_max=stage2_epochs)`.
  5. Ghi `history.json` (gộp hai giai đoạn, thêm trường `stage`), `config.json` (dump `Experiment` + seed + số tham số), `best.pth` cuối cùng vào `OUT_DIR/<exp.name>/`.
- Cờ CLI: `--exp`, `--smoke`, `--resume`, `--out-dir`, `--weights-path`, `--epochs-override` (để test nhanh). `--smoke` ép `epochs=1` mỗi stage và dùng loader nhỏ.

### 5.7 `src/evaluate.py`

```python
def predict(model, loader, df, device) -> pd.DataFrame   # file,y_true,y_pred,prob_male,prob_female,race,age
def compute_metrics(preds: pd.DataFrame) -> dict
def measure_latency(model, device, n=50) -> float        # giây/ảnh, batch=1
def main()   # CLI: --exp, --split {val,test}, --smoke, --force
```

- `compute_metrics` trả: accuracy, precision/recall/F1 từng lớp và macro, confusion matrix (list lồng), `n_params`, `latency_s_per_img`.
- **Bảo vệ test set:** khi `--split test`, nếu `OUT_DIR/<exp>/metrics.json` đã tồn tại thì dừng và in cảnh báo, chỉ chạy lại khi có `--force`.
- Xuất `test_preds.csv` và `metrics.json` vào thư mục thí nghiệm.

### 5.8 `src/analysis.py` (nhẹ, chạy local từ file đã xuất)

- `group_accuracy(preds, by="race" | "age") -> DataFrame` (accuracy + số mẫu mỗi nhóm).
- `plot_history(history_json, path)` – loss/accuracy theo epoch, đánh dấu ranh giới stage 1/stage 2.
- `plot_confusion(cm, classes, path)`.
- `compare_experiments(artifacts_dir) -> DataFrame` – gộp `metrics.json`, `config.json` của mọi thí nghiệm thành một bảng (accuracy, macro-F1, số tham số, latency), xuất `comparison.csv` và `comparison.md`.
- `top_errors(preds, k=24)` – lọc dòng sai, sắp theo độ tin cậy giảm dần.

### 5.9 `src/gradcam.py`

- `explain(model, df_rows, root, out_dir, n=16)` dùng `pytorch-grad-cam` (`GradCAM`) với `target_layers=[model.layer4[-1]]`, chồng bản đồ nhiệt lên ảnh gốc và lưu PNG.
- Chạy được trên CPU, giới hạn `n` ≤ 32; chỉ hỗ trợ ResNet (raise lỗi rõ ràng nếu `arch` khác).

### 5.10 `scripts/check_duplicates.py`

- Tham số: `--limit N` (cho test local), `--out duplicates.csv`.
- Hash MD5 (trùng chính xác) và tùy chọn dHash 9×8 tự cài bằng Pillow (gần trùng). So sánh **giữa train và test**, in số cặp trùng và lưu danh sách.
- Chỉ chạy đủ trên Kaggle-CPU.

### 5.11 `app/demo.py`

- Tham số: `--ckpt artifacts/e1_main/best.pth`. Chạy CPU.
- `preprocess(img)`: `exif_transpose → RGB → (tùy chọn) cắt mặt → Resize(224,224) → Normalize`, **tái dùng `get_transforms(train=False)`** để đảm bảo giống lúc huấn luyện.
- Cắt mặt: mặc định **OpenCV Haar Cascade** (nhẹ cho máy yếu), nới lề ≈ 25% mỗi phía; MTCNN là tùy chọn (`--detector mtcnn`). Nếu không phát hiện mặt → dùng ảnh gốc và hiện cảnh báo.
- Đầu ra Gradio: `gr.Label` với xác suất của `CLASSES`; thêm ghi chú rằng kết quả chỉ mang ý nghĩa học thuật.

---

## 6. Test (phải chạy được ở local, không cần FairFace thật)

`tests/conftest.py` sinh dataset giả: ~40 ảnh RGB 224×224 ngẫu nhiên vào thư mục tạm + CSV đúng định dạng FairFace (đủ `Male/Female`, vài giá trị `race`, `age`).

| File | Kiểm tra |
| --- | --- |
| `test_data.py` | Ánh xạ nhãn Male=0/Female=1; 3 tập không giao nhau; tỉ lệ stratify xấp xỉ đúng; transform val/test **không** có phép ngẫu nhiên; thiếu ảnh → báo lỗi rõ ràng |
| `test_model.py` | Với mỗi `arch` (`pretrained=False`): output `(2, 2)`; ResNet50 mode `head` có đúng **4.098** tham số trainable; `freeze_bn` thực sự đóng băng BN; `layer3_4` chỉ mở đúng các nhóm tham số |
| `test_smoke.py` | Chạy `train --smoke` trên dữ liệu giả 1 epoch mỗi stage (ResNet18 hoặc `--epochs-override 1`) trong < 90 giây CPU; kiểm tra có `best.pth`, `history.json`, `config.json`; chạy `evaluate --smoke` ra `metrics.json`; chạy lại test split lần 2 phải bị chặn nếu không có `--force` |

Lệnh: `pytest -q`. Không test nào được phụ thuộc Internet hoặc GPU.

---

## 7. Notebook mỏng cho Kaggle

`02_train_kaggle.ipynb` (người dùng chạy bằng *Save & Run All*):

```python
!find /kaggle/input -maxdepth 3 | head -30      # xác định DATA_DIR rồi sửa src/config.py
!git clone https://github.com/<user>/resnet50-fairface-gender.git
%cd resnet50-fairface-gender
!pip install -q -r requirements-kaggle.txt
!python -m src.train --exp e1_main
!python -m src.evaluate --exp e1_main --split test
```

`03_experiments_kaggle.ipynb`: vòng lặp qua danh sách thí nghiệm, mỗi nhóm nhỏ một lần commit. Mỗi thí nghiệm gọi cùng hai lệnh `train` / `evaluate` ở trên.

`04_analysis_local.ipynb`: gọi các hàm trong `analysis.py` trên `artifacts/`.

---

## 8. Lộ trình milestone cho agent

Sau mỗi milestone: chạy lệnh kiểm tra, báo kết quả, **dừng chờ xác nhận**.

| M | Việc cần làm | Lệnh kiểm tra | Xong khi |
| --- | --- | --- | --- |
| **M0** | Tạo khung repo, `.gitignore`, `requirements*.txt`. Local cài torch bản CPU: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` | `python -c "import torch, torchvision; print(torch.__version__)"` | Import không lỗi |
| **M1** | `config.py`, `utils.py`, `data.py`, `conftest.py`, `test_data.py` | `pytest tests/test_data.py -q` | Test xanh |
| **M2** | `model.py`, `experiments.py`, `test_model.py` | `pytest tests/test_model.py -q` | Test xanh, 4.098 tham số khớp |
| **M3** | `train.py`, `evaluate.py`, `test_smoke.py` | `pytest -q` | Toàn bộ test xanh < ~3 phút trên CPU |
| **M4** | `analysis.py`, `gradcam.py`, `scripts/check_duplicates.py` | `python scripts/check_duplicates.py --limit 50` và test nhỏ cho `group_accuracy` | Chạy được với dữ liệu giả/limit |
| **M5** | `app/demo.py` | `python app/demo.py --ckpt <ckpt_smoke>` rồi thử 1 ảnh | Demo trả xác suất hợp lệ |
| **M6** | Các notebook mỏng + README (cách chạy Local/Kaggle, nguồn dataset, trích dẫn FairFace, giấy phép CC BY 4.0) | Đọc lại README làm theo được | Người dùng chạy thử trên Kaggle |
| **M7** | Sửa lỗi phát sinh từ lần chạy Kaggle đầu tiên (đường dẫn, Internet, tốc độ nạp dữ liệu) | Người dùng báo log | `e1_main` chạy trọn vẹn trên Kaggle |

## 9. Khi nào agent phải dừng lại hỏi người dùng

- Cấu trúc thư mục dataset trên Kaggle khác với giả định (cột `file` không khớp đường dẫn ảnh).
- Tên cột/giá trị nhãn trong CSV khác `file, age, gender, race` / `Male, Female`.
- Muốn thêm thư viện ngoài danh sách cho phép.
- Muốn thay đổi chiến lược chia tập, siêu tham số mặc định hoặc định nghĩa thí nghiệm.
- Bất kỳ việc gì cần chạy nặng ở local.

## 10. Không làm

- Không tự ý "cải thiện" kiến trúc, thêm thí nghiệm ngoài E1–E9, hay đổi tỉ lệ chia tập.
- Không tune siêu tham số bằng test set, không in/ghi metric test trong vòng huấn luyện.
- Không lưu ảnh người thật vào repo hay README (chỉ lưu ảnh kết quả cần cho báo cáo, ở `artifacts/`, không commit).
- Không để lộ token Kaggle (`kaggle.json`) vào repo.
