from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import IMAGENET_MEAN, IMAGENET_STD


def explain(
    model: torch.nn.Module,
    df_rows: object,
    root: str | Path,
    out_dir: str | Path,
    n: int = 16,
) -> None:
    if not hasattr(model, "layer4"):
        raise ValueError("Grad-CAM chi ho tro ResNet")
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError as error:
        raise ImportError("Cai pytorch-grad-cam de dung Grad-CAM") from error
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    target_layers = [model.layer4[-1]]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        for index, (_, row) in enumerate(df_rows.head(min(n, 32)).iterrows()):
            image = (
                Image.open(Path(root) / row["file"]).convert("RGB").resize((224, 224))
            )
            tensor = transform(image).unsqueeze(0)
            grayscale = cam(input_tensor=tensor)[0]
            base = np.asarray(image).astype(np.float32) / 255
            Image.fromarray(show_cam_on_image(base, grayscale, use_rgb=True)).save(
                output / f"gradcam_{index:03d}.png"
            )
