import argparse
from pathlib import Path

import gradio as gr
import torch
from PIL import Image, ImageOps
import cv2
import numpy as np

from src.config import CLASSES, DATA_DIR
from src.data import get_transforms
from src.experiments import EXPERIMENTS
from src.model import build_model


def preprocess(image: Image.Image, detector: str = "haar") -> torch.Tensor:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if detector == "haar":
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces):
            x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
            margin_x, margin_y = int(width * 0.25), int(height * 0.25)
            image = image.crop(
                (
                    max(0, x - margin_x),
                    max(0, y - margin_y),
                    min(image.width, x + width + margin_x),
                    min(image.height, y + height + margin_y),
                )
            )
    return get_transforms(train=False)(image).unsqueeze(0)


def create_demo(checkpoint: str | Path) -> gr.Interface:
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu")
    exp_name = checkpoint.parent.name
    exp = EXPERIMENTS.get(exp_name, EXPERIMENTS["e1_main"])
    model = build_model(exp.arch, pretrained=False)
    model.load_state_dict(state["model"])
    model.eval()

    def classify(image: Image.Image) -> dict[str, float]:
        with torch.no_grad():
            probabilities = model(preprocess(image)).softmax(1)[0].tolist()
        return dict(zip(CLASSES, probabilities))

    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=2),
        title="FairFace Gender Classifier",
        description="Ket qua chi co y nghia hoc thuat.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Tao public Gradio URL, can thiet khi chay tren Google Colab.",
    )
    args = parser.parse_args()
    create_demo(args.ckpt).launch(share=args.share)
