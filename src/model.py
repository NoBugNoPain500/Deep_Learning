from pathlib import Path

import torch
from torch import nn
from torchvision import models


def build_model(
    arch: str,
    pretrained: bool = True,
    num_classes: int = 2,
    weights_path: str | None = None,
) -> nn.Module:
    constructors = {
        "resnet50": models.resnet50,
        "resnet18": models.resnet18,
        "vgg16": models.vgg16,
        "mobilenet_v2": models.mobilenet_v2,
    }
    if arch not in constructors:
        raise ValueError(f"Unsupported architecture: {arch}")
    kwargs = {"weights": None}
    weight_types = {
        "resnet50": models.ResNet50_Weights,
        "resnet18": models.ResNet18_Weights,
        "vgg16": models.VGG16_Weights,
        "mobilenet_v2": models.MobileNet_V2_Weights,
    }
    if pretrained and weights_path is None:
        kwargs["weights"] = weight_types[arch].DEFAULT
    model = constructors[arch](**kwargs)
    if arch.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "vgg16":
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    if weights_path:
        state = torch.load(Path(weights_path), map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
    return model


def _head_names(model: nn.Module) -> set[str]:
    if hasattr(model, "fc"):
        return {"fc"}
    return {"classifier"}


def set_trainable(model: nn.Module, mode: str) -> None:
    if mode not in {"head", "layer4", "layer3_4", "all"}:
        raise ValueError(f"Unsupported train mode: {mode}")
    for parameter in model.parameters():
        parameter.requires_grad = mode == "all"
    if mode != "all":
        prefixes = _head_names(model)
        if mode in {"layer4", "layer3_4"} and hasattr(model, "layer4"):
            prefixes.add("layer4")
        if mode == "layer3_4" and hasattr(model, "layer3"):
            prefixes.add("layer3")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(
                name == prefix or name.startswith(prefix + ".") for prefix in prefixes
            )


def param_groups(
    model: nn.Module, mode: str, lr_backbone: float, lr_head: float
) -> list[dict]:
    set_trainable(model, mode)
    head = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad
        and any(n == h or n.startswith(h + ".") for h in _head_names(model))
    ]
    head_ids = {id(parameter) for parameter in head}
    backbone = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    return [{"params": backbone, "lr": lr_backbone}, {"params": head, "lr": lr_head}]


def freeze_bn(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False


def count_params(model: nn.Module) -> tuple[int, int]:
    return sum(p.numel() for p in model.parameters()), sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
