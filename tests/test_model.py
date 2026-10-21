import torch

from src.model import build_model, count_params, freeze_bn, set_trainable


def test_supported_models_output_shape():
    for arch in ("resnet18", "resnet50", "vgg16", "mobilenet_v2"):
        model = build_model(arch, pretrained=False)
        with torch.no_grad():
            assert model(torch.randn(2, 3, 224, 224)).shape == (2, 2)


def test_resnet_head_count_and_freeze_bn():
    model = build_model("resnet50", pretrained=False)
    set_trainable(model, "head")
    assert count_params(model)[1] == 4098
    freeze_bn(model)
    assert all(
        not parameter.requires_grad
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        for parameter in module.parameters()
    )


def test_layer3_4_only():
    model = build_model("resnet18", pretrained=False)
    set_trainable(model, "layer3_4")
    assert all(
        name.startswith(("layer3", "layer4", "fc"))
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
