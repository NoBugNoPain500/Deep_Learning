from dataclasses import asdict, dataclass


@dataclass
class Experiment:
    name: str
    arch: str = "resnet50"
    pretrained: bool = True
    augment: bool = True
    stage1_epochs: int = 8
    stage1_lr: float = 1e-3
    stage2_epochs: int = 15
    lr_backbone: float = 1e-5
    lr_head: float = 1e-4
    unfreeze: str = "layer3_4"
    freeze_bn_stage1: bool = False
    use_class_weights: bool = False
    patience: int = 4
    selection_metric: str = "val_acc"

    def to_dict(self) -> dict:
        return asdict(self)


EXPERIMENTS = {"e1_main": Experiment("e1_main")}
EXPERIMENTS["e2_scratch"] = Experiment(
    "e2_scratch",
    pretrained=False,
    unfreeze="all",
    stage1_epochs=0,
    stage2_epochs=23,
    lr_backbone=1e-3,
    lr_head=1e-3,
)
EXPERIMENTS["e3_no_aug"] = Experiment("e3_no_aug", augment=False)
EXPERIMENTS["e4_freeze"] = Experiment("e4_freeze", stage2_epochs=0)
EXPERIMENTS["e5_ft_l4"] = Experiment("e5_ft_l4", unfreeze="layer4")
EXPERIMENTS["e6_ft_all"] = Experiment("e6_ft_all", unfreeze="all")
EXPERIMENTS["e7_vgg16"] = Experiment("e7_vgg16", arch="vgg16")
EXPERIMENTS["e8_resnet18"] = Experiment("e8_resnet18", arch="resnet18")
EXPERIMENTS["e9_mobilenetv2"] = Experiment("e9_mobilenetv2", arch="mobilenet_v2")
