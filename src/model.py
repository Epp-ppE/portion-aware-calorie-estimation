import timm #for importing EfficientNet-B0
import torch
import torch.nn as nn


class CalorieRegressor(nn.Module):
    """EfficientNet-B0 backbone with its classifier replaced by a 4-output
    regression head (calories, protein, fat, carb), instead of ImageNet's
    1000-class output.
    """

    def __init__(self, backbone_name="efficientnet_b0", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=4)

    def forward(self, x):
        return self.backbone(x)

    def set_backbone_frozen(self, frozen):
        """True: freeze every backbone weight except the final regression
        head, for head-only warmup. False: unfreeze everything, for full
        fine-tuning."""
        for name, param in self.backbone.named_parameters():
            is_head = "classifier" in name or "fc" in name
            param.requires_grad = is_head or not frozen


class TargetNormalizer:
    """Normalize each of the 4 targets by its train-set mean/std, so
    calories (~hundreds) doesn't dominate the loss over fat/carb/protein
    (~tens). Un-normalize predictions back to real units for reporting.
    """

    def __init__(self, means, stds):
        self.means = torch.tensor(means, dtype=torch.float32)
        self.stds = torch.tensor(stds, dtype=torch.float32)

    @classmethod
    def from_targets(cls, targets):
        """targets: tensor of shape (N, 4), e.g. all training labels stacked."""
        return cls(targets.mean(dim=0).tolist(), targets.std(dim=0).tolist())

    def normalize(self, targets):
        return (targets - self.means.to(targets.device)) / self.stds.to(targets.device)

    def denormalize(self, targets):
        return targets * self.stds.to(targets.device) + self.means.to(targets.device)
