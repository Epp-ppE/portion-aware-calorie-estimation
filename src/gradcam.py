"""Grad-CAM: visualizes which pixels the model actually responded to for a
given prediction, so we can check whether it's looking at the food or at
something spurious (e.g. the scanning rig background) — the "shortcut
learning check" from the plan.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer=None):
        """target_layer defaults to the backbone's last conv layer
        (conv_head for EfficientNet-B0) — the standard Grad-CAM choice,
        since it's the last point where spatial location is still
        preserved before global pooling collapses it away."""
        self.model = model
        self.activations = None
        self.gradients = None

        if target_layer is None:
            target_layer = model.backbone.conv_head
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, target_index):
        """image_tensor: (1, 3, 224, 224), already transformed/normalized.
        target_index: which of the 4 outputs to explain (0=calories,
        1=protein, 2=fat, 3=carb — see TARGET_NAMES in data.py).

        Returns a (224, 224) heatmap, values 0-1, 1 = most responsible for
        this prediction.
        """
        self.model.zero_grad()
        output = self.model(image_tensor)
        score = output[0, target_index]
        score.backward()

        # average each channel's gradient over space -> how much that
        # channel mattered for this prediction, then weight that channel's
        # activation map by it and sum across channels
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # only positive influence, same as the original Grad-CAM paper

        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(image, cam, alpha=0.5):
    """image: PIL Image (any size). cam: (H, W) tensor, values 0-1.

    Returns a PIL Image: the original image with a red-hot heatmap overlay,
    resized to match the heatmap's resolution.
    """
    cam_np = cam.numpy()
    heatmap = np.zeros((*cam_np.shape, 3), dtype=np.uint8)
    heatmap[..., 0] = (cam_np * 255).astype(np.uint8)  # red channel = how responsible this pixel was
    heatmap_image = Image.fromarray(heatmap).convert("RGB")

    base = image.convert("RGB").resize((cam_np.shape[1], cam_np.shape[0]))
    return Image.blend(base, heatmap_image, alpha=alpha)
