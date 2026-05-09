import cv2
import numpy as np
import torch


class GradCAM:
    """Grad-CAM implementation for CNN-based classifiers (e.g., EfficientNet-B0)."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()

        self.target_layer = target_layer if target_layer is not None else self._find_last_conv_layer()
        self.activations = None
        self.gradients = None
        self._hooks = []
        self._register_hooks()

    def _find_last_conv_layer(self):
        """Automatically locate the last Conv2d layer in the model."""
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d layer found in model for Grad-CAM.")
        return last_conv

    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""

        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            _ = grad_input
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def __del__(self):
        if hasattr(self, "_hooks"):
            self.remove_hooks()

    def compute(self, input_tensor, target_class=None):
        """
        Compute normalized Grad-CAM heatmap in [0, 1] for one input tensor.

        Returns:
            heatmap (np.ndarray): 2D normalized heatmap in [0, 1]
            logits (torch.Tensor): model output logits
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        # Backpropagate score for the selected class.
        score = logits[:, target_class].sum()
        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hooks did not capture gradients/activations.")

        # Global average pooling over gradient spatial dimensions.
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted sum of forward activations.
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # ReLU to keep only positive influence regions.
        cam = torch.relu(cam)

        # Convert to numpy heatmap.
        heatmap = cam[0, 0].cpu().numpy()

        # Normalize to [0, 1].
        min_val = float(heatmap.min())
        max_val = float(heatmap.max())
        if max_val - min_val > 1e-8:
            heatmap = (heatmap - min_val) / (max_val - min_val)
        else:
            heatmap = np.zeros_like(heatmap, dtype=np.float32)

        return heatmap.astype(np.float32), logits


def generate_gradcam(model, input_tensor, target_class):
    """
    Convenience function required by integration spec.

    Returns:
        np.ndarray heatmap normalized to [0, 1].
    """
    gradcam = GradCAM(model)
    try:
        heatmap, _ = gradcam.compute(input_tensor=input_tensor, target_class=target_class)
    finally:
        gradcam.remove_hooks()
    return heatmap


def overlay_heatmap_on_rgb(original_rgb, heatmap, alpha=0.4):
    """
    Resize heatmap, apply JET colormap, and blend onto an RGB image.

    Args:
        original_rgb (np.ndarray): HxWx3 uint8 RGB image.
        heatmap (np.ndarray): 2D heatmap in [0, 1].
        alpha (float): blend factor for heatmap.

    Returns:
        np.ndarray: HxWx3 uint8 RGB overlay image.
    """
    height, width = original_rgb.shape[:2]

    resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
    heatmap_u8 = np.uint8(np.clip(resized, 0.0, 1.0) * 255.0)

    # cv2 colormap is BGR by default.
    colored_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(original_rgb, 1.0 - alpha, colored_rgb, alpha, 0)
    return blended
