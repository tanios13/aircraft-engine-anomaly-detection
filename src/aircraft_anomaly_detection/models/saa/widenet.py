from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import resize, to_pil_image
from typing_extensions import deprecated

from aircraft_anomaly_detection.interface.model import SaliencyModelInterface


class ResizeLongestSide:
    """Resize to a target *long side* while keeping aspect ratio.

    This helper provides both NumPy and Torch flavours so the same maths can run
    on the CPU for quick vis or on the GPU for prompt transformations.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """Return resized HxWxC uint8 image (long side == `target_length`)."""
        new_h, new_w = self._target_shape(*image.shape[:2])
        return np.array(resize(to_pil_image(image), (new_h, new_w)))

    def apply_coords(self, coords: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
        coords = deepcopy(coords).astype(float)
        scale_h, scale_w = self._scale_factors(original_size)
        coords[..., 0] *= scale_w
        coords[..., 1] *= scale_h
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
        boxes_ = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes_.reshape(-1, 4)

    # -------- Torch API -------------------------------------------------------
    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        new_h, new_w = self._target_shape(*image.shape[-2:])
        return F.interpolate(  # type: ignore
            image[None],
            (new_h, new_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

    def apply_coords_torch(self, coords: torch.Tensor, original_size: tuple[int, int]) -> torch.Tensor:
        coords = coords.clone().float()
        scale_h, scale_w = self._scale_factors(original_size)
        coords[..., 0] *= scale_w
        coords[..., 1] *= scale_h
        return coords

    def apply_boxes_torch(self, boxes: torch.Tensor, original_size: tuple[int, int]) -> torch.Tensor:
        boxes_ = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes_.reshape(-1, 4)

    # -------- Internals -------------------------------------------------------
    def _target_shape(self, old_h: int, old_w: int) -> tuple[int, int]:
        scale = self.target_length / max(old_h, old_w)
        return int(round(old_h * scale)), int(round(old_w * scale))

    def _scale_factors(self, original_size: tuple[int, int]) -> tuple[float, float]:
        old_h, old_w = original_size
        new_h, new_w = self._target_shape(old_h, old_w)
        return new_h / old_h, new_w / old_w


class ModelINet(SaliencyModelInterface):
    """Multi-scale ImageNet feature extractor for SAA+ self-similarity maps.

    Args:
        device:         Torch device string. Auto-selects *cuda* if available.
        backbone_name:  Any **timm** backbone with ``features_only=True``. The
                        default *wide_resnet50_2* matches the original paper.
        out_indices:    Which stages to concatenate; higher index ⇒ deeper feat.
        resize_longest: Final canvas side length (default 1024, as SAA+).
        pool_last:      If ``True`` also append a global pooled vector (rare).
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        *,
        device: str | None = None,
        resize_longest: int = 1024,
    ) -> None:
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ---- backbone --------------------------------------------------------
        kwargs: dict[str, Any] = {"features_only": True, "out_indices": (1, 2, 3)}
        self.backbone = (
            timm.create_model(
                backbone_name,
                pretrained=True,
                **kwargs,
            )
            .to(self.device)
            .eval()
        )
        # get model specific transforms (normalization, resize)
        self.data_config = timm.data.resolve_model_data_config(self.backbone)  # type: ignore
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
                transforms.Normalize(
                    mean=self.data_config["mean"], std=self.data_config["std"]
                ),  # Normalize with ImageNet stats
            ]
        )

        # ---- geometry helpers ------------------------------------------------
        self.resize_longest: int = resize_longest
        self._resizer = ResizeLongestSide(resize_longest)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    @deprecated("ModelINet.set_img_size is deprecated. The size is taken from the input image")
    def set_img_size(self, size: int) -> None:
        """Change the target square canvas side length at runtime."""
        self.resize_longest = size
        self._resizer = ResizeLongestSide(size)

    def generate_saliency_map(self, image: Image.Image) -> torch.Tensor:
        """Return L2-normalised dense features and resize ratios.

                Args:
                    image: Original RGB image as **PIL.Image**.

                Returns:
                    features:  (C_total, H', W') tensor on `self.device`.
                    ratio_h:   H_original / resize_longest.
                    ratio_w:   W_original / resize_longest.
        from warnings import deprecated
        """
        # ---- preprocess -------------------------------------------------
        self.resize_longest = max(image.size)
        self._resizer = ResizeLongestSide(self.resize_longest)

        img = np.array(image.convert("RGB"))  # Convert PIL.Image to NumPy array
        x_np = self._resizer.apply_image(img)  # aspect‑ratio resize
        x = self.transforms(x_np).to(self.device)  # to tensor, normalize

        # pad to square
        pad_h = self.resize_longest - x.shape[1]
        pad_w = self.resize_longest - x.shape[2]
        x = F.pad(x, (0, pad_w, 0, pad_h))
        x = x.unsqueeze(0)  # add batch dim

        ratio_h = img.shape[0] / self.resize_longest
        ratio_w = img.shape[1] / self.resize_longest

        # ---- backbone forward -------------------------------------------
        with torch.no_grad():
            # get features from all stages
            feats: Iterable[torch.Tensor] = self.backbone(x)
        feats = list(feats)
        # upsample deeper maps to shallow map size and concat
        base_size = feats[0].shape[-2:]
        feats = [feats[0]] + [F.interpolate(f, base_size) for f in feats[1:]]
        feats_cat = torch.cat(feats, dim=1).squeeze(0)  # (C_total,H,W)
        feats_cat = F.normalize(feats_cat, dim=0)  # per‑channel L2 norm

        return feats_cat
