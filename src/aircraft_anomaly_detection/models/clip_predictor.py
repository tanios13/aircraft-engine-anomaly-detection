import os
import clip
import numpy as np
import torch
from PIL import Image
from typing import List, Union

from ..interfaces import Annotation, ModelInterface

class CLIP(ModelInterface):
    def __init__(
        self,
        undamaged_idxes: List[int],     # e.g. [1] for ["defect", "no defect"]
        class_names: List[str],         # e.g. ["defect", "no defect"]
        model_name: str = "ViT-B/32",
        device: str | None = None
    ):
        """
        Compute similarities between image embeddings and text prompts via OpenAI CLIP.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, self.device)

        self.class_names = class_names
        self.undamaged_idxes = undamaged_idxes
        texts = [f"a photo of a {c}" for c in class_names]
        self.text_inputs = torch.cat([clip.tokenize(t) for t in texts]).to(self.device)

    def predict(
        self,
        image_input: str | Image.Image | np.ndarray,
    ) -> Union[Annotation, List[Annotation]]:
        """
        If `image_input` is a directory path, returns List[Annotation];
        otherwise returns a single Annotation.

        undamaged_idxes: indices of class_names considered "undamaged".
        """
        # Directory?
        if isinstance(image_input, str) and os.path.isdir(image_input):
            anns: List[Annotation] = []
            for fname in sorted(os.listdir(image_input)):
                if not fname.lower().endswith(("png","jpg","jpeg","bmp")):
                    continue
                full = os.path.join(image_input, fname)
                ann = self._predict_single(full, self.undamaged_idxes)
                anns.append(ann)
            return anns
        else:
            # Single image
            return self._predict_single(image_input, self.undamaged_idxes)

    def _predict_single(
        self,
        image_input: str | Image.Image | np.ndarray,
        undamaged_idxes: List[int]
    ) -> Annotation:
        # --- load into PIL ---
        if isinstance(image_input, str):
            if not os.path.isfile(image_input):
                raise ValueError(f"File not found: {image_input}")
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            arr = image_input
            if arr.ndim == 2:
                pil_img = Image.fromarray(arr, mode="L").convert("RGB")
            elif arr.ndim == 3:
                pil_img = Image.fromarray(arr).convert("RGB")
            else:
                raise ValueError("NumPy array must be 2D or 3D")
        else:
            raise ValueError("Unsupported input type")

        # --- preprocess & encode ---
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_feats = self.model.encode_image(img_tensor)
            txt_feats = self.model.encode_text(self.text_inputs)

        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

        # similarity and top‑k
        sims = (100.0 * img_feats @ txt_feats.T).softmax(dim=-1)
        values, indices = sims[0].topk(len(self.class_names))

        # determine binary label based on highest-class index
        top_idx = indices[0].item()
        is_undamaged = top_idx in undamaged_idxes
        label_flag = False if is_undamaged else True

        # build full predictions list
        scores = [float(v.item()) * 100 for v in values]
        labels = [self.class_names[i] for i in indices]

        return Annotation(
            image=pil_img,
            label=label_flag,
            bboxes=[],
            scores=scores,
            bboxes_labels=labels,
            mask=None
        )
