import os
from collections.abc import Sequence
from itertools import compress
from typing import Any, Literal

import cv2
import numpy as np
import torch
from PIL import Image

from aircraft_anomaly_detection.interface.model import (
    DetectorInterface,
    ModelInterface,
    SaliencyModelInterface,
    SegmentorInterface,
)
from aircraft_anomaly_detection.models.saa import dino, sam, widenet
from aircraft_anomaly_detection.schemas import Annotation, ObjectPrompt, PromptPair
from aircraft_anomaly_detection.viz_utils import draw_annotation

from .utils import box_area_xyxy, mask_to_box, nms_xyxy

RegionProposalModelType = Literal["GroundingDINO"] | DetectorInterface
RegionRefinerModelType = Literal["SAM"] | SegmentorInterface
SaliencyModelType = Literal["ModelINet"] | SaliencyModelInterface


class SAA(ModelInterface):
    """SAA model for anomaly detection."""

    def __init__(
        self,
        region_proposal_model: RegionProposalModelType,
        region_refiner_model: RegionRefinerModelType,
        saliency_model: SaliencyModelType,
        box_threshold: float,
        text_threshold: float,
        region_proposal_model_config: dict[str, Any] = {},
        region_refiner_model_config: dict[str, Any] = {},
        saliency_model_config: dict[str, Any] = {},
        device: str | None = None,
        debug: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Initialize the SAA model.

        Args:
            dino_config_file: Path to the DINO config file.
            dino_checkpoint: Path to the DINO checkpoint.
            sam_checkpoint: Path to the SAM checkpoint.
            box_threshold: Threshold for box filtering.
            text_threshold: Threshold for text filtering.
            out_size: Desired output resolution of the anomaly map.
            device: Device to run the model on (e.g., 'cuda').
        """

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize region proposal model
        if isinstance(region_proposal_model, str):
            if region_proposal_model == "GroundingDINO":
                self.anomaly_region_generator: DetectorInterface = dino.DINO(
                    model_id=region_proposal_model_config.get("model_id", "IDEA-Research/grounding-dino-tiny"),
                    device=device,
                )
        elif isinstance(region_proposal_model, DetectorInterface):
            self.anomaly_region_generator = region_proposal_model
        else:
            raise ValueError("region_proposal_model must be a string or a DetectorInterface instance.")

        # Initialize region refiner model
        if isinstance(region_refiner_model, str):
            if region_refiner_model == "SAM":
                self.anomaly_region_refiner: SegmentorInterface = sam.SamSegmentorHF(
                    model_id=region_refiner_model_config.get("model_id", "facebook/sam-vit-base"),
                    device=device,
                )
        elif isinstance(region_refiner_model, SegmentorInterface):
            self.anomaly_region_refiner = region_refiner_model
        else:
            raise ValueError("region_refiner_model must be a string or a SegmentorInterface instance.")

        if isinstance(saliency_model, str):
            if saliency_model == "ModelINet":
                self.feature_extractor: SaliencyModelInterface = widenet.ModelINet(
                    backbone_name=saliency_model_config.get("model_id", "wide_resnet50_2"),
                    resize_longest=saliency_model_config.get("resize_longest", 1024),
                    device=device,
                )
        elif isinstance(saliency_model, SaliencyModelInterface):
            self.feature_extractor = saliency_model
        else:
            raise ValueError("saliency_model must be a string or a SaliencyModelInterface instance.")

        # Set parameters
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Set prompts parameters
        if kwargs.get("prompt_pairs"):
            try:
                self.set_ensemble_prompts([PromptPair(target=p[0], background=p[1]) for p in kwargs["prompt_pairs"]])
            except Exception as e:
                raise ValueError(f"Invalid prompt pairs: {kwargs['prompt_pairs']}") from e
        else:
            self._prompt_pairs: Sequence[PromptPair] = []
        if kwargs.get("object_prompt"):
            try:
                self._obj_prompt: ObjectPrompt | None = ObjectPrompt(**kwargs["object_prompt"])
            except Exception as e:
                raise ValueError(f"Invalid object prompt: {kwargs['object_prompt']}") from e
        else:
            self._obj_prompt: ObjectPrompt | None = None

        # debug flag
        self.debug = debug

        # nms
        self.nms_iou_thr: float = kwargs.get("nms_iou_thr", 0.5)  # type: ignore

        # saliency
        self.scale: float = kwargs.get("scale", 3.0)  # type: ignore

    def predict(self, input_image: str | Image.Image | np.ndarray, **kwargs: dict[str, Any]) -> Annotation:
        """
        Predict the anomaly map for the given image.

        Args:
            input_image: The input image for prediction.
            **kwargs: Additional arguments for the prediction.
        Returns:
            Annotation: The predicted anomaly map and any additional information.
        """
        if self._obj_prompt is None or len(self._prompt_pairs) == 0:
            raise ValueError("Object prompt and defect prompts must be set before calling predict.")

        image = self.load_image(input_image)

        # object segmentation
        obj_masks, obj_scores, obj_area, _ = self.ensemble_text_guided_mask_proposal(
            image=image,
            object_prompt=self._obj_prompt,
            box_thr=self.box_threshold,
            text_thr=self.text_threshold,
            nms_iou_thr=self.nms_iou_thr,
            debug_path=kwargs.get("debug_path_1", "1_object.png") if self.debug else "",  # type: ignore
        )
        if len(obj_masks) > 0:
            defect_max_area = obj_area * self._obj_prompt.anomaly_area_ratio
        else:
            defect_max_area = self._obj_prompt.object_max_area * self._obj_prompt.anomaly_area_ratio
        defect_min_area = 0.0

        # object-level TGMP
        defect_masks, defect_scores, _, defect_labels = self.ensemble_text_guided_mask_proposal(
            image,
            defect_prompts=self._prompt_pairs,
            box_thr=self.box_threshold,
            text_thr=self.text_threshold,
            area_min=defect_min_area,
            area_max=defect_max_area,
            debug_path=kwargs.get("debug_path_2", "2_defect.png") if self.debug else "",  # type: ignore
        )

        # saliency map (self-similarity)
        if self._obj_prompt.count > 1 and len(obj_masks) > 1:
            self_similarity_map = self.multiple_object_similarity(image, object_masks=obj_masks)
        else:
            self_similarity_map = self.self_similarity_calculation(image)

        if self.debug and (debug_path_3 := kwargs.get("debug_path_3", "3_self_similarity.png")):
            # normalize self_similarity_map to [0,1]
            min_val, max_val = self_similarity_map.min(), self_similarity_map.max()
            norm_map = (self_similarity_map - min_val) / (max_val - min_val + 1e-8)

            # convert to 8-bit and apply blue-to-red colormap
            heatmap_uint8 = (norm_map * 255).astype(np.uint8)
            colored_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            # save the colored heatmap
            cv2.imwrite(debug_path_3, colored_map)  # type: ignore

        # rescoring
        rescored_defect_scores = self.rescore(
            defect_masks=defect_masks,
            defect_scores=defect_scores,
            similarity_map=self_similarity_map,
            scale=self.scale,
        )

        # top-k thresholding
        anomaly_map = self.top_k_filtering(
            image,
            defect_masks=defect_masks,
            defect_scores=rescored_defect_scores,
            k=self._obj_prompt.max_anomalies,
        )

        result = Annotation(
            image=image,
            damaged=True if len(rescored_defect_scores) == 0 or max(rescored_defect_scores) > 0.0 else False,
            bboxes=[mask_to_box(mask) for mask in defect_masks],
            scores=rescored_defect_scores,
            bboxes_labels=defect_labels,
            mask=anomaly_map,
        )

        if self.debug and (debug_path_4 := kwargs.get("debug_path_4", "4_anomaly_map.png")):
            draw_annotation(image, result, show_mask=True, show_boxes=True, save_path=debug_path_4)  # type: ignore

        return result

    def set_ensemble_prompts(self, prompts: Sequence[PromptPair]) -> None:
        """Register the prompts that drive Grounding DINO.

        Args:
            prompts: Sequence of (defect, background) pairs.
        """
        self._prompt_pairs = [
            PromptPair(target=p.target.strip().lower().rstrip("."), background=p.background.strip().lower().rstrip("."))
            for p in prompts
        ]

    def set_object_prompt(self, prompt: ObjectPrompt) -> None:
        """Set constraints derived from domain knowledge."""
        self._obj_prompt = prompt

    def ensemble_text_guided_mask_proposal(
        self,
        image: Image.Image,
        *,
        defect_prompts: Sequence[PromptPair] | None = None,
        object_prompt: ObjectPrompt | None = None,
        box_thr: float,
        text_thr: float,
        area_min: float | None = None,
        area_max: float | None = None,
        nms_iou_thr: float = 0.5,
        debug_path: str = "",
    ) -> tuple[list[np.ndarray], list[float], float, list[str]]:
        """Run detector → filter → segmentor for all prompts and fuse results.

        Args:
            image:              The *RGB* image as a :class:`PIL.Image.Image`.
            prompts:            Positive/negative prompt pairs.
            box_thr:            Confidence threshold for detector boxes.
            text_thr:           Confidence threshold for per-token logits.
            area_min:           Normalised lower bound on box area (0-1).
            area_max:           Normalised upper bound on box area (0-1).
            nms_iou_thr:        IoU threshold for Non-Max Suppression.

        Returns:
            masks:    Boolean masks at full resolution, one per kept box.
            scores:   Max-token score per mask (float list).
            max_area: Largest *unnormalised* CXCYWH area among surviving boxes.
            labels:   Text labels for each mask.
        """

        w, h = image.width, image.height
        self.anomaly_region_refiner.set_image(image)  # encode once for speed

        all_boxes: list[torch.Tensor] = []
        all_scores: list[float] = []
        all_phrases: list[str] = []
        max_box_area = 0.0

        if object_prompt is not None:
            prompts: Sequence[PromptPair] = [PromptPair(target=object_prompt.name, background="")]
            area_max = object_prompt.object_max_area
            area_min = object_prompt.object_min_area
        elif defect_prompts is not None:
            prompts = defect_prompts
            area_min = area_min if area_min is not None else 0.0
            area_max = area_max if area_max is not None else 0.1
        else:
            raise ValueError("No prompts provided for ensemble text-guided mask proposal.")

        if area_min is None or area_max is None:
            raise ValueError("area_min and area_max must be provided.")

        for pair in prompts:
            # 1️⃣ Run the open-vocab detector
            det_boxes, det_scores, det_phrases = self.anomaly_region_generator.predict(
                image=image,
                text_prompts=[pair.target],
                box_threshold=box_thr,
                text_threshold=text_thr,
            )
            if det_boxes.size == 0:
                continue

            boxes_t = torch.as_tensor(det_boxes, dtype=torch.float32)  # already XYXY abs
            scores_t = torch.as_tensor(det_scores, dtype=torch.float32)
            areas_norm = box_area_xyxy(boxes_t) / (w * h)

            keep = (areas_norm >= area_min) & (areas_norm <= area_max)
            if pair.background:
                bg_mask = torch.tensor(
                    [pair.background in p.lower() for p in det_phrases],
                    dtype=torch.bool,
                )
                keep &= ~bg_mask
            if keep.sum() == 0:
                continue

            boxes_t = boxes_t[keep]
            scores_t = scores_t[keep]

            all_boxes.append(boxes_t)
            all_scores.extend(scores_t.tolist())
            all_phrases.extend(compress(det_phrases, keep))
            max_box_area = max(max_box_area, box_area_xyxy(boxes_t).max().item())

        if not all_boxes:
            # empty_mask = np.zeros((h, w), dtype=bool)
            return [], [], 1.0, []

        # 3️⃣ Stack, convert to XYXY abs, NMS
        boxes_xyxy = torch.cat(all_boxes, dim=0)
        scores_cat = torch.as_tensor(all_scores)
        keep_idx: torch.IntTensor = nms_xyxy(boxes_xyxy, scores_cat, nms_iou_thr)

        boxes_xyxy = boxes_xyxy[keep_idx]
        scores_cat = scores_cat[keep_idx]
        pred_labels = [all_phrases[i] for i in keep_idx.cpu().tolist()]

        # 4️⃣ Run the segmentor once for all kept boxes
        masks = self.anomaly_region_refiner.predict(boxes_xyxy.cpu().numpy(), multimask_output=True)
        scores_out = scores_cat.tolist()

        if len(debug_path):
            from aircraft_anomaly_detection.viz_utils import draw_annotation  # local helper

            ann = Annotation(
                bboxes=boxes_xyxy.cpu().numpy().tolist(),  # type: ignore
                scores=scores_out,
                bboxes_labels=pred_labels,
                mask=np.any(masks, axis=0).astype(np.uint8),
            )

            _ = draw_annotation(image, ann, show_mask=True, show_boxes=True, save_path=debug_path)

        return list(masks), scores_out, max_box_area / (w * h), pred_labels

    def self_similarity_calculation(
        self,
        image: Image.Image,
    ) -> np.ndarray:
        """Calculate self-similarity for the given image and object masks.

        Args:
            image: The input image.
            object_masks: The object masks.

        Returns:
            The self-similarity map.
        """
        resize_image = image.resize((256, 256))
        features = self.feature_extractor.generate_saliency_map(resize_image)

        C, H, W = features.shape
        flattened_feats = features.view(C, H * W)

        feat_sim = flattened_feats.T @ flattened_feats  # (4096, 4096) cosine-sim
        feat_sim = 0.5 * (1 - feat_sim)  # convert to *distance* maps [−1…1] → [1…0]

        topk_vals, _ = torch.topk(feat_sim, k=400, dim=1, largest=True, sorted=False)
        heat_map = topk_vals.mean(dim=1).view(H, W).cpu().numpy()

        mask_anomaly_scores = cv2.resize(heat_map, (image.width, image.height))
        return mask_anomaly_scores

    def multiple_object_similarity(
        self,
        image: Image.Image,
        object_masks: list[np.ndarray],
    ) -> np.ndarray:
        """
        Compute an anomaly heatmap by measuring feature-based similarity across objects.

        This method performs the following steps:
            1. Resize the input image to the fixed size expected by the saliency model.
            2. Generate a multi-channel feature map from the resized image.
            3. Resize each binary object mask to the feature-map resolution.
            4. For each object:
                a. Extract feature vectors inside the target mask.
                b. Extract feature vectors from all other masks.
                c. Compute cosine similarity between target and other features.
                d. Convert similarity to anomaly scores (lower similarity → higher anomaly).
                e. Scatter per-pixel anomaly scores back into a 2D map.
            5. Fuse per-object anomaly maps by taking a pixel-wise maximum.
            6. Upsample the fused map to the original image resolution.

        Args:
            image (PIL.Image.Image): Input RGB image.
            object_masks (list[np.ndarray]): List of binary masks for each detected object.

        Returns:
            np.ndarray: A 2D anomaly map of shape (image.height, image.width),
                        where higher values indicate greater anomaly likelihood.
        """

        # Resize the input image to the fixed size expected by the saliency model
        resize_image = image.resize((1024, 1024))
        # Compute the feature/saliency map for the resized image
        feats = self.feature_extractor.generate_saliency_map(resize_image)

        # Unpack feature dimensions: C channels, H×W spatial size
        _, H, W = feats.shape
        feat_size = (H, W)  # used for resizing masks to feature-map resolution

        # Resize each binary object mask to the feature-map resolution
        resized_obj_masks = []
        for obj_mask in object_masks:
            # cast to int for interpolation
            new_obj_mask = obj_mask.astype(np.int32)
            # nearest‐neighbor keeps the mask binary
            resized_obj_mask = cv2.resize(new_obj_mask, feat_size, interpolation=cv2.INTER_NEAREST)
            resized_obj_masks.append(resized_obj_mask)

        # For each object, extract feature vectors and compute inter‐object similarity
        mask_anomaly_scores = []
        for indx in range(len(resized_obj_masks)):
            # all masks except the current one
            other_obj_masks = resized_obj_masks[:indx] + resized_obj_masks[indx + 1 :]
            # the current target object mask
            target_obj_mask = resized_obj_masks[indx]

            # Extract feature vectors & pixel locations for target vs. other masks
            #    - target_mask_feats: (N1, C) feature vectors within the target mask
            #    - target_feat_locations: (N1, 2) (row,col) of each target pixel
            #    - other_mask_feats: (N2, C) feature vectors from all other masks
            target_mask_feats, target_feat_locations, other_mask_feats = self.region_feature_extraction(
                feats,
                target_obj_mask,
                other_obj_masks,
            )
            # subsequent similarity→anomaly scoring follows...

            # Compute cosine similarity between target‐mask features and other‐mask features
            # target_mask_feats: (N1, C), other_mask_feats: (N2, C)
            sim_matrix = target_mask_feats @ other_mask_feats.T  # (N1, N2)

            # For each target pixel, pick the maximum similarity across all other objects
            max_sim_vals, _ = torch.max(sim_matrix, dim=1)  # (N1,)
            # Convert similarity to anomaly score (lower similarity ⇒ higher anomaly)
            anomaly_scores = 0.5 * (1.0 - max_sim_vals)  # (N1,)
            anomaly_scores_np = anomaly_scores.cpu().numpy()

            # Scatter the per‐pixel anomaly scores back into a H_feat×W_feat map
            H_feat, W_feat = feat_size
            obj_map = np.zeros((H_feat, W_feat), dtype=float)
            for (row, col), score in zip(target_feat_locations, anomaly_scores_np):
                obj_map[row, col] = score
            mask_anomaly_scores.append(obj_map)

        # Fuse all object maps: take pixel‐wise maximum across objects
        stacked = np.stack(mask_anomaly_scores, axis=0)  # (num_objs, H_feat, W_feat)
        fused_map = np.max(stacked, axis=0)  # (H_feat, W_feat)

        # Upsample anomaly map to original image size
        orig_w, orig_h = image.width, image.height
        final_map = cv2.resize(fused_map, (orig_w, orig_h))

        return final_map  # type: ignore

    def region_feature_extraction(
        self,
        features: torch.Tensor,
        target_obj_mask: np.ndarray,
        other_obj_masks: list[np.ndarray],
    ) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
        """Extract feature vectors for a target mask and its complement regions.

        Given a feature map and a binary mask of one object region, zero out
        those features and collect
          1. The features and positions of the masked pixels.
          2. The features of all pixels belonging to the other object masks.

        Args:
            features: A tensor of shape (C, H, W)
            one_object_mask: A binary mask of shape (H, W) for the target object.
            other_object_masks: A list of binary masks, each of shape (H, W),
                representing other object regions.

        Returns:
            one_mask_feature: Tensor of shape (N1, C), where N1 is the number of
                pixels in `one_object_mask`. Each row is the C-dim feature vector
                at that location.
            one_feature_locations: NumPy array of shape (N1, 2), giving (row, col)
                coordinates of each pixel in the target mask.
            other_mask_features: Tensor of shape (N2, C), concatenated feature
                vectors of all pixels in `other_object_masks`, where N2 is the
                total count over all other masks.
        """
        C, _, W = features.shape
        feat = features.clone()

        # Flatten spatial dims for easy indexing
        feat_flat = feat.view(C, -1)  # shape: (C, H*W)
        mask_flat = target_obj_mask.ravel()  # shape: (H*W,)

        # 1) Extract features & positions for the target mask
        non_zero_idx = np.nonzero(mask_flat)[0]  # indices of mask pixels
        target_mask_feats = feat_flat[:, non_zero_idx].T  # shape: (N1, C)
        # Convert flat indices to (row, col)
        target_feat_locations = np.vstack((non_zero_idx // W, non_zero_idx % W)).T  # shape: (N1, 2)

        # Zero out those features in the feature map
        feat_flat[:, non_zero_idx] = 0

        # 2) Extract features for all other masks
        other_features = []
        for mask in other_obj_masks:
            mask_flat = mask.ravel().astype(bool)
            non_zero_idx = np.nonzero(mask_flat)[0]
            if non_zero_idx.size > 0:
                other_features.append(feat_flat[:, non_zero_idx])

        if other_features:
            other_mask_feats = torch.cat(other_features, dim=1).T  # (N2, C)
        else:
            other_mask_feats = torch.empty((0, C), device=features.device)

        return target_mask_feats, target_feat_locations, other_mask_feats

    def rescore(
        self,
        defect_masks: Sequence[np.ndarray],
        defect_scores: Sequence[float],
        similarity_map: np.ndarray,
        scale: float = 3.0,
    ) -> list[float]:
        """Rescore defect logits based on the local similarity map.

        For each defect mask, compute the mean similarity within the mask region.
        Convert that into an anomaly factor via an exponential warp
        and multiply it with the original logit.

        Args:
            defect_masks: List of boolean masks (HxW) for each defect region.
            defect_scores: List of original scores for each defect mask.
            similarity_map: 2D array (HxW) of similarity values in [0, 1].
            scale:       Exponential scaling factor (default: 3.0).

        Returns:
            A tuple of
            - the original `defect_masks`,
            - list of rescored logits (float32).
        """
        if len(defect_masks) != len(defect_scores):
            raise ValueError(f"Expected {len(defect_masks)} logits, got {len(defect_scores)}.")

        rescored: list[float] = []
        for mask, score in zip(defect_masks, defect_scores):
            # Ensure mask is boolean
            mask_bool = mask.astype(bool)
            region_vals = similarity_map[mask_bool]

            if region_vals.size > 0:
                anomaly_factor = float(np.exp(scale * region_vals.mean()))
            else:
                # No pixels under mask → neutral similarity
                anomaly_factor = 1.0

            # Higher anomaly when similarity is low
            rescored.append(score * anomaly_factor)

        return rescored

    def top_k_filtering(
        self,
        image: Image.Image,
        defect_masks: Sequence[np.ndarray],
        defect_scores: Sequence[float],
        k: int,
    ) -> np.ndarray:
        """
        Select the top-k defect masks by score and fuse them into a single anomaly map.

        Args:
            defect_masks: Sequence of HxW boolean masks for each defect region.
            defect_scores: Sequence of floats corresponding to each mask's score.
            k: Number of highest-scoring masks to keep. If None, uses self.k_mask.

        Returns:
            An H_outxW_out anomaly map (float32) where H_out=W_out=self.out_size.
        """
        # Determine how many masks to keep
        if k <= 0:
            raise ValueError(f"Expected k>0, got {k}")

        # Validate inputs
        if len(defect_masks) != len(defect_scores):
            raise ValueError(f"Number of masks ({len(defect_masks)}) and scores ({len(defect_scores)}) must match.")

        # return if empty score array
        if len(defect_masks) == 0:
            return np.zeros((image.height, image.width), dtype=np.float32)

        # Pick top‐k indices
        scores_arr = np.array(defect_scores, dtype=float)
        topk_idx = np.argsort(scores_arr)[-k:]
        topk_masks = [defect_masks[i] for i in topk_idx]
        topk_scores = scores_arr[topk_idx]

        # Initialize fusion maps
        h, w = topk_masks[0].shape
        anomaly_map = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        # Accumulate weighted mask scores
        for mask, score_val in zip(topk_masks, topk_scores):
            mask_f = mask.astype(float)
            anomaly_map += mask_f * score_val
            weight_map += mask_f

        # Normalize by coverage
        nonzero = weight_map > 0
        anomaly_map[nonzero] /= weight_map[nonzero]

        return cv2.resize(anomaly_map, image.size, interpolation=cv2.INTER_LINEAR)
