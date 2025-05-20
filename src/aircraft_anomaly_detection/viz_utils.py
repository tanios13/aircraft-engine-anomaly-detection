from __future__ import annotations

import random
from collections.abc import Iterable
from os import PathLike
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from numpy.ma import MaskedArray
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

from .interfaces import Annotation


class CocoVisualizer:
    """
    Quick-and-dirty COCO image & annotation viewer.

    Example
    -------
    >>> viz = CocoVisualizer("data/annotations.json", "data/images")
    >>> viz.show()                 # random image
    >>> viz.show(img_id=17)        # specific image
    """

    def __init__(
        self,
        coco_json: PathLike[str] | Path | COCO,
        image_dir: PathLike[str] | Path,
        alpha: float = 0.35,
        linewidth: float = 2.0,
    ) -> None:
        self.coco = coco_json if isinstance(coco_json, COCO) else COCO(str(coco_json))
        self.image_dir = Path(image_dir)
        self.alpha = alpha
        self.linewidth = linewidth
        self._cat_color = self._build_color_map()

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def show(
        self,
        img_id: int | None = None,
        show_masks: bool = True,
        show_bbox: bool = True,
        show_label: bool = True,
        ax=None,
    ):
        """
        Display a COCO image with its annotations.

        Parameters
        ----------
        img_id : int, optional
            COCO image id. If *None*, a random image is sampled.
        show_masks : bool
            Fill polygons / decoded RLE masks.
        show_bbox : bool
            Draw bounding rectangles.
        show_label : bool
            Write category name at top‑left of each object.
        ax : matplotlib axis, optional
            If provided, draw in this axis; else create a new figure.
        """
        if img_id is None:
            img_id = random.choice(self.coco.getImgIds())

        # --- load image --------------------------------------------------- #
        info = self.coco.loadImgs(img_id)[0]
        img_path = self.image_dir / info["file_name"]
        img = np.asarray(Image.open(img_path).convert("RGB"))

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{info['file_name']} (id={img_id})")

        # --- draw each annotation ---------------------------------------- #
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        for ann in self.coco.loadAnns(ann_ids):
            color = self._cat_color[ann["category_id"]]

            # bounding‑box
            if show_bbox and "bbox" in ann:
                x, y, w, h = ann["bbox"]
                ax.add_patch(
                    mpatches.Rectangle(
                        (x, y),
                        w,
                        h,
                        edgecolor=color,
                        facecolor="none",
                        linewidth=self.linewidth,
                    )
                )

            # mask / polygon
            if show_masks and "segmentation" in ann:
                self._draw_mask(ax, ann, color)

            # label
            if show_label:
                cat_name = self.coco.loadCats([ann["category_id"]])[0]["name"]
                x, y = ann["bbox"][:2]
                ax.text(
                    x,
                    y - 2,
                    cat_name,
                    color="white",
                    fontsize=9,
                    bbox=dict(facecolor=color, alpha=0.6, pad=1, edgecolor="none"),
                )

        return ax

    def _build_color_map(self) -> dict[int, tuple[float, float, float, float]]:
        """Assign a distinct (repeatable) rgba color per category id."""
        import matplotlib.cm as cm

        cat_ids = self.coco.getCatIds()
        tab = cm.get_cmap("tab20", len(cat_ids))
        return {cid: tab(i) for i, cid in enumerate(sorted(cat_ids))}

    def _draw_mask(self, ax, ann: dict, color):
        """Draw polygon or RLE mask with transparency."""
        seg = ann["segmentation"]
        if isinstance(seg, list):  # polygon
            for poly in seg:
                poly = np.array(poly).reshape((-1, 2))
                patch = mpatches.PathPatch(
                    mpath.Path(poly),
                    facecolor=color,
                    edgecolor="none",
                    alpha=self.alpha,
                )
                ax.add_patch(patch)
        else:  # RLE
            m = mask_utils.decode(seg)
            ax.imshow(np.ma.MaskedArray(m, m == 0), alpha=self.alpha, cmap="inferno")


def draw_annotation(
    image: Image.Image,
    annotation: Annotation,
    *,
    show_boxes: bool = True,
    show_mask: bool = True,
    cmap: Iterable[str] | None = None,
    ax: plt.Axes | None = None,
    alpha: float = 0.4,
    linewidth: float = 2.0,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """
    Overlay an Annotation on a PIL image.
    Overlay an Annotation on a PIL image.

    Parameters
    ----------
    image        : PIL.Image
    annotation   : Annotation instance
    show_boxes   : draw bounding-boxes if True
    show_mask    : draw segmentation mask if True
    cmap         : colour cycle for successive boxes
    ax           : existing matplotlib axis (created if None)
    alpha        : mask transparency
    linewidth    : box edge width
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    ax.imshow(image)
    ax.axis("off")

    if show_boxes and annotation.bboxes:
        cmap = cmap or plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, (bbox, label) in enumerate(zip(annotation.bboxes, annotation.bboxes_labels)):
            x1, y1, x2, y2 = bbox
            colour = cmap[i % len(cmap)]  # type: ignore
            ax.add_patch(
                mpatches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    edgecolor=colour,
                    facecolor="none",
                    linewidth=linewidth,
                )
            )
            ax.text(
                x1,
                y1 - 2,
                label,
                color="white",
                fontsize=9,
                bbox=dict(facecolor=colour, alpha=0.6, pad=1, edgecolor="none"),
            )
    if show_mask and annotation.mask is not None:
        ax.imshow(
            MaskedArray(annotation.mask, annotation.mask == 0),  # type: ignore
            alpha=alpha,
            cmap="RdBu",
        )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return ax


def visualize_mask_overlap_with_image(
    image: Image.Image,
    mask1: np.ndarray,
    mask2: np.ndarray,
    title: str = "Mask Overlap",
    save_path: str | Path | None = None,
) -> None:
    """
    Show a PIL image with overlaid ground truth and predicted masks,
    Show a PIL image with overlaid ground truth and predicted masks,
    and display a legend explaining the colors.
    """
    image_np = np.array(image.convert("RGB")) / 255.0  # Normalize to [0, 1]
    height, width = mask1.shape

    # Create overlay
    overlay = np.zeros((height, width, 3), dtype=np.float32)

    # Color codes
    red = np.array([1, 0, 0])  # Pred only
    green = np.array([0, 1, 0])  # Ground truth only
    yellow = np.array([1, 1, 0])  # Intersection

    # Apply colors
    only_pred = (mask1 == 0) & (mask2 == 1)
    only_gt = (mask1 == 1) & (mask2 == 0)
    intersection = (mask1 == 1) & (mask2 == 1)

    overlay[only_pred] = red
    overlay[only_gt] = green
    overlay[intersection] = yellow

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(image_np)
    plt.imshow(overlay, alpha=0.4)
    plt.title(title)
    plt.axis("off")

    # Legend
    legend_patches = [
        Patch(color="red", label="Predicted only"),
        Patch(color="green", label="Ground truth only"),
        Patch(color="yellow", label="Intersection"),
    ]
    plt.legend(handles=legend_patches, loc="upper right")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
