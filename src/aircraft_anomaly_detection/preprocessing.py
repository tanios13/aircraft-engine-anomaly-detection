from collections.abc import Callable
from multiprocessing import process

import cv2
import numpy as np
from PIL import Image
from torch.functional import F
from transformers import AutoProcessor, CLIPSegForImageSegmentation


class CLIPBackgroundRemover:
    def __init__(
        self,
        background_text: str = "background",
        foreground_text: str = "metal component or part",
        threshold: float = 0.5,
        background_color: list = [255, 255, 255],
    ):
        """
        Initializes the CLIPBackgroundRemover.

        Args:
            background_text (str): Text description of the background.
            foreground_text (str): Text description of the foreground (object of interest).
            treshold (float): Threshold to binarize the segmentation mask.
            background_color (List): Color to use for background replacement (currently unused).
        """
        self.threshold = threshold
        self.background_text = background_text
        self.foreground_text = foreground_text
        self.background_color = background_color

        self.processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    def __call__(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a segmentation mask for the input image.

        Args:
            image (PIL.Image): Input image to segment.

        Returns:
            np.ndarray: A binary mask where 1 represents foreground and 0 background.
        """
        # Background segmentation
        texts = [self.background_text, self.foreground_text]
        image = image.convert("RGB")
        inputs = self.processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")
        logits = self.model(**inputs).logits
        mask = F.softmax(logits, dim=0).detach().cpu().numpy()[0]
        mask = (mask > self.threshold).astype(int)
        resized_mask = cv2.resize(mask, image.size, interpolation=cv2.INTER_NEAREST)

        image_np = np.array(image)
        output_np = np.where(resized_mask[..., None] == 1, self.background_color, image_np)
        output_image = Image.fromarray(output_np.astype(np.uint8), mode="RGB")
        return output_image, resized_mask


class PreProcessor:
    def __init__(self, options: list = []):
        self.options = options
        self.func_map = {
            "clahe_rgb": clahe_rgb,
            "clahe_gs": clahe_gs,
            "lightnorm": lighting_normalization,
            "sobel": sobel_filter,
            "gabor": gabor_filter,
        }

    def __call__(self, image: Image.Image) -> Image.Image:
        mod_image = image
        for opt in self.options:
            if opt in self.func_map:
                mod_image = self.func_map[opt](mod_image)
        return mod_image


# Helpers ######################


def clahe_gs(image: Image.Image) -> Image.Image:
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(gray)

    # Convert back to PIL
    return Image.fromarray(enhanced)


def clahe_rgb(image: Image.Image) -> Image.Image:
    img_np = np.array(image)
    r, g, b = cv2.split(img_np)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))

    # Apply CLAHE to each channel
    r_clahe = clahe.apply(r)
    g_clahe = clahe.apply(g)
    b_clahe = clahe.apply(b)

    # Merge back into an RGB image
    img_clahe = cv2.merge((r_clahe, g_clahe, b_clahe))

    return Image.fromarray(img_clahe)


def lighting_normalization(image: Image.Image) -> Image.Image:
    img_np = np.array(image)

    # Convert to grayscale and float for precision
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

    blurred = cv2.GaussianBlur(gray, (55, 55), 0)
    norm = gray - blurred

    norm += 70

    # Clip to 0-255 range and convert back to uint8
    norm = np.clip(norm, 0, 255).astype(np.uint8)

    return Image.fromarray(norm)


def sobel_filter(image: Image.Image) -> Image.Image:
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Apply Sobel operator
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)

    # Normalize to 0–255 and convert to uint8
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    sobel_img = magnitude.astype(np.uint8)

    return Image.fromarray(sobel_img)


def gabor_filter(image: Image.Image, theta: float = 0) -> Image.Image:
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

    kernel = cv2.getGaborKernel(
        ksize=(21, 21),
        sigma=4.0,
        theta=theta,
        lambd=10.0,
        gamma=0.5,
        psi=0,
        ktype=cv2.CV_32F,
    )

    # Use float output to preserve negative values
    filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)

    # Normalize to 0–255 and convert to uint8 for display
    filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
    return Image.fromarray(filtered.astype(np.uint8))
