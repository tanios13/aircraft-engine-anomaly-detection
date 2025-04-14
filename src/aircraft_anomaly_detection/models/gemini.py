import google.generativeai as genai
import numpy as np
from PIL import Image


class GeminiVision:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """Initializes the GeminiVision class with an API key and model name."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def _load_image(self, image_input: str | np.ndarray | Image.Image) -> Image.Image:
        """Loads an image from various input types into a PIL Image."""
        if isinstance(image_input, str):
            try:
                img = Image.open(image_input)
                return img.convert("RGB")
            except FileNotFoundError:
                raise ValueError(f"Unable to load image from path: {image_input}")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            try:
                img = Image.fromarray(image_input)
                return img.convert("RGB")
            except ValueError:
                raise ValueError("Numpy array could not be converted to PIL Image. Ensure correct shape and dtype.")
        else:
            raise ValueError("image_input must be a file path (str), a numpy.ndarray, or a PIL.Image.Image.")

    def generate_with_image_and_prompt(self, image_input, prompt: str):
        """Sends an image and text prompt to the Gemini model and returns the response."""
        image = self._load_image(image_input)
        contents = [image, prompt]
        try:
            response = self.model.generate_content(contents)
            return response.text
        except Exception as e:
            return f"An error occurred: {e}"
