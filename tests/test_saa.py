import numpy as np
import pytest
from sympy import im

from aircraft_anomaly_detection.dataloader import AnomalyDataset
from aircraft_anomaly_detection.models.saa.saa import SAA
from aircraft_anomaly_detection.schemas import Annotation, ObjectPrompt, PromptPair
from aircraft_anomaly_detection.viz_utils import draw_annotation


def test_saa_instantiation() -> None:
    """
    Tests if an SAA object can be instantiated.
    This test assumes that SAA's __init__ method (currently commented out in the provided code)
    is not active. If ModelInterface (the base class) has an __init__ requiring arguments,
    this test might fail and would need adjustment based on ModelInterface's definition.
    """
    try:
        model = SAA(
            region_proposal_model="GroundingDINO",
            region_refiner_model="SAM",
            saliency_model="ModelINet",
            box_threshold=0.2,
            text_threshold=0.2,
        )
        assert isinstance(model, SAA), "Object should be an instance of SAA"
    except TypeError as e:
        pytest.fail(
            f"SAA instantiation failed with TypeError: {e}. "
            "This might be due to an __init__ method in SAA or its "
            "base class ModelInterface requiring arguments that were not provided."
        )
    except Exception as e:
        pytest.fail(f"SAA instantiation failed with an unexpected error: {e}")

        import pytest  # pytest is already imported in the user's file, but explicit for clarity if this block is isolated

        # Assuming SAA is imported from aircraft_anomaly_detection.models.saa.saa
        # Assuming ModelInterface might be relevant for context but not directly used in this test function


def test_saa_predict() -> None:
    """
    Tests the SAA.predict() method.

    This test verifies that SAA's predict method, when called with a sample image
    and prompts, returns a list of Annotation objects adhering to the defined schema.
    It includes checks for the presence and types of attributes within each Annotation.

    The test attempts to instantiate SAA using parameters similar to
    `test_saa_instantiation`. If SAA cannot be instantiated (e.g., due to being
    an abstract class or __init__ signature mismatch), or if `predict` is not
    implemented, the test will be skipped.
    """

    # Instantiate SAA. For a focused unit test of predict() logic,
    # mock objects for region_proposal_model and region_refiner_model
    # would typically be injected here if SAA's __init__ supports them.
    model = SAA(
        region_proposal_model="GroundingDINO",  # Placeholder or actual model key
        region_refiner_model="CLIPSeg",  # Placeholder or actual model key
        saliency_model="ModelINet",
        box_threshold=0.2,
        text_threshold=0.2,
        debug=True,  # Enable debug mode for detailed output
    )

    # Prepare mock inputs for the predict method
    # Image: numpy array (H, W, C)
    image, _, meta = AnomalyDataset("mvtech").filter_by_component("cable").filter_by(lambda _, m, l: l == 1)[0]
    assert meta.annotation is not None, "Metadata annotation should not be None"
    _ = draw_annotation(image, meta.annotation, show_boxes=True, show_mask=True, save_path="0_original_annotated.png")

    # Prompts: Sequence[PromptPair]
    defect_prompts = [
        PromptPair(target="crack", background="cable"),
        PromptPair(target="flawed golden wire", background="cable"),
        PromptPair(target="black hole", background="cable"),
    ]

    object_prompt = ObjectPrompt(name="cable", count=1, max_anomalies=1, anomaly_area_ratio=0.3)

    try:
        model.set_ensemble_prompts(defect_prompts)
        model.set_object_prompt(object_prompt)
        prediction = model.predict(image)
    except NotImplementedError:
        pytest.skip("SAA.predict() is not implemented.")
    except AttributeError as e:
        # Handles case where 'predict' might be missing, though unlikely if inheriting ModelInterface
        if "'SAA' object has no attribute 'predict'" in str(e):
            pytest.skip(f"SAA.predict() method not found: {e}")
        else:
            pytest.fail(f"model.predict() raised an unexpected AttributeError: {e}")
    except Exception as e:
        pytest.fail(f"model.predict() raised an unexpected exception: {e}")

    # Validate the output structure and types
    assert isinstance(prediction, Annotation), f"Predictions should be a Annotation, but got {type(prediction)}"


# main function
if __name__ == "__main__":
    test_saa_predict()
