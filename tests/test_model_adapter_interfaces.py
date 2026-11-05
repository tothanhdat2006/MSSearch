"""
Self-check unit test for VLM Model Adapters (ModelGlobalLocal, ModelInternvl, ModelQwenVL).
Verifies:
1. Interface contract: get_confidence_value, batch_get_confidence_values, generate_response, batch_generate_response exist.
2. Root handling: is_root_only(None), include_root(None), process_nodes_to_image_list(None, img) do not crash.
3. No silent fallback: importing ModelGlobalLocal raises ImportError if llava is missing or loads ModelGlobalLocal.
"""

import sys
import os
from unittest.mock import MagicMock
from PIL import Image

# Ensure repo root is in sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Mock torch and heavy ML libraries if not available in local test environment
for mod_name in [
    "torch", "torch.nn", "torch.nn.functional", "torchvision", "torchvision.transforms",
    "torchvision.transforms.functional", "transformers", "transformers.models",
    "spacy", "sentence_transformers", "cv2", "skimage", "skimage.segmentation",
    "skimage.measure", "skimage.graph", "sklearn", "sklearn.cluster",
    "sklearn.metrics", "sklearn.preprocessing", "networkx"
]:
    if mod_name not in sys.modules:
        m = MagicMock()
        m.__spec__ = MagicMock()
        m.__path__ = []
        sys.modules[mod_name] = m


def test_root_handling_logic():
    from mssearch.models.modeling_internvl import ModelInternvl
    from mssearch.models.modeling_llava import ModelGlobalLocal

    # Test unbound methods directly without full model weight loading
    dummy_img = Image.new("RGB", (100, 100), color="white")

    class DummyInstance:
        background_color = (255, 255, 255)
        input_size = (384, 384)
        is_root_only = ModelGlobalLocal.is_root_only
        include_root = ModelGlobalLocal.include_root
        process_nodes_to_image_list = ModelGlobalLocal.process_nodes_to_image_list

    inst = DummyInstance()

    # 1. Test None nodes
    assert inst.is_root_only(None) is False, "is_root_only(None) must be False"
    assert inst.include_root(None) is False, "include_root(None) must be False"
    img_list = inst.process_nodes_to_image_list(None, dummy_img)
    assert len(img_list) == 1, f"Expected 1 image for nodes=None, got {len(img_list)}"

    # 2. Test empty list
    assert inst.is_root_only([]) is False, "is_root_only([]) must be False"
    assert inst.include_root([]) is False, "include_root([]) must be False"
    img_list_empty = inst.process_nodes_to_image_list([], dummy_img)
    assert len(img_list_empty) == 1, f"Expected 1 image for nodes=[], got {len(img_list_empty)}"

    # 3. Test InternVL methods
    class DummyInternvlInstance:
        background_color = (255, 255, 255)
        input_size = (384, 384)
        is_root_only = ModelInternvl.is_root_only
        include_root = ModelInternvl.include_root
        process_nodes_to_image_list = ModelInternvl.process_nodes_to_image_list

    inst_intern = DummyInternvlInstance()

    assert inst_intern.is_root_only(None) is False
    assert inst_intern.include_root(None) is False
    img_list_intern = inst_intern.process_nodes_to_image_list(None, dummy_img)
    assert len(img_list_intern) == 1

    print("[PASS] test_root_handling_logic")


def test_interface_contract():
    from mssearch.models.modeling_llava import ModelGlobalLocal
    from mssearch.models.modeling_internvl import ModelInternvl
    from mssearch.models.modeling_qwenvl import ModelQwenVL

    required_methods = [
        "get_confidence_value",
        "batch_get_confidence_values",
        "generate_response",
        "batch_generate_response",
    ]

    for cls in [ModelGlobalLocal, ModelInternvl, ModelQwenVL]:
        for method in required_methods:
            assert hasattr(cls, method), f"Class {cls.__name__} missing required method: {method}"

    print("[PASS] test_interface_contract")


def test_no_silent_fallback():
    import mssearch.models as models_mod

    # If ModelGlobalLocal is requested, it should either be ModelGlobalLocal or raise ImportError, never ModelQwenVL
    try:
        cls = getattr(models_mod, "ModelGlobalLocal")
        from mssearch.models.modeling_llava import ModelGlobalLocal as ExpectedClass
        assert cls is ExpectedClass, f"ModelGlobalLocal resolved to {cls}, expected {ExpectedClass}"
        from mssearch.models.modeling_qwenvl import ModelQwenVL
        assert cls is not ModelQwenVL, "ModelGlobalLocal must NOT silently fall back to ModelQwenVL!"
    except ImportError:
        # Raising ImportError when llava is not installed is the correct fail-loud behavior
        pass

    print("[PASS] test_no_silent_fallback")


if __name__ == "__main__":
    test_root_handling_logic()
    test_interface_contract()
    test_no_silent_fallback()
    print("All model adapter self-checks passed successfully.")
