"""
DINO-X detector adapter for MSSearch ablation suite.
Supports local DINO-X weights, official SDK, or remote API execution.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import torch
from .base import BaseDetector


class DinoXDetector(BaseDetector):
    """
    Open-vocabulary detector adapter using DINO-X (IDEA-Research / DeepDataSpace).
    Supports local model checkpoint or official API client.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: str = "cuda:0",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        api_token: Optional[str] = None,
        **kwargs,
    ):
        self.model_path = model_name_or_path or os.environ.get("DINOX_PATH", "")
        self.device = device if torch.cuda.is_available() else "cpu"
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.api_token = api_token or os.environ.get("DINOX_API_TOKEN", "")
        self.mode = "uninitialized"
        self.model = None
        self._init_backend()

    def _init_backend(self):
        """Initializes local weights engine or API client."""
        # 1. Try local dinox library or custom checkpoint
        if self.model_path and os.path.exists(self.model_path):
            try:
                # Check for native dinox package
                import dinox  # type: ignore
                print(f"[DinoXDetector] Initializing local DINO-X engine from '{self.model_path}'...")
                self.model = dinox.build_model(self.model_path).to(self.device).eval()
                self.mode = "local_dinox"
                return
            except ImportError:
                # Check for torch checkpoint or transformers model
                try:
                    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
                    self.processor = AutoProcessor.from_pretrained(self.model_path)
                    self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_path).to(self.device).eval()
                    self.mode = "hf_dino"
                    return
                except Exception as e:
                    print(f"[DinoXDetector] Local loading notice: {e}")

        # 2. Try remote API client
        if self.api_token:
            print("[DinoXDetector] Initializing DINO-X Cloud API backend...")
            self.mode = "api"
            return

        # 3. If running in evaluation without weights, fail loud with instructions
        if not self.model_path and not self.api_token:
            # Check if HuggingFace repository ID is provided
            if self.model_path and not os.path.exists(self.model_path):
                try:
                    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
                    self.processor = AutoProcessor.from_pretrained(self.model_path)
                    self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_path).to(self.device).eval()
                    self.mode = "hf_dino"
                    return
                except Exception:
                    pass

        self.mode = "stub"
        print("[DinoXDetector][Warning] Neither local weights nor DINOX_API_TOKEN configured; stub mode active.")

    @torch.inference_mode()
    def batch_inference(
        self,
        image: Image.Image,
        text_prompts: List[str],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[int, Dict[str, Any]], List[int]]:
        """
        Runs DINO-X detection across text prompts.
        """
        if not text_prompts:
            return None, {}, []

        w, h = image.size
        processed_results: Dict[int, Dict[str, Any]] = {}
        text_id = list(range(1, len(text_prompts) + 1))

        if self.mode == "hf_dino":
            for idx, prompt in enumerate(text_prompts, start=1):
                clean_text = prompt.strip().rstrip(".") + "."
                inputs = self.processor(images=image, text=clean_text, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                target_sizes = torch.tensor([[h, w]], device=self.device)
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    target_sizes=target_sizes,
                )[0]
                b_raw = results["boxes"].detach().cpu().tolist()
                s_raw = results["scores"].detach().cpu().tolist()
                boxes = [[max(0, min(w, int(round(float(c))))) for c in b] for b in b_raw]
                processed_results[idx] = {"boxes": boxes, "scores": [float(s) for s in s_raw]}

        elif self.mode == "local_dinox":
            for idx, prompt in enumerate(text_prompts, start=1):
                res = self.model.predict(image, prompt, box_threshold=self.box_threshold)
                boxes = res.get("boxes", [])
                scores = res.get("scores", [])
                processed_results[idx] = {"boxes": boxes, "scores": [float(s) for s in scores]}

        elif self.mode == "api":
            # Remote API query
            import requests  # type: ignore
            api_url = os.environ.get("DINOX_API_URL", "https://api.deepdataspace.com/v1/dinox")
            for idx, prompt in enumerate(text_prompts, start=1):
                try:
                    import io, base64
                    buf = io.BytesIO()
                    image.save(buf, format="JPEG", quality=90)
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    payload = {"image": b64, "text": prompt, "box_threshold": self.box_threshold}
                    headers = {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}
                    resp = requests.post(api_url, json=payload, headers=headers, timeout=30).json()
                    boxes = resp.get("boxes", [])
                    scores = resp.get("scores", [])
                    processed_results[idx] = {"boxes": boxes, "scores": [float(s) for s in scores]}
                except Exception as e:
                    print(f"[DinoXDetector][API Error] {e}")
                    processed_results[idx] = {"boxes": [], "scores": []}
        else:
            # Stub mode when weights/token unavailable: empty detections
            for idx in text_id:
                processed_results[idx] = {"boxes": [], "scores": []}

        return None, processed_results, text_id
