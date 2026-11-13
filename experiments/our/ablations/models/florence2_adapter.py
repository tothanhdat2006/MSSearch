"""
Florence-2 detector adapter for MSSearch ablation suite.
Uses the native transformers Florence-2 (AutoModelForImageTextToText / AutoProcessor,
florence-community/Florence-2-large) with task prompt <CAPTION_TO_PHRASE_GROUNDING>.
"""

from typing import List, Tuple, Dict, Any, Optional
import math
from PIL import Image
import torch
from .base import BaseDetector


class Florence2Detector(BaseDetector):
    """
    Open-vocabulary detector adapter using Florence-2-large
    (florence-community/Florence-2-large or florence-community/Florence-2-base).
    """

    def __init__(
        self,
        model_name_or_path: str = "florence-community/Florence-2-large",
        device: str = "cuda:0",
        torch_dtype: Optional[torch.dtype] = None,
        lazy_load: bool = False,
        **kwargs,
    ):
        self.model_name = model_name_or_path or "florence-community/Florence-2-large"
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)

        self.model = None
        self.processor = None

        if not lazy_load and self.model_name != "stub":
            self._load_model()

    def _load_model(self):
        """Loads Florence-2 processor and model (supports both native florence-community and legacy microsoft checkpoints)."""
        from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM

        print(f"[Florence2Detector] Loading '{self.model_name}' on {self.device} ({self.dtype})...")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
            ).to(self.device).eval()
        except Exception:
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            ).to(self.device).eval()

    def _ensure_loaded(self):
        if self.model is None and self.model_name != "stub":
            self._load_model()

    @torch.inference_mode()
    def batch_inference(
        self,
        image: Image.Image,
        text_prompts: List[str],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[int, Dict[str, Any]], List[int]]:
        """
        Runs phrase grounding using Florence-2 <CAPTION_TO_PHRASE_GROUNDING>.

        Returns:
            Tuple of (None, processed_results, text_id)
        """
        if not text_prompts:
            return None, {}, []

        text_id = list(range(1, len(text_prompts) + 1))

        if self.model_name == "stub":
            return None, {i: {"boxes": [], "scores": []} for i in text_id}, text_id

        self._ensure_loaded()
        w, h = image.size
        processed_results: Dict[int, Dict[str, Any]] = {}
        backbone_out = None

        for idx, prompt in enumerate(text_prompts, start=1):
            clean_text = prompt.strip()
            task_prompt = f"<CAPTION_TO_PHRASE_GROUNDING> {clean_text}"

            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if "pixel_values" in inputs and self.dtype in (torch.float16, torch.bfloat16):
                inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

            # Extract DaViT Spatial Features & Reshape to (1, C, H_f, W_f)
            if backbone_out is None and "pixel_values" in inputs:
                try:
                    pixel_vals = inputs["pixel_values"]
                    vision_feats = None
                    if hasattr(self.model, "_encode_image"):
                        vision_feats = self.model._encode_image(pixel_vals)
                    elif hasattr(self.model, "get_image_features"):
                        vision_feats = self.model.get_image_features(pixel_vals)

                    if vision_feats is not None and vision_feats.ndim == 3:
                        B_f, N_f, C_f = vision_feats.shape
                        grid_size = int(math.isqrt(N_f))
                        if grid_size > 0:
                            feat = vision_feats[:, :grid_size * grid_size, :].permute(0, 2, 1).reshape(B_f, C_f, grid_size, grid_size)
                            backbone_out = {"vision_features": feat}
                except Exception:
                    backbone_out = None

            generated_ids = self.model.generate(
                input_ids=inputs.get("input_ids"),
                pixel_values=inputs.get("pixel_values"),
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = self.processor.post_process_generation(
                generated_text,
                task="<CAPTION_TO_PHRASE_GROUNDING>",
                image_size=(w, h),
            )

            res_dict = parsed.get("<CAPTION_TO_PHRASE_GROUNDING>", {}) if isinstance(parsed, dict) else {}
            raw_boxes = res_dict.get("bboxes", [])

            clamped_boxes = []
            scores = []
            for b in raw_boxes:
                x1 = max(0, min(w, int(round(float(b[0])))))
                y1 = max(0, min(h, int(round(float(b[1])))))
                x2 = max(0, min(w, int(round(float(b[2])))))
                y2 = max(0, min(h, int(round(float(b[3])))))
                if x2 > x1 and y2 > y1:
                    clamped_boxes.append([x1, y1, x2, y2])
                    scores.append(0.95)

            processed_results[idx] = {
                "boxes": clamped_boxes,
                "scores": scores,
            }

        return backbone_out, processed_results, text_id
