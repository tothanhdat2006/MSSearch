"""
Florence-2 detector adapter for MSSearch ablation suite.
Uses Microsoft Florence-2-large via Hugging Face AutoModelForCausalLM / AutoProcessor
with task prompt <CAPTION_TO_PHRASE_GROUNDING>.
"""

from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import torch
from .base import BaseDetector


class Florence2Detector(BaseDetector):
    """
    Open-vocabulary detector adapter using Microsoft Florence-2-large
    (microsoft/Florence-2-large or microsoft/Florence-2-base).
    """

    def __init__(
        self,
        model_name_or_path: str = "microsoft/Florence-2-large",
        device: str = "cuda:0",
        torch_dtype: Optional[torch.dtype] = None,
        lazy_load: bool = False,
        **kwargs,
    ):
        self.model_name = model_name_or_path or "microsoft/Florence-2-large"
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)

        self.model = None
        self.processor = None

        if not lazy_load and self.model_name != "stub":
            self._load_model()

    def _load_model(self):
        """Loads Florence-2 processor and model."""
        from transformers import AutoProcessor, AutoModelForCausalLM

        print(f"[Florence2Detector] Loading '{self.model_name}' on {self.device} ({self.dtype})...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
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

        for idx, prompt in enumerate(text_prompts, start=1):
            clean_text = prompt.strip()
            task_prompt = f"<CAPTION_TO_PHRASE_GROUNDING> {clean_text}"

            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if "pixel_values" in inputs and self.dtype in (torch.float16, torch.bfloat16):
                inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

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

        return None, processed_results, text_id
