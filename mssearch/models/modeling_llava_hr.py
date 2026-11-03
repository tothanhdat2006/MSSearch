import os
import torch
from typing import List, Optional, Dict, Any, Union
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelLlavaHR:
    """
    Adapter for LLaVA-HR-X series:
    - favor123/llava-hr-7b-sft-1024
    - favor123/llava-hr-13b-x-sft-1024
    Handles high-resolution dual-branch visual inputs.
    """
    def __init__(
        self,
        model_path: str = "favor123/llava-hr-7b-sft-1024",
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.float16,
        **kwargs
    ) -> None:
        self.device = device
        self.dtype = torch_dtype if torch_dtype is not None else torch.float16
        local_files_only = kwargs.get("local_files_only", os.path.isdir(str(model_path)))

        print(f"[ModelLlavaHR] Loading LLaVA-HR model from '{model_path}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True,
            local_files_only=local_files_only
        )

        if torch.cuda.device_count() >= 2:
            device_map = "auto"
        else:
            device_map = {"": self.device}

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=local_files_only
        ).eval()

        vision_tower = getattr(self.model, "get_vision_tower", lambda: None)()
        if vision_tower is not None and not getattr(vision_tower, "is_loaded", True):
            vision_tower.load_model()
            vision_tower.to(device=self.device, dtype=self.dtype)
            self.image_processor = vision_tower.image_processor
        elif hasattr(self.model, "image_processor"):
            self.image_processor = self.model.image_processor
        else:
            from transformers import AutoImageProcessor
            self.image_processor = AutoImageProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=local_files_only
            )

    def get_prompt_from_qs(self, question: str) -> str:
        """Standard LLaVA conversational template."""
        return f"USER: <image>\n{question}\nASSISTANT:"

    def generate_response(
        self,
        image_pil: Image.Image,
        question: str,
        max_new_tokens: int = 128
    ) -> str:
        """Run single-image forward inference on LLaVA-HR."""
        prompt = self.get_prompt_from_qs(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        if hasattr(self.image_processor, "preprocess"):
            img_out = self.image_processor.preprocess(image_pil, return_tensors="pt")
            image_tensor = img_out["pixel_values"] if isinstance(img_out, dict) else img_out
        else:
            image_tensor = self.image_processor(image_pil, return_tensors="pt")["pixel_values"]

        if isinstance(image_tensor, list):
            image_tensor = [t.to(dtype=self.dtype, device=self.device) for t in image_tensor]
        else:
            image_tensor = image_tensor.to(dtype=self.dtype, device=self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                images=image_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True
            )

        input_len = inputs["input_ids"].shape[1]
        decoded = self.tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)
        return decoded.strip()
