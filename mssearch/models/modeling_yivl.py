import os
import torch
from typing import List, Optional, Dict, Any, Union
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelYiVL:
    """
    Adapter for Yi-VL-34B (01-ai/Yi-VL-34B).
    Provides standard unified generate_response and multi-image prompt formatting.
    """
    def __init__(
        self,
        model_path: str = "01-ai/Yi-VL-34B",
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs
    ) -> None:
        self.device = device
        self.dtype = torch_dtype if torch_dtype is not None else torch.bfloat16
        local_files_only = kwargs.get("local_files_only", os.path.isdir(str(model_path)))

        print(f"[ModelYiVL] Loading Yi-VL model from '{model_path}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True,
            local_files_only=local_files_only
        )
        
        # Determine device map and memory budget
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

        # Image processor from vision tower
        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=self.device, dtype=self.dtype)
        self.image_processor = vision_tower.image_processor

    def get_prompt_from_qs(self, question: str) -> str:
        """Wrap query into Yi-VL conversational format."""
        return f"<|im_start|>user\n<image>\n{question}<|im_end|>\n<|im_start|>assistant\n"

    def generate_response(
        self,
        image_pil: Image.Image,
        question: str,
        max_new_tokens: int = 512
    ) -> str:
        """Run single-image forward inference on Yi-VL."""
        prompt = self.get_prompt_from_qs(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        image_tensor = self.image_processor.preprocess(image_pil, return_tensors="pt")["pixel_values"][0]
        image_tensor = image_tensor.unsqueeze(0).to(dtype=self.dtype, device=self.device)

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
