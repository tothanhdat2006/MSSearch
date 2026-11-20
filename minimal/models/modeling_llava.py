"""
LLaVA-OneVision Model Adapter for STASH-Search Minimal.
Ported directly from mssearch/models/modeling_llava.py with ModelGlobalLocal architecture.
"""

import os
import re
import torch
from abc import ABC
from typing import List, Dict, Optional, Union, Any
from PIL import Image
import numpy as np
from transformers import AutoTokenizer

from .base import BaseMLLM

class _LlavaUnavailable:
    _error: Optional[Exception] = None

    @classmethod
    def raise_err(cls, *args, **kwargs):
        msg = "The 'llava' package is required to use ModelGlobalLocal (LLaVA-OneVision). Please install llava."
        if cls._error is not None:
            raise ImportError(f"{msg} Original error: {cls._error}") from cls._error
        raise ImportError(msg)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.raise_err()

    def __init__(self, *args, **kwargs):
        self.raise_err()

    def __getattr__(self, name):
        self.raise_err()


try:
    from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
    from llava.utils import disable_torch_init
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, process_images
except (ModuleNotFoundError, ImportError) as e:
    _LlavaUnavailable._error = e
    LlavaQwenForCausalLM = _LlavaUnavailable
    LlavaLlamaForCausalLM = _LlavaUnavailable
    LlavaConfig = _LlavaUnavailable
    disable_torch_init = lambda: None
    IMAGE_TOKEN_INDEX = -200
    DEFAULT_IMAGE_TOKEN = "<image>"
    conv_templates = {}
    tokenizer_image_token = _LlavaUnavailable.raise_err
    process_images = _LlavaUnavailable.raise_err


class ModelGlobalLocal(BaseMLLM):
    """
    ModelGlobalLocal wrapper for LLaVA-OneVision models, matching mssearch/models/modeling_llava.py.
    """

    def __init__(
        self,
        model_path: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
        conv_type: str = "qwen_1_5",
        device: str = "cuda:0",
        torch_dtype=torch.bfloat16,
        attn_implementation: str = "sdpa",
        padding_side: str = "left",
        max_anyres_num: int = 12,
        **kwargs
    ) -> None:
        if LlavaQwenForCausalLM is _LlavaUnavailable:
            _LlavaUnavailable.raise_err()

        disable_torch_init()
        self.device = device
        self.dtype = torch_dtype if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)

        attn_impl = kwargs.pop("attn_implementation", attn_implementation) or os.environ.get("ATTN_IMPLEMENTATION", "sdpa")

        def _load_causal_lm(model_cls, *load_args, **load_kwargs):
            try:
                return model_cls.from_pretrained(*load_args, **load_kwargs)
            except Exception as e_attn:
                curr_attn = load_kwargs.get("attn_implementation")
                if curr_attn == "flash_attention_2":
                    print(f"[ModelGlobalLocal] Warning: FlashAttention-2 failed ({e_attn}). Falling back to 'sdpa'/'eager'.")
                    load_kwargs["attn_implementation"] = "sdpa"
                    try:
                        return model_cls.from_pretrained(*load_args, **load_kwargs)
                    except Exception:
                        load_kwargs["attn_implementation"] = "eager"
                        return model_cls.from_pretrained(*load_args, **load_kwargs)
                elif curr_attn == "sdpa":
                    print(f"[ModelGlobalLocal] Warning: SDPA failed ({e_attn}). Falling back to 'eager'.")
                    load_kwargs["attn_implementation"] = "eager"
                    return model_cls.from_pretrained(*load_args, **load_kwargs)
                raise

        if "qwen" in model_path.lower():
            self.model = _load_causal_lm(
                LlavaQwenForCausalLM,
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype=self.dtype,
                attn_implementation=attn_impl,
            )
        else:
            llava_cfg = LlavaConfig.from_pretrained(model_path)
            if "v1.5" in model_path.lower():
                llava_cfg.delay_load = True
            self.model = _load_causal_lm(
                LlavaLlamaForCausalLM,
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype=self.dtype,
                attn_implementation=attn_impl,
                config=llava_cfg,
            )

        self.model.config.tokenizer_padding_side = padding_side
        self.model.to(self.device)

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=self.device)
            vision_tower.to(device=self.device, dtype=self.dtype)
        self.image_processor = vision_tower.image_processor

        if conv_type == "qwen_1_5" and "vicuna" in model_path.lower():
            conv_type = "vicuna_v1"
        self.conv_type = conv_type

        # Configure anyres image pinpoints
        self.set_anyres_num(max_anyres_num)
        self.init_prompts()

    def set_anyres_num(self, max_anyres_num: int):
        self.max_anyres_num = max_anyres_num
        if hasattr(self, "image_processor") and hasattr(self.image_processor, "crop_size"):
            patch_size = self.image_processor.crop_size["height"]
            if hasattr(self.model.config, "image_grid_pinpoints") and self.model.config.image_grid_pinpoints:
                new_possible_resolutions = []
                for p in self.model.config.image_grid_pinpoints:
                    x = p[0] // patch_size
                    y = p[1] // patch_size
                    if x * y <= max_anyres_num:
                        new_possible_resolutions.append(p)
                self.model.config.image_grid_pinpoints = new_possible_resolutions

    def init_prompts(self):
        self.prompts = {
            "search": {
                "pre_information_image1": f"{DEFAULT_IMAGE_TOKEN}\n",
                "answering_prompt": "Question: '{}'\nCould you answer the question based on the available visual information? Answer Yes or No.",
            }
        }

    def get_prompt_from_qs(self, qs: str) -> str:
        conv = conv_templates[self.conv_type].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def process_image_list_to_tensor(self, image_list: List[Image.Image]):
        image_tensor = process_images(image_list, self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=self.dtype, device=self.device) for _image in image_tensor]
        image_sizes = [image.size for image in image_list]
        return image_tensor, image_sizes

    @torch.inference_mode()
    def generate_response(self, image_pil=None, question: str = "", max_new_tokens: int = 512, **kwargs) -> str:
        """Direct free-form response generation for single crop or pure text."""
        if image_pil is None:
            conv = conv_templates[self.conv_type].copy()
            clean_q = question[len("<image>\n"):].strip() if question.startswith("<image>\n") else question.strip()
            conv.append_message(conv.roles[0], clean_q)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
            output_ids = self.model.generate(
                input_ids,
                images=None,
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
            )
            if output_ids.shape[1] > input_ids.shape[1]:
                gen_ids = output_ids[:, input_ids.shape[1]:]
            else:
                gen_ids = output_ids
            return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        # Image-grounded crop evaluation
        image_input = [image_pil] if not isinstance(image_pil, (list, tuple)) else list(image_pil)
        if question.startswith("<image>\n") or question.startswith(DEFAULT_IMAGE_TOKEN):
            qs = question
        else:
            qs = f"{DEFAULT_IMAGE_TOKEN}\n" + question

        prompt = self.get_prompt_from_qs(qs)
        image_tensor, image_sizes = self.process_image_list_to_tensor(image_input)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)

        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=max_new_tokens,
        )
        if output_ids.shape[1] > input_ids.shape[1]:
            gen_ids = output_ids[:, input_ids.shape[1]:]
        else:
            gen_ids = output_ids
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    # BaseMLLM Interface implementation
    def generate_text(
        self,
        prompt_or_messages: Union[str, List[Dict[str, Any]]],
        max_new_tokens: int = 96
    ) -> str:
        """Stage 1 Target Extraction text interface."""
        from ..stages.target_extraction import format_prompt_for_text_completion
        text = format_prompt_for_text_completion(prompt_or_messages)
        return self.generate_response(image_pil=None, question=text, max_new_tokens=max_new_tokens)

    def generate_response_batch(
        self,
        images: List[Image.Image],
        question: str,
        max_new_tokens: int = 512,
        **kwargs
    ) -> List[str]:
        """Stage 3 Native crop reading batch interface."""
        answers = []
        for img in images:
            ans = self.generate_response(image_pil=img, question=question, max_new_tokens=max_new_tokens)
            answers.append(ans)
        return answers
