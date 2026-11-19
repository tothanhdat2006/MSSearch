"""
InternVL3 Model Adapter for STASH-Search Minimal.
Ported directly from mssearch/models/modeling_internvl.py with ModelInternvl architecture.
"""

import os
import torch
from typing import List, Dict, Optional, Union, Any
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import types
from transformers import AutoTokenizer, AutoModel, GenerationConfig, GenerationMixin

from .base import BaseMLLM

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'


def new_prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **model_kwargs
):
    past_length = 0
    if past_key_values is not None:
        if hasattr(past_key_values, "get_seq_length"):
            past_length = past_key_values.get_seq_length()
        elif isinstance(past_key_values, (list, tuple)) and past_key_values[0] is not None:
            past_length = past_key_values[0][0].shape[2]
        if past_length > 0:
            input_ids = input_ids[:, -1:]

    position_ids = model_kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_length > 0:
            position_ids = position_ids[:, past_length:]

    if inputs_embeds is not None and past_length == 0:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "past_key_values": past_key_values,
            "use_cache": model_kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
    )
    return model_inputs


def legacy_forward_wrapper(self, *args, **kwargs):
    pkv = kwargs.get('past_key_values', None)
    if pkv is not None and hasattr(pkv, "to_legacy_cache"):
        if pkv.get_seq_length() == 0:
            kwargs['past_key_values'] = None
        else:
            kwargs['past_key_values'] = pkv.to_legacy_cache()
    return self._original_forward(*args, **kwargs)


def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = orig_width * orig_height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image: Image.Image, input_size: int = 448, max_num: int = 12, use_anyres: bool = True):
    transform = build_transform(input_size=input_size)
    if use_anyres:
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    else:
        images = [image.resize((input_size, input_size))]
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class ModelInternvl(BaseMLLM):
    """
    ModelInternvl adapter for InternVL3 models, matching mssearch/models/modeling_internvl.py.
    """

    def __init__(
        self,
        model_path: str = "OpenGVLab/InternVL3-8B",
        device: str = "cuda:0",
        torch_dtype=torch.bfloat16,
        **kwargs
    ) -> None:
        self.device = device
        self.dtype = torch_dtype if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=True if torch.cuda.is_available() else False,
            device_map=self.device,
            trust_remote_code=True
        ).eval()

        try:
            if hasattr(self.model, "language_model") and "internlm" in self.model.language_model.__class__.__name__.lower():
                lm_model = self.model.language_model
                lm_class = lm_model.__class__

                if not issubclass(lm_class, GenerationMixin):
                    lm_class.__bases__ = lm_class.__bases__ + (GenerationMixin,)

                lm_class.prepare_inputs_for_generation = new_prepare_inputs_for_generation

                if hasattr(lm_model, "model"):
                    inner_model = lm_model.model
                    if not hasattr(inner_model, "_original_forward"):
                        inner_model._original_forward = inner_model.forward
                        inner_model.forward = types.MethodType(legacy_forward_wrapper, inner_model)

                if not hasattr(lm_model, "generation_config") or lm_model.generation_config is None:
                    lm_model.generation_config = GenerationConfig.from_model_config(lm_model.config)
                    lm_model.generation_config.pad_token_id = self.tokenizer.pad_token_id
                    lm_model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        except Exception as e:
            print(f"[ModelInternvl] Warning: Failed to apply runtime patches: {e}")

        if hasattr(self.model, "img_context_token_id"):
            self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        self.input_size = (
            getattr(self.model.config, "force_image_size", 448),
            getattr(self.model.config, "force_image_size", 448)
        )
        self.generation_config = dict(max_new_tokens=1024, do_sample=False)
        self.anyres_num = kwargs.get("anyres_num", 12)
        self.init_prompts()

    def init_prompts(self):
        self.prompts = {
            "search": {
                "pre_information_image1": "<image>\n",
            }
        }

    @torch.inference_mode()
    def generate_response(self, image_pil=None, question: str = "", max_new_tokens: int = 512, **kwargs) -> str:
        """Direct free-form response generation interface for single image, crop, or text-only query."""
        if image_pil is None:
            clean_q = question[len("<image>\n"):].strip() if question.startswith("<image>\n") else question.strip()
            generation_cfg = dict(self.generation_config)
            if max_new_tokens is not None:
                generation_cfg["max_new_tokens"] = max_new_tokens
            return self.model.chat(self.tokenizer, None, clean_q, generation_cfg)

        image_input = [image_pil] if not isinstance(image_pil, (list, tuple)) else list(image_pil)
        clean_q = question
        while clean_q.startswith("<image>\n") or clean_q.startswith("<image>"):
            clean_q = clean_q[len("<image>\n"):] if clean_q.startswith("<image>\n") else clean_q[len("<image>"):]
        clean_q = clean_q.strip()

        qs = self.prompts["search"]["pre_information_image1"] + clean_q

        # Process images
        full_pixel_values = []
        num_patches_list = []
        for img in image_input:
            pixel_values = load_image(img, input_size=self.input_size[0], max_num=self.anyres_num, use_anyres=True)
            full_pixel_values.append(pixel_values.to(self.dtype))
            num_patches_list.append(pixel_values.shape[0])

        full_pixel_values = torch.cat(full_pixel_values, dim=0).to(dtype=self.dtype, device=self.device)
        generation_cfg = dict(self.generation_config)
        if max_new_tokens is not None:
            generation_cfg["max_new_tokens"] = max_new_tokens

        return self.model.chat(
            self.tokenizer,
            full_pixel_values,
            qs,
            generation_cfg,
            num_patches_list=num_patches_list
        )

    # BaseMLLM Interface implementation
    def generate_text(
        self,
        prompt_or_messages: Union[str, List[Dict[str, Any]]],
        max_new_tokens: int = 96
    ) -> str:
        """Stage 1 Target Extraction text interface."""
        if isinstance(prompt_or_messages, list):
            text = "\n".join([m.get("content", "") for m in prompt_or_messages if m.get("content")])
        else:
            text = str(prompt_or_messages)
        return self.generate_response(image_pil=None, question=text, max_new_tokens=max_new_tokens)

    def generate_response_batch(
        self,
        images: List[Image.Image],
        question: str,
        max_new_tokens: int = 512
    ) -> List[str]:
        """Stage 3 Native crop reading batch interface."""
        answers = []
        for img in images:
            ans = self.generate_response(image_pil=img, question=question, max_new_tokens=max_new_tokens)
            answers.append(ans)
        return answers
