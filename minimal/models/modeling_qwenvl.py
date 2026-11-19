"""
Qwen-VL Model Adapter for Qwen3-VL-8B and Qwen2.5-VL.
Uses AutoModelForImageTextToText (compatible with transformers==5.6.2).
Provides pure-text generation for Stage 1 and batched crop reading for Stage 3.
"""

import os
from typing import List, Dict, Any, Union, Optional
from PIL import Image
import torch
from transformers import AutoProcessor, AutoConfig

from .base import BaseMLLM


class ModelQwenVL(BaseMLLM):
    """
    Qwen-VL adapter supporting Qwen3-VL-8B-Instruct and Qwen2.5-VL models
    under transformers==5.6.2 using AutoModelForImageTextToText.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda:0",
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
        self.model_path = model_name_or_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = torch_dtype if torch_dtype is not None else (
            torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
                torch.float16 if torch.cuda.is_available() else torch.float32
            )
        )

        hf_offline = os.environ.get("HF_HUB_OFFLINE", "0") in ("1", "true", "True") or \
                     os.environ.get("TRANSFORMERS_OFFLINE", "0") in ("1", "true", "True")
        local_files = kwargs.get("local_files_only", hf_offline or os.path.isdir(str(self.model_path)))

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=local_files
        )

        # In transformers 5.6.2 / Qwen3-VL, AutoModelForImageTextToText is the primary loader
        loaders = []
        try:
            from transformers import AutoModelForImageTextToText
            loaders.append(AutoModelForImageTextToText)
        except ImportError:
            pass
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            loaders.append(Qwen2_5_VLForConditionalGeneration)
        except ImportError:
            pass
        try:
            from transformers import AutoModelForVision2Seq
            loaders.append(AutoModelForVision2Seq)
        except ImportError:
            pass

        self.model = None
        for loader_cls in loaders:
            try:
                self.model = loader_cls.from_pretrained(
                    self.model_path,
                    dtype=self.dtype,
                    device_map="auto" if torch.cuda.device_count() > 1 else self.device,
                    trust_remote_code=True,
                    local_files_only=local_files,
                    low_cpu_mem_usage=True
                )
                print(f"[ModelQwenVL] Loaded '{self.model_path}' with {loader_cls.__name__} ({self.dtype})")
                break
            except (TypeError, ValueError, AttributeError):
                try:
                    self.model = loader_cls.from_pretrained(
                        self.model_path,
                        torch_dtype=self.dtype,
                        device_map="auto" if torch.cuda.device_count() > 1 else self.device,
                        trust_remote_code=True,
                        local_files_only=local_files,
                        low_cpu_mem_usage=True
                    )
                    print(f"[ModelQwenVL] Loaded '{self.model_path}' with {loader_cls.__name__} (torch_dtype={self.dtype})")
                    break
                except Exception:
                    continue
            except Exception:
                continue

        if self.model is None:
            raise RuntimeError(f"[ModelQwenVL] Failed to load '{self.model_path}' across available loaders: {[l.__name__ for l in loaders]}")

        self.model.eval()

    def generate_text(
        self,
        prompt_or_messages: Union[str, List[Dict[str, Any]]],
        max_new_tokens: int = 96
    ) -> str:
        """
        Pure text generation interface (zero visual image overhead).
        Used by Stage 1 Target Extraction for in-context semantic query parsing.
        """
        if isinstance(prompt_or_messages, str):
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt_or_messages}]}]
        else:
            messages = prompt_or_messages

        if hasattr(self.processor, "apply_chat_template"):
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text_prompt = str(prompt_or_messages)

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        inputs = tokenizer(text_prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        input_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[0, input_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return response

    def generate_response_batch(
        self,
        images: List[Image.Image],
        question: str,
        max_new_tokens: int = 512,
        **kwargs
    ) -> List[str]:
        """
        Batch evaluation of candidate native-resolution image crops.
        Evaluates each crop with the canonical prompt '<image>\\n{question}'.
        """
        if not images:
            return []

        clean_q = question
        while clean_q.startswith("<image>\n") or clean_q.startswith("<image>"):
            clean_q = clean_q[len("<image>\n"):] if clean_q.startswith("<image>\n") else clean_q[len("<image>"):]
        clean_q = clean_q.strip()

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        results: List[str] = []
        batch_size = kwargs.get("batch_size", 4)

        for i in range(0, len(images), max(1, batch_size)):
            chunk_imgs = images[i : i + max(1, batch_size)]

            if len(chunk_imgs) == 1 or batch_size <= 1:
                # Single-item path
                img = chunk_imgs[0]
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": clean_q}
                        ]
                    }
                ]
                if hasattr(self.processor, "apply_chat_template"):
                    prompt = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    prompt = f"<image>\n{clean_q}"

                inputs = self.processor(
                    text=[prompt],
                    images=[img],
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)

                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False
                    )
                input_len = inputs.input_ids.shape[1]
                gen_ids = output_ids[0, input_len:]
                resp = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                results.append(resp)
                continue

            try:
                prompts = []
                for img in chunk_imgs:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": clean_q}
                            ]
                        }
                    ]
                    if hasattr(self.processor, "apply_chat_template"):
                        p = self.processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    else:
                        p = f"<image>\n{clean_q}"
                    prompts.append(p)

                model_inputs = self.processor(
                    text=prompts,
                    images=chunk_imgs,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt"
                ).to(self.device)

                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        **model_inputs,
                        use_cache=True,
                        max_new_tokens=max_new_tokens,
                        do_sample=False
                    )

                trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
                if hasattr(self.processor, "batch_decode"):
                    decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                else:
                    decoded = tokenizer.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                results.extend([d.strip() for d in decoded])
            except Exception:
                # Safe fallback to single-item generation
                for img in chunk_imgs:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": clean_q}
                            ]
                        }
                    ]
                    if hasattr(self.processor, "apply_chat_template"):
                        prompt = self.processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    else:
                        prompt = f"<image>\n{clean_q}"

                    inputs = self.processor(
                        text=[prompt],
                        images=[img],
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)

                    with torch.inference_mode():
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False
                        )
                    input_len = inputs.input_ids.shape[1]
                    gen_ids = output_ids[0, input_len:]
                    resp = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    results.append(resp)

        return results


# Alias for backward compatibility
UnifiedMLLM = ModelQwenVL
