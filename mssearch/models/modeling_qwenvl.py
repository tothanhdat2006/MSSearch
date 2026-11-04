import warnings
import logging
from typing import Optional, List, Dict, Any, Tuple, Union
import torch
from torch.nn import CrossEntropyLoss
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from .tree import Node, NodeA
from .utils import *

# Suppress harmless upstream deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*min_pixels.*max_pixels.*")
warnings.filterwarnings("ignore", message=r".*`torch_dtype` is deprecated.*")


class ModelQwenVL:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cuda:0", torch_dtype=torch.float16, **kwargs) -> None:
        if "model_name" in kwargs and (model_path is None or model_path == "Qwen/Qwen2.5-VL-7B-Instruct"):
            model_path = kwargs.pop("model_name")
        self.dtype = torch_dtype if torch_dtype is not None else torch.float16
        num_gpus = torch.cuda.device_count()

        # Multi-GPU Sharded Allocation Budget for 2x T4 in pure float16 (NO 4-bit quantization):
        if num_gpus >= 2:
            device_map = "auto"
            max_memory = {
                0: "8.5GiB",   # GPU 0 hosts Vision Tower + Layers 0-14 (~7.8GB), leaves dynamic headroom for visual tokens
                1: "7.0GiB",   # GPU 1 hosts Layers 15-27 + Head (~6.5GB), leaves headroom for SAM3
                "cpu": "12GiB" # Bounded CPU ceiling to prevent Linux host OOM kill
            }
        else:
            device_map = device
            max_memory = None

        # SDPA backend for NVIDIA T4 (Turing sm_75 cannot run FlashAttention-2)
        actual_attn_implementation = "sdpa"
        local_files_only = kwargs.get("local_files_only", os.path.isdir(str(model_path)))
        print(f"[ModelQwenVL] Multi-GPU device_map: {device_map} | max_memory: {max_memory}")
        print(f"[ModelQwenVL] Attention Backend: {actual_attn_implementation} | Precision: {self.dtype} | Local files only: {local_files_only}")

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.tokenizer = self.processor.tokenizer
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype=self.dtype,
                attn_implementation=actual_attn_implementation,
                device_map=device_map,
                max_memory=max_memory,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
        except (TypeError, ValueError, AttributeError, Exception):
            try:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=self.dtype,
                    attn_implementation=actual_attn_implementation,
                    device_map=device_map,
                    max_memory=max_memory,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    local_files_only=local_files_only,
                )
            except Exception:
                try:
                    from transformers import AutoModelForImageTextToText
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        model_path,
                        torch_dtype=self.dtype,
                        attn_implementation=actual_attn_implementation,
                        device_map=device_map,
                        max_memory=max_memory,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        local_files_only=local_files_only,
                    )
                except Exception:
                    from transformers import AutoModelForVision2Seq
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        model_path,
                        torch_dtype=self.dtype,
                        attn_implementation=actual_attn_implementation,
                        device_map=device_map,
                        max_memory=max_memory,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        local_files_only=local_files_only,
                    )

        self.model.eval()
        self.device = self.model.device

        # Visual pixel budget: only override if explicitly provided, else use processor defaults
        if kwargs.get("min_pixels") is not None and hasattr(self.processor.image_processor, "min_pixels"):
            self.processor.image_processor.min_pixels = kwargs["min_pixels"]
        if kwargs.get("max_pixels") is not None and hasattr(self.processor.image_processor, "max_pixels"):
            self.processor.image_processor.max_pixels = kwargs["max_pixels"]

        self.bias_value = kwargs.get("bias_value", 0.6)
        print("bias_value:", self.bias_value)

        # We hard code the input size to 448x448 for QwenVL
        self.input_size = (448, 448)
        print("input size:", self.input_size)

        self.view_size = 336 ### crop size
        self.scale_size = 672

        self.background_color = tuple(int(x * 255) for x in self.processor.image_processor.image_mean)
        print("background color:", self.background_color)

        self.patch_scale = kwargs.get("patch_scale", None)
        print("patch scale:", self.patch_scale)

        self.init_prompts()
        self.init_index_yes_no()

    def init_index_yes_no(self):
        print("Yes:", self.tokenizer("Yes").input_ids)
        print("No:", self.tokenizer("No").input_ids)
        if len(self.tokenizer("Yes").input_ids) == 1 and len(self.tokenizer("No").input_ids) == 1:
            self.index_yes = self.tokenizer("Yes").input_ids[0]
            self.index_no = self.tokenizer("No").input_ids[0]
        else:
            assert len(self.tokenizer("Yes").input_ids) == 2 and len(self.tokenizer("No").input_ids) == 2
            self.index_yes = self.tokenizer("Yes").input_ids[1]
            self.index_no = self.tokenizer("No").input_ids[1]
        print("index_yes:", self.index_yes)
        print("index_no:", self.index_no)

    def get_confidence_weight(self, node: Node, max_depth: int):
        coeff = (1 - self.bias_value) / (max_depth ** 2)
        return coeff * (node.depth ** 2) + self.bias_value

    def init_prompts(self):
        self.prompts = {
            "search": {
                "pre_information_image2": "<image>\nThis is the main image.\n<image>\nThis is the zoomed-in view of the focus region.\n",
                "pre_information_image1": "<image>\n",
                "latent_prompt": "According to your common sense knowledge and the content of image, is it possible to find a {} by further zooming in the image? Answer Yes or No.",
                "existence_prompt": "Is there a '{}' in the image? Answer Yes or No.",
                "answering_prompt": "Question: '{}'\nCould you answer the question based on the available visual information? Answer Yes or No.",
            }
        }

    @torch.no_grad()
    def generate_visual_cues_using_ic(self, ic_examples, question: str, split_tag=r' and |, '):
        ic_question_template = ic_examples["question_template"]
        ic_question_list = ic_examples["question_list"]
        ic_response_list = ic_examples["response_list"]

        message = []
        for q, a in zip(ic_question_list, ic_response_list):
            message.extend([
                {"role": "user", "content": [{"type": "text", "text": ic_question_template.format(q)}]},
                {"role": "assistant", "content": [{"type": "text", "text": a}]}
            ])
        message.extend([
            {"role": "user", "content": [{"type": "text", "text": ic_question_template.format(question)}]},
        ])
        texts = [self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)]
        model_inputs = self.processor(
            text=texts,
            images=None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False)
        model_inputs = model_inputs.to(self.device)
        if model_inputs.input_ids.shape[0] == 1 and "attention_mask" in model_inputs:
            if not (model_inputs.attention_mask == 0).any():
                model_inputs.pop("attention_mask", None)
        generated_ids = self.model.generate(**model_inputs, use_cache=True, max_new_tokens=256, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]

        targets_sentence = extract_targets_SGVS(response)
        if targets_sentence is not None:
            targets = split_targets_sentence(targets_sentence, split_tag)
        else:
            targets = []

        return targets

    def filter_visual_cues(self, image_pil: Image.Image, targets, root_node, decomposed_question_template,answering_confidence_threshold_upper, show_value=False):
        assert root_node.is_root
        ret = []
        if len(targets) > 1:
            for target in targets:
                if target.startswith("all "):
                    ret.append(target)
                    continue
                inputs = {
                    "node": root_node,
                    "image_pil": image_pil,
                    "confidence_type": "answering",
                    "input_ele": decomposed_question_template.format(target)
                }
                inputs['root_anyres'] = False
                conf_root = self.get_confidence_value(**inputs)
                if show_value:
                    print(target, conf_root)
                if conf_root < answering_confidence_threshold_upper:
                    ret.append(target)
        else:
            ret = targets[:]
        return ret

    def get_prompt_from_qs(self, qs, response=None, show_prompt=False):
        message = []
        message.append({"role": "user", "content": []})
        while "<image>\n" in qs:
            index = qs.find("<image>\n")
            if index == 0:
                message[0]["content"].append({"type": "image"})
                qs = qs[len("<image>\n"):]
            else:
                message[0]["content"].append({"type": "text", "text": qs[:index]})
                qs = qs[index:]
        if len(qs) > 0:
            message[0]["content"].append({"type": "text", "text": qs})
        if response is not None:
            message.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        texts = [self.processor.apply_chat_template(message, tokenize=False,add_generation_prompt=True if response is None else False)]

        return texts[0]

    def resize_image(self, image_pil):
        img = deepcopy(image_pil)
        input_size = self.input_size[0]
        if img.width > img.height:
            img = img.resize((input_size, int(img.height * input_size / img.width)))
        else:
            img = img.resize((int(img.width * input_size / img.height), input_size))
        return img

    def resize_image_seg(self, image_pil, max_size):
        img = deepcopy(image_pil)
        input_size = max_size
        if img.width > img.height:
            img = img.resize((input_size, int(img.height * input_size / img.width)))
        else:
            img = img.resize((int(img.width * input_size / img.height), input_size))
        return img

    def get_prompt_tag(self, image_list):
        if len(image_list) == 1:
            prompt_tag = "global"
        elif len(image_list) == 2:
            prompt_tag = "zoom"
        else:
            raise ValueError
        return prompt_tag

    def save_crop(self, image_pil, node, image_path):
        resized_bboxes = self.get_patch(node.state.bbox, image_pil.width, image_pil.height,patch_size=self.input_size[0], patch_scale=self.patch_scale)

        cropped_image = image_pil.crop(resized_bboxes)
        cropped_image.save(image_path)

    def is_root_only(self, nodes: Optional[List[Node]]):
        if not nodes:
            return False
        return (len(nodes) == 1 and getattr(nodes[0], 'is_root', False))

    def include_root(self, nodes: Optional[List[Node]]):
        if not nodes:
            return False
        return any(getattr(node, 'is_root', False) for node in nodes)

    def get_bbox_in_square_image(self, bbox, left, top):
        x1, y1, x2, y2 = bbox
        return [x1 + left, y1 + top, x2 + left, y2 + top]

    def zoom_in(self, croped_view):
        c_width = croped_view.width
        c_height = croped_view.height
        resize_ratio = min(self.scale_size * 1.0 / c_width, self.scale_size * 1.0 / c_height)
        resize_ratio = max(1.0, resize_ratio)
        zoomed_view = croped_view.resize((int(c_width * resize_ratio), int(c_height * resize_ratio)))

        return zoomed_view

    def get_patch(self, bbox, image_width, image_height, patch_size, patch_scale=None):
        object_width = int(np.ceil(bbox[2]))
        object_height = int(np.ceil(bbox[3]))

        object_center_x = int(bbox[0] + bbox[2] / 2)
        object_center_y = int(bbox[1] + bbox[3] / 2)

        patch_width = max(object_width, patch_size)
        patch_height = max(object_height, patch_size)
        if patch_scale is not None:
            patch_width = int(patch_width * patch_scale)
            patch_height = int(patch_height * patch_scale)

        left = max(0, object_center_x - patch_width // 2)
        right = min(left + patch_width, image_width)

        top = max(0, object_center_y - patch_height // 2)
        bottom = min(top + patch_height, image_height)

        return [left, top, right, bottom]

    def process_nodes_to_image_list(self, nodes, image_pil, root_anyres=True):
        square_image, left, top = expand2square(image_pil, self.background_color)
        if not nodes:
            # Physical resolution shield: clamp root image to max 1024px to prevent SDPA attention buffer OOM
            if max(image_pil.width, image_pil.height) > 1024:
                return [self.resize_image_seg(image_pil, 1024)]
            return [deepcopy(image_pil)] if root_anyres else [self.resize_image(image_pil)]
        if self.is_root_only(nodes):
            if max(image_pil.width, image_pil.height) > 1024:
                return [self.resize_image_seg(image_pil, 1024)]
            return [deepcopy(image_pil)] if root_anyres else [self.resize_image(image_pil)]
        if len(nodes) == 0 or self.include_root(nodes):
            if max(image_pil.width, image_pil.height) > 1024:
                return [self.resize_image_seg(image_pil, 1024)]
            return [deepcopy(image_pil)]

        resized_bboxes = []
        for node in nodes:
            source = getattr(node, 'search_source', 'fine')
            if source == 'fast':
                # Fast: view_size=1/3, none scale
                current_patch_size = self.view_size // 3
                current_patch_scale = None
            else:
                # Fine
                current_patch_size = self.view_size
                current_patch_scale = self.patch_scale

            patch_bbox = self.get_patch(
                bbox=node.state.bbox,
                image_width=image_pil.width,
                image_height=image_pil.height,
                patch_size=current_patch_size,
                patch_scale=current_patch_scale
            )
            resized_bboxes.append(patch_bbox)

        resized_bboxes = merge_bbox_list(resized_bboxes, threshold=0)

        target_bboxes = [self.get_bbox_in_square_image(b, left, top) for b in resized_bboxes]
        union_bboxes = union_all_bboxes(target_bboxes)
        if union_bboxes is None:
            return [square_image]
        croped_view = square_image.crop(union_bboxes)
        zoomed_view = self.zoom_in(croped_view)

        return [self.resize_image(image_pil), croped_view, zoomed_view]

    @torch.no_grad()
    def get_confidence_value(self, node: Optional[List[NodeA]] = None, image_pil: Image.Image = None, confidence_type: str = 'existence', input_ele: str = '', root_anyres: bool = True, node_list: Optional[List[NodeA]] = None):
        assert confidence_type in ['existence', 'latent', 'answering']
        actual_nodes = node if node is not None else node_list
        image_list = self.process_nodes_to_image_list(actual_nodes, image_pil, root_anyres=root_anyres)
        if len(image_list) > 1:
            raw_image_resize, croped_view, zoom_view = image_list
            image_input = [zoom_view]
        else:
            image_input = image_list
        prompt_tag = "search"
        if len(image_input) > 1:
            pre_information = "pre_information_image2"
        else:
            pre_information = "pre_information_image1"

        qs = self.prompts[prompt_tag][pre_information] + self.prompts[prompt_tag][f"{confidence_type}_prompt"].format(input_ele)
        prompt = self.get_prompt_from_qs(qs)

        model_inputs = self.processor(
            text=[prompt],
            images=image_input,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        model_inputs = model_inputs.to(self.device)
        # Fused Causal SDPA: Strip redundant dense attention mask on batch_size=1 to avoid 27.73 GiB OOM
        if model_inputs.input_ids.shape[0] == 1 and "attention_mask" in model_inputs:
            if not (model_inputs.attention_mask == 0).any():
                model_inputs.pop("attention_mask", None)
        outputs = self.model(**model_inputs)
        conf = self._cal_confidence(outputs)
        del model_inputs
        del outputs
        return conf

    @torch.no_grad()
    def _cal_confidence(self, outputs):
        logits_yesno = outputs.logits[0, -1, [self.index_yes, self.index_no]]
        confidence = torch.softmax(logits_yesno, dim=-1)[0]
        confidence = 2 * (confidence.item() - 0.5)  # [-1, 1]
        return confidence

    @torch.inference_mode()
    def free_form_using_nodes(self, image_pil, question, searched_nodes: List[NodeA], return_zoomed_view=False, max_new_tokens: int = 128):
        if isinstance(image_pil, (list, tuple)):
            image_input = list(image_pil)
            image_list = image_input
        else:
            image_list = self.process_nodes_to_image_list(searched_nodes, image_pil)
            if len(image_list) > 1:
                raw_image_resize, croped_view, zoom_view = image_list
                image_input = [zoom_view]
            else:
                image_input = image_list

        prompt_tag = "search"
        if len(image_input) > 1:
            pre_information = "pre_information_image2"
        else:
            pre_information = "pre_information_image1"

        if question.startswith("<image>\n"):
            qs = question
        else:
            qs = self.prompts[prompt_tag][pre_information] + question
        prompt = self.get_prompt_from_qs(qs)
        model_inputs = self.processor(
            text=[prompt],
            images=image_input,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        model_inputs = model_inputs.to(self.device)
        if model_inputs.input_ids.shape[0] == 1 and "attention_mask" in model_inputs:
            if not (model_inputs.attention_mask == 0).any():
                model_inputs.pop("attention_mask", None)
        generated_ids = self.model.generate(**model_inputs, use_cache=True, max_new_tokens=max_new_tokens, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        outputs = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]

        del model_inputs
        del generated_ids

        if return_zoomed_view:
            response = {'text': outputs, 'output_image': image_list[1] if len(image_list) > 1 else image_list[0]}
            return response

        return outputs

    @torch.inference_mode()
    def generate_response(self, image_pil: Union[Image.Image, List[Image.Image]], question: str, max_new_tokens: int = 128, **kwargs) -> str:
        """Direct free-form text response generation interface for single image or dual-scale [global, crop] list."""
        return self.free_form_using_nodes(
            image_pil=image_pil,
            question=question,
            searched_nodes=[],
            return_zoomed_view=False,
            max_new_tokens=max_new_tokens
        )

    @torch.no_grad()
    def batch_get_confidence_values(
        self,
        image_list: List[Image.Image],
        confidence_type: str = 'existence',
        input_ele: Union[str, List[str]] = '',
        batch_size: int = 4,
        custom_prompt_templates: Optional[Union[str, List[str]]] = None
    ) -> List[float]:
        """
        Batch-evaluates VLM existence or latent logit probes across candidate images (batch size 4 or 8).
        Calculates normalized confidence in [-1, 1] via softmax over [Yes, No] logits.
        Supports both homogeneous (string) and heterogeneous (list of strings) queries and prompt templates.
        """
        assert confidence_type in ['existence', 'latent', 'answering']
        if not image_list:
            return []

        prompt_tag = "search"
        n_imgs = len(image_list)

        # Build individual queries
        if isinstance(input_ele, (list, tuple)):
            queries = list(input_ele)
            if len(queries) < n_imgs:
                queries.extend([queries[-1] if queries else ""] * (n_imgs - len(queries)))
        else:
            queries = [str(input_ele)] * n_imgs

        # Build templates
        if custom_prompt_templates is not None:
            if isinstance(custom_prompt_templates, (list, tuple)):
                templates = list(custom_prompt_templates)
                if len(templates) < n_imgs:
                    templates.extend([templates[-1] if templates else None] * (n_imgs - len(templates)))
            else:
                templates = [custom_prompt_templates] * n_imgs
        else:
            templates = [None] * n_imgs

        # Build pre-computed prompts for all items
        all_prompts = []
        for q_str, tmpl in zip(queries, templates):
            if tmpl:
                q_prompt = tmpl.format(q_str)
            else:
                q_prompt = self.prompts[prompt_tag][f"{confidence_type}_prompt"].format(q_str)
            qs = self.prompts[prompt_tag]["pre_information_image1"] + q_prompt
            all_prompts.append(self.get_prompt_from_qs(qs))

        results: List[float] = []
        for i in range(0, len(image_list), max(1, batch_size)):
            chunk = image_list[i : i + max(1, batch_size)]
            chunk_prompts = all_prompts[i : i + max(1, batch_size)]
            chunk_queries = queries[i : i + max(1, batch_size)]
            chunk_templates = templates[i : i + max(1, batch_size)]

            if len(chunk) == 1 or batch_size <= 1:
                for img, q_str, tmpl in zip(chunk, chunk_queries, chunk_templates):
                    orig_prompt = None
                    if tmpl and hasattr(self, "prompts") and "search" in self.prompts:
                        orig_prompt = self.prompts["search"].get(f"{confidence_type}_prompt")
                        self.prompts["search"][f"{confidence_type}_prompt"] = tmpl
                    try:
                        results.append(self.get_confidence_value(node=None, image_pil=img, confidence_type=confidence_type, input_ele=q_str))
                    finally:
                        if orig_prompt is not None:
                            self.prompts["search"][f"{confidence_type}_prompt"] = orig_prompt
                continue

            try:
                model_inputs = self.processor(
                    text=chunk_prompts,
                    images=chunk,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left"
                ).to(self.device)

                outputs = self.model(**model_inputs)
                logits_yesno = outputs.logits[:, -1, [self.index_yes, self.index_no]]
                probs = torch.softmax(logits_yesno, dim=-1)[:, 0]
                confidences = (2 * (probs - 0.5)).tolist()
                results.extend(confidences)
                del model_inputs
                del outputs
            except Exception:
                # Safe single-item fallback if batch forward pass fails
                for img, q_str, tmpl in zip(chunk, chunk_queries, chunk_templates):
                    orig_prompt = None
                    if tmpl and hasattr(self, "prompts") and "search" in self.prompts:
                        orig_prompt = self.prompts["search"].get(f"{confidence_type}_prompt")
                        self.prompts["search"][f"{confidence_type}_prompt"] = tmpl
                    try:
                        results.append(self.get_confidence_value(node=None, image_pil=img, confidence_type=confidence_type, input_ele=q_str))
                    finally:
                        if orig_prompt is not None:
                            self.prompts["search"][f"{confidence_type}_prompt"] = orig_prompt

        return results

    @torch.inference_mode()
    def batch_generate_response(
        self,
        image_inputs: List[Union[Image.Image, List[Image.Image]]],
        questions: List[str],
        max_new_tokens: int = 128,
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """
        Batched text generation across candidate image crops / dual-scale views.
        Processes up to batch_size items concurrently with automatic single-item fallback.
        """
        if not image_inputs:
            return []
        if len(image_inputs) != len(questions):
            raise ValueError(f"image_inputs length ({len(image_inputs)}) must match questions length ({len(questions)})")

        results: List[str] = []
        for i in range(0, len(image_inputs), max(1, batch_size)):
            chunk_imgs = image_inputs[i : i + max(1, batch_size)]
            chunk_qs = questions[i : i + max(1, batch_size)]

            if len(chunk_imgs) == 1 or batch_size <= 1:
                for img, q in zip(chunk_imgs, chunk_qs):
                    results.append(self.generate_response(image_pil=img, question=q, max_new_tokens=max_new_tokens, **kwargs))
                continue

            try:
                prompts = []
                for img_inp, q in zip(chunk_imgs, chunk_qs):
                    is_multi_img = isinstance(img_inp, (list, tuple)) and len(img_inp) > 1
                    pre_info = "pre_information_image2" if is_multi_img else "pre_information_image1"
                    if q.startswith("<image>\n"):
                        qs_full = q
                    else:
                        qs_full = self.prompts["search"][pre_info] + q
                    prompts.append(self.get_prompt_from_qs(qs_full))

                model_inputs = self.processor(
                    text=prompts,
                    images=chunk_imgs,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left"
                ).to(self.device)

                generated_ids = self.model.generate(
                    **model_inputs,
                    use_cache=True,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
                trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
                decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                results.extend([d.strip() for d in decoded])
            except Exception:
                # Safe fallback to single-item generation
                for img, q in zip(chunk_imgs, chunk_qs):
                    results.append(self.generate_response(image_pil=img, question=q, max_new_tokens=max_new_tokens, **kwargs))

        return results

    @torch.inference_mode()
    def generate_text(self, prompt: Union[str, List[Dict[str, Any]]], max_new_tokens: int = 64) -> str:
        """
        Pure text generation interface (zero visual image overhead) for question parsing and entity extraction.
        Supports both raw text prompt and structured chat message list with system/few-shot turns.
        """
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        else:
            messages = prompt
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text_prompt],
            images=None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False
        ).to(self.device)
        if inputs.input_ids.shape[0] == 1 and "attention_mask" in inputs:
            if not (inputs.attention_mask == 0).any():
                inputs.pop("attention_mask", None)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True
        )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    @torch.inference_mode()
    def multiple_choices_inference(self, image_pil, question, options, searched_nodes: List[NodeA] = None):
        image_list = self.process_nodes_to_image_list(searched_nodes, image_pil)
        if len(image_list) > 1:
            raw_image_resize, croped_view, zoom_view = image_list
            image_input = [zoom_view]
        else:
            image_input = image_list

        prompt_tag = "search"
        if len(image_input) > 1:
            pre_information = "pre_information_image2"
        else:
            pre_information = "pre_information_image1"

        if question.startswith("<image>\n"):
            qs = question
        else:
            qs = self.prompts[prompt_tag][pre_information] + question
        prompt = self.get_prompt_from_qs(qs)

        model_inputs = self.processor(
            text=[prompt],
            images=image_input,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        model_inputs = model_inputs.to(self.device)
        if model_inputs.input_ids.shape[0] == 1 and "attention_mask" in model_inputs:
            if not (model_inputs.attention_mask == 0).any():
                model_inputs.pop("attention_mask", None)
        question_input_ids = model_inputs.input_ids
        len_question_input_ids = question_input_ids.shape[1]
        output_question = self.model(**model_inputs)
        len_question_logits = output_question.logits.shape[1]

        loss_list = []
        for option in options:
            full_prompt = self.get_prompt_from_qs(qs, option)

            full_model_inputs = self.processor(
                text=[full_prompt],
                images=image_input,
                return_tensors="pt",
                padding=True,
                padding_side="left",
            )
            full_model_inputs = full_model_inputs.to(self.device)
            if full_model_inputs.input_ids.shape[0] == 1 and "attention_mask" in full_model_inputs:
                if not (full_model_inputs.attention_mask == 0).any():
                    full_model_inputs.pop("attention_mask", None)
            full_input_ids = full_model_inputs.input_ids

            output_option = self.model(**full_model_inputs)
            logits = output_option.logits[:, len_question_logits - 1:-1]

            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, logits.size(-1))
            labels = full_input_ids[:, len_question_input_ids:].view(-1)

            loss = loss_fct(logits, labels)
            loss_list.append(loss)

        option_chosen = torch.stack(loss_list).argmin()
        return option_chosen.cpu().item()

