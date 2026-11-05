import torch
from torch.nn import CrossEntropyLoss
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any
from PIL import Image
import numpy as np
from transformers import AutoTokenizer
try:
    from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
    from llava.utils import disable_torch_init
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, process_images
except (ModuleNotFoundError, ImportError) as e:
    def _raise_llava_import_error(*args, **kwargs):
        raise ImportError(
            f"The 'llava' package is required to use ModelGlobalLocal. "
            f"Please ensure it is installed in your environment. Original error: {e}"
        ) from e
    LlavaQwenForCausalLM = _raise_llava_import_error
    LlavaLlamaForCausalLM = _raise_llava_import_error
    LlavaConfig = _raise_llava_import_error
    disable_torch_init = lambda: None
    IMAGE_TOKEN_INDEX = -200
    DEFAULT_IMAGE_TOKEN = "<image>"
    conv_templates = {}
    tokenizer_image_token = _raise_llava_import_error
    process_images = _raise_llava_import_error
from .tree import Node, NodeA
from .utils import *
from sentence_transformers import SentenceTransformer, util

class Model(ABC):
    def __init__(self, model_path: str, conv_type: str = "qwen_1_5", device: str = "cuda:0", torch_dtype=torch.float16,
                 attn_implementation="flash_attention_2", padding_side="left", **kwargs) -> None:
        disable_torch_init()
        self.device = device
        self.dtype = torch_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)

        print("padding side:", self.tokenizer.padding_side)

        if "qwen" in model_path:
            self.model = LlavaQwenForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype=self.dtype,
                attn_implementation=attn_implementation,
            )
        else:
            llava_cfg = LlavaConfig.from_pretrained(model_path)
            if "v1.5" in model_path.lower():
                llava_cfg.delay_load = True
            self.model = LlavaLlamaForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype=self.dtype,
                attn_implementation=attn_implementation,
                config=llava_cfg
            )
        print("model:", type(self.model))
        self.model.config.tokenizer_padding_side = padding_side
        self.model.to(self.device)

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=self.device)
            vision_tower.to(device=self.device, dtype=torch.float16)
        self.image_processor = vision_tower.image_processor

        if conv_type == "qwen_1_5" and "vicuna" in model_path.lower():
            conv_type = "vicuna_v1"
        self.conv_type = conv_type
        self.bias_value = kwargs.get("bias_value", 0.2)
        print("bias_value:", self.bias_value)

        self.input_size = (self.image_processor.crop_size['width'], self.image_processor.crop_size['height'])
        self.background_color = tuple(int(x * 255) for x in self.image_processor.image_mean)
        print("input size:", self.input_size)
        print("background color:", self.background_color)

        self.patch_scale = kwargs.get("patch_scale", None)
        print("patch scale:", self.patch_scale)

        self.init_prompts()
        self.init_index_yes_no()

        self.view_size = max(self.image_processor.crop_size['width'], self.image_processor.crop_size['height'])
        self.scale_size = max(self.image_processor.crop_size['width'], self.image_processor.crop_size['height'])

    @abstractmethod
    def init_prompts(self):
        pass

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

    @abstractmethod
    def process_nodes_to_image_list(self, nodes: List[NodeA], image_pil: Image.Image) -> List[Image.Image]:
        pass

    @abstractmethod
    def get_prompt_tag(self, image_list: List[Image.Image]):
        pass

    @abstractmethod
    def process_image_list_to_tensor(self, image_list: List[Image.Image]):
        pass

    def get_prompt_from_qs(self, qs, response=None, show_prompt=False):
        conv = conv_templates[self.conv_type].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], response)
        prompt = conv.get_prompt()
        return prompt

    @torch.no_grad()
    def get_confidence_value(self, node: List[NodeA], image_pil: Image.Image, confidence_type: str, input_ele):
        assert confidence_type in ['existence', 'latent', 'answering']
        image_list = self.process_nodes_to_image_list(node, image_pil)
        prompt_tag = self.get_prompt_tag(image_list)
        qs = self.prompts[prompt_tag]["pre_information"] + self.prompts[prompt_tag][f"{confidence_type}_prompt"].format(
            input_ele)
        prompt = self.get_prompt_from_qs(qs)

        image_tensor, image_sizes = self.process_image_list_to_tensor(image_list)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        outputs = self.model(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            modalities=["image"] * len(input_ids),
            return_dict=True
        )
        return self._cal_confidence(outputs)

    @torch.no_grad()
    def _cal_confidence(self, outputs):
        logits_yesno = outputs.logits[0, -1, [self.index_yes, self.index_no]]
        confidence = torch.softmax(logits_yesno, dim=-1)[0]
        confidence = 2 * (confidence.item() - 0.5)  # [-1, 1]
        return confidence

    @torch.inference_mode()
    def free_form_using_nodes(self, image_pil, question, searched_nodes: List[Node] = None, return_zoomed_view=False, max_new_tokens: int = 128):
        if searched_nodes is None:
            searched_nodes = []
        if isinstance(image_pil, (list, tuple)):
            image_list = list(image_pil)
        else:
            image_list = self.process_nodes_to_image_list(searched_nodes, image_pil)
        prompt_tag = self.get_prompt_tag(image_list)
        if question.startswith("<image>\n") or question.startswith(DEFAULT_IMAGE_TOKEN):
            qs = question
        elif prompt_tag in self.prompts and "pre_information" in self.prompts[prompt_tag]:
            qs = self.prompts[prompt_tag]["pre_information"] + question
        elif "search" in self.prompts and "pre_information_image1" in self.prompts["search"]:
            qs = self.prompts["search"]["pre_information_image1"] + question
        else:
            qs = f"{DEFAULT_IMAGE_TOKEN}\n" + question
        prompt = self.get_prompt_from_qs(qs)

        image_tensor, image_sizes = self.process_image_list_to_tensor(image_list)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).to(self.device)

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
        outputs = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        if return_zoomed_view:
            response = {'text': outputs, 'output_image': image_list[1] if len(image_list) > 1 else image_list[0]}
            return response

        return outputs

    @torch.inference_mode()
    def generate_response(self, image_pil=None, question: str = "", max_new_tokens: int = 128, **kwargs) -> str:
        """Direct free-form text response generation interface for single image, crop, or text-only query."""
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
        return self.free_form_using_nodes(
            image_pil=image_pil,
            question=question,
            searched_nodes=[],
            return_zoomed_view=False,
            max_new_tokens=max_new_tokens
        )

    @torch.inference_mode()
    def multiple_choices_inference(self, image_pil, question, options, searched_nodes: List[Node] = None):
        image_list = self.process_nodes_to_image_list(searched_nodes, image_pil)
        prompt_tag = self.get_prompt_tag(image_list)
        qs = self.prompts[prompt_tag]["pre_information"] + question
        prompt = self.get_prompt_from_qs(qs)
        image_tensor, image_sizes = self.process_image_list_to_tensor(image_list)

        question_input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).to(self.device)

        output_question = self.model(
            question_input_ids,
            use_cache=True,
            images=image_tensor,
            image_sizes=image_sizes)

        question_logits = output_question.logits
        question_past_key_values = output_question.past_key_values

        loss_list = []

        for option in options:
            full_prompt = self.get_prompt_from_qs(qs, option)
            full_input_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).to(self.device)
            option_answer_input_ids = full_input_ids[:, question_input_ids.shape[1]:]
            output_option = self.model(input_ids=option_answer_input_ids,
                                       use_cache=True,
                                       attention_mask=torch.ones(1, question_logits.shape[1] +
                                                                 option_answer_input_ids.shape[1],
                                                                 device=full_input_ids.device),
                                       past_key_values=question_past_key_values)

            logits = torch.cat([question_logits[:, -1:], output_option.logits[:, :-1]], 1)

            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, self.model.config.vocab_size)
            labels = option_answer_input_ids.view(-1)
            loss = loss_fct(logits, labels)
            loss_list.append(loss)

        option_chosen = torch.stack(loss_list).argmin()

        return option_chosen.cpu().item()

    @torch.inference_mode()
    def get_semantic_order(self, image_pil: Image.Image, input_ele, semantic_list, semantic_num):
        outputs, ranked_list = [], []
        return ranked_list

    @torch.inference_mode()
    def get_fast_confidence(self, image_pil, question, sub_region):
        fast_confidence = 0.0
        return fast_confidence

    @torch.no_grad()
    def get_search_confidence(self, node: Node, image_pil: Image.Image, confidence_type: str, input_ele):
        search_confidence = 0.0
        return search_confidence

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

    def save_crop(self, image_pil, node, image_path):
        resized_bboxes = self.get_patch(node.state.bbox, image_pil.width, image_pil.height,
                                        patch_size=self.input_size[0], patch_scale=self.patch_scale)
        cropped_image = image_pil.crop(resized_bboxes)
        cropped_image.save(image_path)

    @torch.no_grad()
    def generate_visual_cues_using_ic(self, ic_examples, question: str, split_tag=r' and |, '):
        ic_question_template = ic_examples["question_template"]
        ic_question_list = ic_examples["question_list"]
        ic_response_list = ic_examples["response_list"]
        fs_prompt = []
        for q, a in zip(ic_question_list, ic_response_list):
            fs_prompt.append((ic_question_template.format(q), a))
        conv = conv_templates[self.conv_type].copy()
        for qa in fs_prompt:
            conv.append_message(conv.roles[0], qa[0])
            conv.append_message(conv.roles[1], qa[1])
        conv.append_message(conv.roles[0], ic_question_template.format(question))
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)

        output_ids = self.model.generate(
            input_ids,
            images=None,
            do_sample=False,
            temperature=0,
            max_new_tokens=256,
        )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        targets_sentence = extract_targets(outputs)
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
                if "anyres" in getattr(self.model.config, "image_aspect_ratio", "square"):
                    inputs['root_anyres'] = False
                conf_root = self.get_confidence_value(**inputs)
                if show_value:
                    print(target, conf_root)
                if conf_root < answering_confidence_threshold_upper:
                    ret.append(target)
        else:
            ret = targets[:]
        return ret


class ModelLocal(Model):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.side_threshold = kwargs.get("side_threshold", 1000)

    def init_prompts(self):
        self.prompts = {
            "local": {
                "pre_information": f"{DEFAULT_IMAGE_TOKEN}\n",
                "latent_prompt": "According to your common sense knowledge and the content of the image, is it possible to find a {} in the image? Answer Yes or No and tell the reason.",
                "existence_prompt": "Is there a {} in the image? Answer Yes or No.",
                "answering_prompt": "Question: {}\nCould you answer the question based on the the available visual information? Answer Yes or No.",
            }
        }

    def get_bbox_in_square_image(self, bbox, left, top):
        x1, y1, x2, y2 = bbox
        return [x1 + left, y1 + top, x2 + left, y2 + top]

    def process_nodes_to_image_list(self, nodes: List[Node], image_pil: Image.Image):
        square_image, left, top = expand2square(image_pil, self.background_color)
        if len(nodes) == 0 or (len(nodes) == 1 and nodes[0].is_root):
            return [square_image.resize(self.input_size)]

        resized_bboxes = [
            self.get_patch(node.state.bbox, image_pil.width, image_pil.height, patch_size=self.input_size[0],
                           patch_scale=self.patch_scale) for node in nodes]
        resized_bboxes = merge_bbox_list(resized_bboxes)
        resized_bboxes = [self.get_bbox_in_square_image(bbox, left, top) for bbox in resized_bboxes]
        union_bbox = union_all_bboxes(resized_bboxes)
        w, h = union_bbox[2] - union_bbox[0], union_bbox[3] - union_bbox[1]
        if max(w, h) > self.side_threshold and len(resized_bboxes) == 2:
            result = union_blocks_independent(square_image, resized_bboxes[0], resized_bboxes[1], self.input_size[0],
                                              self.background_color)
        else:
            result = square_image.crop(union_bbox)

        return [result]

    def get_prompt_tag(self, image_list):
        return "local"

    def process_image_list_to_tensor(self, image_list):
        image_tensor = process_images(image_list, self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=self.dtype, device=self.device) for _image in image_tensor]
        image_sizes = [image.size for image in image_list]
        return image_tensor, image_sizes



class ModelGlobalLocal(Model):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.set_anyres_num(kwargs.get("max_anyres_num", 12))

    def set_anyres_num(self, max_anyres_num):
        patch_size = self.input_size[0]
        new_possible_resolutions = []
        for p in self.model.config.image_grid_pinpoints:
            x = p[0] // patch_size
            y = p[1] // patch_size
            if x * y <= max_anyres_num:
                new_possible_resolutions.append(p)
        self.model.config.image_grid_pinpoints = new_possible_resolutions
        print("new_image_grid_pinpoints:", self.model.config.image_grid_pinpoints)
        self.max_anyres_num = max_anyres_num

    def init_prompts(self):
        self.prompts = {
            "search": {
                "pre_information_image2": f"{DEFAULT_IMAGE_TOKEN}\nThis is the main image.\n{DEFAULT_IMAGE_TOKEN}\nThis is the zoomed-in view of the focus region.\n",
                "pre_information_image1": f"{DEFAULT_IMAGE_TOKEN}\n",
                "latent_prompt": "According to your common sense knowledge and the content of image, is it possible to find a {} by further zooming in the image? Answer Yes or No.",
                "existence_prompt": "Is there a '{}' in the image? Answer Yes or No.",
                "answering_prompt": "Question: '{}'\nCould you answer the question based on the available visual information? Answer Yes or No.",
            },
        }

    def is_root_only(self, nodes: Optional[List[Node]]):
        return bool(nodes and len(nodes) == 1 and getattr(nodes[0], "is_root", False))

    def include_root(self, nodes: Optional[List[Node]]):
        return bool(nodes and any(getattr(node, "is_root", False) for node in nodes))

    def get_bbox_in_square_image(self, bbox, left, top):
        x1, y1, x2, y2 = bbox
        return [x1 + left, y1 + top, x2 + left, y2 + top]

    def get_prompt_tag(self, image_list):
        if len(image_list) == 1:
            prompt_tag = "global"
        elif len(image_list) == 2:
            prompt_tag = "zoom"
        else:
            raise ValueError
        return prompt_tag

    def process_image_list_to_tensor(self, image_list):
        image_tensor = process_images(image_list, self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=self.dtype, device=self.device) for _image in image_tensor]
        # (2, 3, 384, 384) --> (1, 3, 384, 384)
        image_tensor = [_image[[0], ...] if _image.shape[0] == 2 else _image for _image in image_tensor]
        image_sizes = [image.size for image in image_list]
        return image_tensor, image_sizes

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
            return [deepcopy(image_pil)]
        if self.is_root_only(nodes):
            return [deepcopy(image_pil)] if root_anyres else [square_image.resize(self.input_size)]
        if self.include_root(nodes):
            return [deepcopy(image_pil)]

        resized_bboxes = []
        for node in nodes:
            source = getattr(node, 'search_source', 'fine')
            if source == 'fast':
                current_patch_size = self.view_size
                current_patch_scale = None
            else:
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

        return [square_image.resize(self.input_size), croped_view, zoomed_view]

    @torch.no_grad()
    def get_confidence_value(
        self,
        node: Optional[List[Node]] = None,
        image_pil: Image.Image = None,
        confidence_type: str = 'existence',
        input_ele: str = '',
        root_anyres: bool = True,
        node_list: Optional[List[Node]] = None,
        **kwargs
    ):
        assert confidence_type in ['existence', 'latent', 'answering']
        actual_nodes = node if node is not None else (node_list if node_list is not None else [])
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

        image_tensor, image_sizes = self.process_image_list_to_tensor(image_input)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        outputs = self.model(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            modalities=["image"] * len(input_ids),
            return_dict=True
        )
        return self._cal_confidence(outputs)

    @torch.no_grad()
    def _cal_confidence(self, outputs):
        logits_yesno = outputs.logits[0, -1, [self.index_yes, self.index_no]]
        confidence = torch.softmax(logits_yesno, dim=-1)[0]
        confidence = 2 * (confidence.item() - 0.5)  # [-1, 1]
        return confidence

    @torch.inference_mode()
    def free_form_using_nodes(self, image_pil, question, searched_nodes: List[Node] = None, return_zoomed_view=False, max_new_tokens: int = 128):
        if searched_nodes is None:
            searched_nodes = []
        if isinstance(image_pil, (list, tuple)):
            image_input = list(image_pil)
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

        if question.startswith("<image>\n") or question.startswith(DEFAULT_IMAGE_TOKEN):
            qs = question
        else:
            qs = self.prompts[prompt_tag][pre_information] + question
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
        outputs = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        if return_zoomed_view:
            response = {'text': outputs, 'output_image': image_list[1] if len(image_list) > 1 else image_list[0]}
            return response

        return outputs

    free_form_using_nodes_sgavs = free_form_using_nodes

    @torch.inference_mode()
    def generate_response(self, image_pil=None, question: str = "", max_new_tokens: int = 128, **kwargs) -> str:
        """Direct free-form text response generation interface for single image, crop, or text-only query."""
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
        """Unified batched confidence probing for ModelGlobalLocal."""
        assert confidence_type in ['existence', 'latent', 'answering']
        if not image_list:
            return []

        n_imgs = len(image_list)
        if isinstance(input_ele, (list, tuple)):
            queries = list(input_ele)
            if len(queries) < n_imgs:
                queries.extend([queries[-1] if queries else ""] * (n_imgs - len(queries)))
        else:
            queries = [str(input_ele)] * n_imgs

        if custom_prompt_templates is not None:
            if isinstance(custom_prompt_templates, (list, tuple)):
                templates = list(custom_prompt_templates)
                if len(templates) < n_imgs:
                    templates.extend([templates[-1] if templates else None] * (n_imgs - len(templates)))
            else:
                templates = [custom_prompt_templates] * n_imgs
        else:
            templates = [None] * n_imgs

        results: List[float] = []
        for img, q_str, tmpl in zip(image_list, queries, templates):
            orig_prompt = None
            if tmpl and hasattr(self, "prompts") and "search" in self.prompts:
                orig_prompt = self.prompts["search"].get(f"{confidence_type}_prompt")
                self.prompts["search"][f"{confidence_type}_prompt"] = tmpl
            try:
                conf = self.get_confidence_value(
                    node=None,
                    image_pil=img,
                    confidence_type=confidence_type,
                    input_ele=q_str
                )
                if hasattr(conf, "item"):
                    conf = conf.item()
                results.append(float(conf))
            finally:
                if orig_prompt is not None:
                    self.prompts["search"][f"{confidence_type}_prompt"] = orig_prompt

        return results

    @torch.inference_mode()
    def batch_generate_response(
        self,
        image_list: List[Image.Image],
        questions: Union[str, List[str]],
        max_new_tokens: int = 128,
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """Unified sequential/batched response generation for ModelGlobalLocal."""
        if not image_list:
            return []
        n_imgs = len(image_list)
        if isinstance(questions, (list, tuple)):
            q_list = list(questions)
            if len(q_list) < n_imgs:
                q_list.extend([q_list[-1] if q_list else ""] * (n_imgs - len(q_list)))
        else:
            q_list = [str(questions)] * n_imgs

        return [
            self.generate_response(image_pil=img, question=q, max_new_tokens=max_new_tokens)
            for img, q in zip(image_list, q_list)
        ]

    @torch.inference_mode()
    def multiple_choices_inference(self, image_pil, question, options, searched_nodes: List[Node] = None):
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

        qs = self.prompts[prompt_tag][pre_information] + question
        prompt = self.get_prompt_from_qs(qs)
        image_tensor, image_sizes = self.process_image_list_to_tensor(image_input)

        question_input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).to(self.device)
        output_question = self.model(
            question_input_ids,
            use_cache=True,
            images=image_tensor,
            image_sizes=image_sizes)

        question_logits = output_question.logits
        question_past_key_values = output_question.past_key_values

        loss_list = []

        for option in options:
            full_prompt = self.get_prompt_from_qs(qs, option)
            full_input_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).to(self.device)
            option_answer_input_ids = full_input_ids[:, question_input_ids.shape[1]:]

            output_option = self.model(input_ids=option_answer_input_ids,
                                       use_cache=True,
                                       attention_mask=torch.ones(1, question_logits.shape[1] +
                                                                 option_answer_input_ids.shape[1],
                                                                 device=full_input_ids.device),
                                       past_key_values=question_past_key_values)

            logits = torch.cat([question_logits[:, -1:], output_option.logits[:, :-1]], 1)

            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, self.model.config.vocab_size)
            labels = option_answer_input_ids.view(-1)
            loss = loss_fct(logits, labels)
            loss_list.append(loss)

        option_chosen = torch.stack(loss_list).argmin()

        return option_chosen.cpu().item()
