"""
Unified Vision Extractor for MSSearch ablation suite.
Extracts standardized (C, H, W) spatial feature maps and object detections
for 'grounding_dino', 'owlv2', or 'florence2'.
"""

import math
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import numpy as np
import torch


class UnifiedVisionExtractor:
    """
    Extracts standardized (C, H, W) spatial feature maps and object detections
    for 'grounding_dino', 'owlv2', or 'florence2'.
    """
    def __init__(self, model_type: str, model_id: str = None, device: str = None):
        self.model_type = model_type.lower()
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.model_type == "grounding_dino":
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self.model_id = model_id or "IDEA-Research/grounding-dino-base"
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device).eval()

        elif self.model_type == "owlv2":
            from transformers import Owlv2Processor, Owlv2ForObjectDetection
            self.model_id = model_id or "google/owlv2-base-patch16-ensemble"
            self.processor = Owlv2Processor.from_pretrained(self.model_id)
            self.model = Owlv2ForObjectDetection.from_pretrained(self.model_id).to(self.device).eval()

        elif self.model_type == "florence2":
            from transformers import AutoProcessor, AutoModelForImageTextToText
            self.model_id = model_id or "florence-community/Florence-2-large"
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = AutoModelForImageTextToText.from_pretrained(self.model_id).to(self.device).eval()
            except Exception:
                from transformers import AutoModelForCausalLM
                self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True).to(self.device).eval()

        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'grounding_dino', 'owlv2', or 'florence2'.")

        print(f"[UnifiedVisionExtractor] Loaded {self.model_type} ({self.model_id}) on {self.device}")

    @torch.inference_mode()
    def run_inference(self, image: Image.Image, text_prompts: list, threshold: float = 0.3):
        w, h = image.size
        detections = {prompt: {"boxes": np.empty((0, 4)), "scores": np.empty((0,))} for prompt in text_prompts}

        if self.model_type == "grounding_dino":
            caption = ". ".join([p.strip().rstrip(".") for p in text_prompts]) + "."
            inputs = self.processor(images=image, text=caption, return_tensors="pt").to(self.device)

            # Extract Backbone Feature Map (B, C, H, W)
            vision_outputs = self.model.model.backbone(inputs.pixel_values, inputs.pixel_mask)
            feat_map = vision_outputs[0][1][0]  # Scale index 1 -> (1, 256, H_f, W_f)
            feat = feat_map.squeeze(0).cpu().numpy()

            # Extract Detections
            outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=threshold,
                text_threshold=threshold,
                target_sizes=[(h, w)]
            )[0]

            for box, score, label in zip(results["boxes"].cpu().numpy(), results["scores"].cpu().numpy(), results["labels"]):
                for p in text_prompts:
                    if p.lower() in label.lower():
                        detections[p]["boxes"] = np.vstack([detections[p]["boxes"], box])
                        detections[p]["scores"] = np.append(detections[p]["scores"], score)

        elif self.model_type == "owlv2":
            inputs = self.processor(text=[text_prompts], images=image, return_tensors="pt").to(self.device)

            # Extract ViT Patch Tokens & Reshape to (C, H, W)
            vision_outputs = self.model.owlv2.vision_model(pixel_values=inputs.pixel_values)
            patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]  # Remove [CLS] token
            _, N, C = patch_tokens.shape
            grid_size = int(math.sqrt(N))
            feat = patch_tokens.permute(0, 2, 1).reshape(C, grid_size, grid_size).cpu().numpy()

            # Extract Detections
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([[h, w]], device=self.device)
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=threshold
            )[0]

            for box, score, label_idx in zip(results["boxes"].cpu().numpy(), results["scores"].cpu().numpy(), results["labels"].cpu().numpy()):
                prompt_name = text_prompts[int(label_idx)]
                detections[prompt_name]["boxes"] = np.vstack([detections[prompt_name]["boxes"], box])
                detections[prompt_name]["scores"] = np.append(detections[prompt_name]["scores"], score)

        elif self.model_type == "florence2":
            prompt_text = "<CAPTION_TO_PHRASE_GROUNDING>" + ", ".join(text_prompts)
            inputs = self.processor(text=prompt_text, images=image, return_tensors="pt").to(self.device)

            # Extract DaViT Spatial Features & Reshape to (C, H, W)
            vision_feats = self.model._encode_image(inputs.pixel_values)
            _, N, C = vision_feats.shape
            grid_size = int(math.sqrt(N))
            feat = vision_feats.permute(0, 2, 1).reshape(C, grid_size, grid_size).cpu().numpy()

            # Extract Detections
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = self.processor.post_process_generation(
                generated_text, task="<CAPTION_TO_PHRASE_GROUNDING>", image_size=(w, h)
            )["<CAPTION_TO_PHRASE_GROUNDING>"]

            for box, label in zip(parsed.get("bboxes", []), parsed.get("labels", [])):
                for p in text_prompts:
                    if p.lower() in label.lower():
                        detections[p]["boxes"] = np.vstack([detections[p]["boxes"], np.array(box)])
                        detections[p]["scores"] = np.append(detections[p]["scores"], 1.0)

        return feat, detections
