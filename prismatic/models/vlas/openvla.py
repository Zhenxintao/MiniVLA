"""
openvla.py

PyTorch Module defining OpenVLA as a lightweight wrapper around a PrismaticVLM; defines custom logic around
discretizing actions with the ActionTokenizer.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL.Image import Image as Img
from transformers import LlamaTokenizerFast
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.action_tokenizer import ActionTokenizer
from experiments.robot.trt_llm_backbone import TRTHttpLLMBackbone
from transformers import AutoTokenizer
from PIL import Image, ImageOps
import os, time

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class OpenVLA(PrismaticVLM):
    def __init__(
        self,
        *args,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
        action_tokenizer: ActionTokenizer,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.norm_stats = norm_stats
        self.action_tokenizer = action_tokenizer

    @torch.inference_mode()
    def predict_action(
        self, image: Union[Img, List[Img]], instruction: str, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action (de-tokenizes).
        Prioritize TensorRT-LLM HTTP (prompt-tuning to inject visual features); if the conditions are not met, fall back to local super().generate.
        """
        # === 0) Take transform/tokenizer, the same as the original ===
        image_transform = None
        if hasattr(self.vision_backbone, "get_image_transform"):
            try:
                image_transform = self.vision_backbone.get_image_transform()
            except TypeError:
                image_transform = getattr(self.vision_backbone, "get_image_transform", None)
        if image_transform is None:
            image_transform = getattr(self.vision_backbone, "image_transform", None)
        if not callable(image_transform):
            def image_transform(x): return x  

        tokenizer = self.llm_backbone.tokenizer

        tokenizer = getattr(self.llm_backbone, "tokenizer", None)
        if (tokenizer is None) or (not callable(getattr(tokenizer, "__call__", None))):
            tok_dir = getattr(self.llm_backbone, "tok_dir_path", None)
            if not tok_dir:
                raise RuntimeError(
                    "Tokenizer is missing and tok_dir_path is not set;"
                    "Please explicitly pass it in TRTHttpLLMBackbone.from_existing(..., tok_dir_path='/abs/path/to/tokenizer')."
                )
            tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True, use_fast=True)
            self.llm_backbone.tokenizer = tokenizer

        # === 1) Build Prompt ===
        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()


        # === 2) Text ids ===
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (
                        input_ids,
                        torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)
                    ),
                    dim=1
                )
        elif isinstance(tokenizer, Qwen2TokenizerFast):
            pass
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # === 3) Image preprocessing ===
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        action_dim = self.get_action_dim(unnorm_key)

        # === 4) /llm/generate_mm（prompt-tuning）===
        generated_ids = None
        try:
            is_http_llm = isinstance(self.llm_backbone, TRTHttpLLMBackbone)
        except Exception:
            is_http_llm = False
        
        if is_http_llm:
            # 4.1 First run the visual encoder to get visual tokens
            vis_tokens = self.vision_backbone(pixel_values)  
            if isinstance(vis_tokens, (list, tuple)):
                vis_tokens = vis_tokens[0]
            if isinstance(vis_tokens, dict):
                vis_tokens = (
                    vis_tokens.get("last_hidden_state", None)
                    or vis_tokens.get("hidden_states", None)
                    or next((v for v in vis_tokens.values() if isinstance(v, torch.Tensor)), None)
                )
            if not (isinstance(vis_tokens, torch.Tensor) and vis_tokens.dim() == 3):
                raise RuntimeError(f"Unexpected vision features: {type(vis_tokens)} / {getattr(vis_tokens, 'shape', None)}")

            # 4.2 Hidden dimension H cast to LLM
            projector = None
            for name in ("vision_projector", "visual_projector", "mm_projector",
                        "projector", "image_projector", "vision_to_llm"):
                if hasattr(self, name):
                    projector = getattr(self, name)
                    break
            if projector is None and hasattr(self, "vlm") and hasattr(self.vlm, "vision_projector"):
                projector = self.vlm.vision_projector
            if projector is None:
                raise RuntimeError("No projector (vision->LLM) module found on the model.")

            vis_tokens = vis_tokens.to(dtype=self.llm_backbone.half_precision_dtype, device=self.device)
            if hasattr(projector, "to"):
                projector = projector.to(device=vis_tokens.device, dtype=vis_tokens.dtype)
            vis_embeds = projector(vis_tokens)  # [1, Nv, H]
            if not (isinstance(vis_embeds, torch.Tensor) and vis_embeds.dim() == 3):
                raise RuntimeError(f"Projected embeds must be [1,Nv,H], got {type(vis_embeds)} / {getattr(vis_embeds, 'shape', None)}")

            # 4.3 Invoking HTTP LLM：/llm/generate_mm
            generated_ids = self.llm_backbone.generate_mm_ids(
                input_ids=input_ids,          # [1, S]
                visual_embeds=vis_embeds,     # [1, Nv, H]
                max_new_tokens=action_dim,
                temperature=float(kwargs.get("temperature", 0.0)),
                top_k=int(kwargs.get("top_k", 1)),
                top_p=float(kwargs.get("top_p", 1.0)),
            )
        # === 5) Otherwise fall back to the original local super().generate ===
        if generated_ids is None:
            autocast_dtype = self.llm_backbone.half_precision_dtype
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
                generated_ids = super(PrismaticVLM, self).generate(
                    input_ids=input_ids,                       # [1, seq]
                    pixel_values=pixel_values,                 # [1, (opt T,) 3, H, W] or Dict[str, ...]
                    max_new_tokens=action_dim,
                    **kwargs
                )

        # === 6) Post-processing token ===
        predicted_action_token_ids = generated_ids[0, -action_dim:]
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(
            predicted_action_token_ids.detach().cpu().numpy()
        )

        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions
    

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: str) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, please pass a `unnorm_key` from the following "
                f"options to choose the statistics used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        # Error Handling
        assert (
            unnorm_key in norm_stats
        ), f"The `unnorm_key` you chose is not in the set of available statistics; choose from: {norm_stats.keys()}"

        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return self.norm_stats[unnorm_key]["action"]
