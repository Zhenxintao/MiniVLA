# trt_backbone.py
from __future__ import annotations

import base64
import io
import os
from typing import Callable, Dict, List, Tuple, Union, Optional

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image

# === Import the abstract base class/utils from your project ===
from prismatic.models.backbones.vision.base_vision import VisionBackbone, ImageTransform

# === Denormalization parameters consistent with training ===
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def _tensor_chw_to_pil(x: torch.Tensor,
                       mean=_IMAGENET_MEAN,
                       std=_IMAGENET_STD) -> Image.Image:
    """
    Denormalizes a normalized tensor [3,H,W] to PIL RGB.
    Assuming x comes from image_transform (which does the usual process of (x-mean)/std and [0,1]).
    """
    # Ensure CPU and float calculations
    x = x.detach().float().cpu().numpy()              # [3,H,W]
    x = np.transpose(x, (1, 2, 0))                    # [H,W,3]
    x = x * np.array(std, dtype=np.float32) + np.array(mean, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).round().astype(np.uint8)
    return Image.fromarray(x).convert("RGB")


class TRTHttpVisionBackbone(VisionBackbone):
    """
    VisionBackbone implementation that calls the TensorRT service (/vision/encode) over HTTP.

    - Reuse the original backbone's image_transform (injected via from_existing) → Preprocessing is consistent with training
    - Supports single frame [B,3,H,W] and sequence [B,T,3,H,W], and the output is [B,256,2176] / [B,T*256,2176] respectively
    - Assume the server returns {"shape":[1,256,2176], "dtype":"float16", "embedding_b64": "..."}
    """

    def __init__(self,
                 base_url: str,
                 *,
                 identifier: str = "trt-http-vision",
                 image_resize_strategy: str = "resize-crop",
                 image_transform: Optional[ImageTransform] = None,
                 image_sequence_len: int = 1,
                 default_image_size: int = 224,
                 out_num_patches: int = 256,
                 out_embed_dim: int = 2176,
                 half_dtype: torch.dtype = torch.float16,
                 request_timeout: float = 15.0):
        super().__init__(
            vision_backbone_id=identifier,
            image_resize_strategy=image_resize_strategy,
            default_image_size=default_image_size,
            image_sequence_len=image_sequence_len,
        )
        # Reuse upstream transform to ensure consistent pre-processing
        self.image_transform = image_transform

        # Output specifications (same as your TRT engine)
        self._num_patches = int(out_num_patches)
        self._embed_dim   = int(out_embed_dim)
        self._half_dtype  = half_dtype
        self._default_res = (3, default_image_size, default_image_size)

        # HTTP Config
        self.base_url  = base_url.rstrip("/")
        self._timeout  = request_timeout
        self._session  = requests.Session()
        self._require_trt = bool(int(os.environ.get("REQUIRE_TRT", "1")))  # 默认必须成功连上 TRT 服务

        # To satisfy the base class properties, give a placeholder featurizer
        self.featurizer = nn.Identity()

        # Health check (check only, no actual reasoning)
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=2)
            r.raise_for_status()
            h = r.json()
            print(f"[TRT-HTTP] connected to {self.base_url} | health.vision={h.get('vision')}")
        except Exception as e:
            msg = f"[TRT-HTTP] cannot reach {self.base_url}/health: {e}"
            print(msg, flush=True)
            if self._require_trt:
                raise RuntimeError(msg)

    # ---------- Abstract interface implementation ----------
    def get_fsdp_wrapping_policy(self) -> Callable:
        # No special FSDP packaging required
        return lambda module: None

    def forward(self, pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Input:
        - Single frame: [B, 3, H, W] or {"img":[B, 3, H, W]}
        - Sequence: [B, T, 3, H, W] or {"img":[B, T, 3, H, W]}
        Output:
        - Single frame: [B, 256, 2176]
        - Sequence: [B, T*256, 2176]
        """
        if isinstance(pixel_values, dict):
            pixel_values = next(iter(pixel_values.values()))

        assert torch.is_tensor(pixel_values), f"Unsupported pixel_values type: {type(pixel_values)}"

        device = pixel_values.device  # Finally put the result back into the same device as the input
        if pixel_values.ndim == 4:
            # [B,3,H,W]
            B, C, H, W = pixel_values.shape
            assert C == 3, f"Expect 3-channel image, got {C}"
            outs: List[torch.Tensor] = []
            for b in range(B):
                feats = self._encode_one_batch(pixel_values[b])  # [1,256,2176] on CPU-half
                outs.append(feats)
            out = torch.cat(outs, dim=0)  # [B,256,2176]
            return out.to(dtype=self._half_dtype, device=device)

        elif pixel_values.ndim == 5:
            # [B,T,3,H,W]：Encode by frame and then concatenate in the patch dimension (same as the original TimmViTBackbone merging strategy)
            B, T, C, H, W = pixel_values.shape
            assert C == 3, f"Expect 3-channel image, got {C}"
            batch_outs: List[torch.Tensor] = []
            for b in range(B):
                frame_feats: List[torch.Tensor] = []
                for t in range(T):
                    frame_feats.append(self._encode_one_batch(pixel_values[b, t]))  # [1,256,2176]
                # [T,1,256,2176] -> [1, T*256, 2176]
                bt = torch.cat(frame_feats, dim=1)
                batch_outs.append(bt)
            out = torch.cat(batch_outs, dim=0)  # [B, T*256, 2176]
            return out.to(dtype=self._half_dtype, device=device)

        else:
            raise ValueError(f"Unexpected pixel_values.ndim={pixel_values.ndim}")

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self._default_res

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def num_patches(self) -> int:
        return self._num_patches * (self.image_sequence_len if self.image_sequence_len > 1 else 1)

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self._half_dtype

    # ---------- Practical methods ----------
    def _encode_one_batch(self, x3hw: torch.Tensor) -> torch.Tensor:
        """
        x3hw: [3,H,W](normalized output from image_transform )
        Returns: torch.HalfTensor [1, 256, 2176] (assembled on the CPU first, then .to() to the target device by the caller)
        """
        # 1) Denormalization → PNG → base64
        pil = _tensor_chw_to_pil(x3hw, _IMAGENET_MEAN, _IMAGENET_STD)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # 2) POST To Service
        r = self._session.post(
            f"{self.base_url}/vision/encode",
            json={"image_b64": img_b64, "tolist": False},
            timeout=self._timeout,
        )
        r.raise_for_status()
        data = r.json()

        # 3) Strong consistency check (your engine is 1x256x2176)
        shape = data.get("shape", None)
        assert shape == [1, self._num_patches, self._embed_dim], f"Unexpected TRT feature shape: {shape}"

        # 4) Parsing b64 → numpy → torch
        raw = base64.b64decode(data["embedding_b64"])
        arr = np.frombuffer(raw, dtype=np.float16).reshape(1, self._num_patches, self._embed_dim).copy()
        t = torch.from_numpy(arr)  # CPU half
        return t

    # ---------- Factory: clone transform/config based on existing backbone ----------
    @classmethod
    def from_existing(cls,
                      old_backbone: VisionBackbone,
                      base_url: str,
                      *,
                      out_num_patches: int = 256,
                      out_embed_dim: int = 2176,
                      request_timeout: float = 15.0) -> "TRTHttpVisionBackbone":
        return cls(
            base_url=base_url,
            identifier=f"trt::{getattr(old_backbone, 'identifier', 'vision')}",
            image_resize_strategy=getattr(old_backbone, 'image_resize_strategy', 'resize-crop'),
            image_transform=old_backbone.get_image_transform(),
            image_sequence_len=getattr(old_backbone, 'image_sequence_len', 1),
            default_image_size=getattr(old_backbone, 'default_image_size', 224),
            out_num_patches=out_num_patches,
            out_embed_dim=out_embed_dim,
            half_dtype=getattr(old_backbone, 'half_precision_dtype', torch.float16),
            request_timeout=request_timeout,
        )
