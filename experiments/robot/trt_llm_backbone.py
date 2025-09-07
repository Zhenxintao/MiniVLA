# prismatic/vla/llm_backbones/trt_http_llm_backbone.py

from __future__ import annotations
from typing import Callable, List, Optional, Sequence, Type, Union, Dict, Any
import os, json
import requests
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase
from transformers import AutoTokenizer

try:
    from prismatic.models.backbones.llm.base_llm import LLMBackbone
except Exception:
    from prismatic.models.backbones.llm.base_llm import LLMBackbone  # type: ignore[assignment]

DEFAULT_LLM_BASE_URL = os.getenv("TRTLLM_BASE_URL", "http://127.0.0.1:8810")


class TRTHttpLLMBackbone(LLMBackbone):
    """
    Delegates LLM decoding to the TensorRT-LLM service via HTTP.

    Provides:
    - generate_ids(...) : Plain text generation, via /llm/generate_ids
    - generate_mm_ids(...) : Multimodal (soft prompt prompt_table injects visual embeddings), via /llm/generate_mm

    Conventions:
    - The visual embeddings in generate_mm_ids are projected from the upper layer (VLM) to the LLM hidden dimension H, with shape [Nv, H] or [1, Nv, H].
    - H must be equal to the engine hidden_size; Nv cannot exceed the max_prompt_embedding_table_size set during engine build.
    - This class is inference-only by default.
    """

    def __init__(
        self,
        llm_backbone_id: str,
        tokenizer: PreTrainedTokenizerBase,
        base_url: str = DEFAULT_LLM_BASE_URL,
        default_max_new_tokens: int = 64,
        timeout_s: float = 60.0,
        delegate: Optional[LLMBackbone] = None,  # Can inject original backbone as fallback
        tok_dir_path: Optional[str] = None,   
    ) -> None:
        super().__init__(llm_backbone_id)
        self._tokenizer: PreTrainedTokenizerBase = tokenizer
        self.base_url = base_url.rstrip("/")
        self.default_max_new_tokens = int(default_max_new_tokens)
        self.timeout_s = float(timeout_s)

        self._delegate = delegate

        self._prompt_builder_fn = getattr(delegate, "prompt_builder_fn", None)
        self._transformer_layer_cls = getattr(delegate, "transformer_layer_cls", None)
        self._half_dtype = getattr(delegate, "half_precision_dtype", torch.float16)
        self._last_layer_modules = getattr(delegate, "last_layer_finetune_modules", tuple())
        self.tok_dir_path = tok_dir_path  

        self._caps = self._fetch_caps()
        self._hidden_size = int(self._caps.get("hidden_size", -1))
        self._prompt_table_cap = int(self._caps.get("prompt_table_cap", 0))

    # -------------------------------------------------------------------------
    # Base class/meta information bridge
    # -------------------------------------------------------------------------
    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def prompt_builder_fn(self) -> Type:
        if self._prompt_builder_fn is None:
            # For most projects, LLMBackbone will set this; if not, throw a clearer error message.
            raise RuntimeError(
                "TRTHttpLLMBackbone needs `delegate.prompt_builder_fn`. "
                "Use `TRTHttpLLMBackbone.from_existing(old_backbone, ...)` to carry metadata."
            )
        return self._prompt_builder_fn

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return self._transformer_layer_cls or nn.TransformerEncoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self._half_dtype

    @property
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        return self._last_layer_modules or tuple()

    # ---- engine capability ----
    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def prompt_table_cap(self) -> int:
        return self._prompt_table_cap

    @property
    def supports_prompt_table(self) -> bool:
        return self._hidden_size > 0 and self._prompt_table_cap > 0

    @classmethod
    def from_existing(
        cls,
        old_backbone: LLMBackbone,
        base_url: str = "http://127.0.0.1:8810",
        tok_dir_path: Optional[str] = None,       
    ) -> "TRTHttpLLMBackbone":
        tok = None
        if hasattr(old_backbone, "get_tokenizer"):
            try:
                tok = old_backbone.get_tokenizer()
            except Exception:
                tok = None
        if tok is None:
            tok = getattr(old_backbone, "tokenizer", None)

        if tok is None:
            if not tok_dir_path:
                raise RuntimeError(
                    "TRTHttpLLMBackbone.from_existing: The original backbone does not contain a tokenizer;"
                    "Please pass tok_dir_path='/abs/path/to/tokenizer' (avoid using environment variables)."
                )
            tok = AutoTokenizer.from_pretrained(tok_dir_path, trust_remote_code=True, use_fast=True)

        return cls(
            llm_backbone_id=getattr(old_backbone, "identifier", "trt-http"),
            tokenizer=tok,
            base_url=base_url,
            delegate=old_backbone,
            tok_dir_path=tok_dir_path, 
        )

    # -------------------------------------------------------------------------
    # Training related (disabled)
    # -------------------------------------------------------------------------
    def get_fsdp_wrapping_policy(self) -> Callable:
        raise NotImplementedError("TRTHttpLLMBackbone is inference-only.")

    def enable_gradient_checkpointing(self) -> None:
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use generate_ids / generate_mm_ids for inference in TRTHttpLLMBackbone.")

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        raise NotImplementedError("TRTHttpLLMBackbone does not host local embeddings.")

    # -------------------------------------------------------------------------
    # Reasoning: Plain Text
    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def generate_ids(
        self,
        input_ids: torch.LongTensor,                      # [S] 或 [B,S]
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_k: int = 1,
        top_p: float = 1.0,
        add_bos: bool = False,
        eos_id: Optional[int] = None,
        pad_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Plain text generation. Shape alignment rule:
        - If input is [S], return [1, T_total]
        - If input is [B, S], return [B, T_total]
        """
        device = input_ids.device
        input_ids = input_ids if input_ids.dim() == 2 else input_ids.unsqueeze(0)  # -> [B,S]
        ids_list = input_ids.detach().cpu().tolist()

        payload: Dict[str, Any] = {
            "input_ids": ids_list,
            "max_new_tokens": int(max_new_tokens or self.default_max_new_tokens),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "eos_id": int(eos_id if eos_id is not None else (self._tokenizer.eos_token_id or -1)),
            "pad_id": int(pad_id if pad_id is not None else (self._tokenizer.pad_token_id or -1)),
            "add_bos": bool(add_bos and getattr(self._tokenizer, "bos_token_id", None) is not None),
        }

        try:
            data = self._post_json("/llm/generate_ids", payload, timeout=self.timeout_s)
            out_ids = data["ids"]  # [B, ...]
            return torch.tensor(out_ids, dtype=torch.long, device=device)
        except Exception as e:
            if self._delegate is not None:
                print(f"[TRT-HTTP][WARN] generate_ids failed, falling back to delegate: {e}")
                return self._delegate.generate_ids(  # type: ignore[attr-defined]
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    add_bos=add_bos,
                    eos_id=eos_id,
                    pad_id=pad_id,
                )
            raise

    # -------------------------------------------------------------------------
    # Reasoning: Multimodality (soft prompt prompt-table injection)
    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def generate_mm_ids(
        self,
        input_ids: torch.LongTensor,            # [1, S] or [S]
        visual_embeds: torch.Tensor,            # [1, Nv, H] or [Nv, H]
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_k: int = 1,
        top_p: float = 1.0,
    ) -> torch.LongTensor:
        # --- normalize input_ids to [1,S] ---
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        elif input_ids.dim() > 2:
            raise ValueError(f"input_ids must be [B,S] or [S], got {tuple(input_ids.shape)}")
        if input_ids.size(0) != 1:
            input_ids = input_ids[:1]

        # --- normalize visual_embeds to [1,Nv,H] ---
        if visual_embeds.dim() == 2:
            visual_embeds = visual_embeds.unsqueeze(0)
        elif visual_embeds.dim() == 1:
            visual_embeds = visual_embeds.view(1, 1, -1)
        elif visual_embeds.dim() > 3:
            B, H = visual_embeds.size(0), visual_embeds.size(-1)
            visual_embeds = visual_embeds.view(B, -1, H)
        if visual_embeds.size(0) != 1:
            visual_embeds = visual_embeds[:1]

        new_len = int(max_new_tokens or self.default_max_new_tokens)

        payload = {
            "input_ids": input_ids[0].detach().cpu().tolist(),
            "visual_embeds": visual_embeds[0].detach().to("cpu", torch.float16).numpy().tolist(),
            "max_new_tokens": new_len,
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "force_no_eos": True,
        }
        url = f"{self.base_url}/llm/generate_mm"
        r = requests.post(url, json=payload, timeout=max(self.timeout_s, 120.0))
        r.raise_for_status()
        out = r.json()
        ids = out["ids"]  # list[int], The server returns the max_new_tokens length (will not be truncated by EOS)

        if len(ids) < new_len:
            ids = ids + [0] * (new_len - len(ids))
        elif len(ids) > new_len:
            ids = ids[-new_len:]

        return torch.tensor([ids], dtype=torch.long, device=input_ids.device)


    # -------------------------------------------------------------------------
    # Internal: HTTP Tools
    # -------------------------------------------------------------------------
    def _fetch_caps(self) -> Dict[str, Any]:
        """调用 /health，探测 hidden_size / prompt_table_cap。"""
        try:
            r = requests.get(f"{self.base_url}/health", timeout=self.timeout_s)
            r.raise_for_status()
            caps = r.json()
            return {
                "hidden_size": int(caps.get("hidden_size", caps.get("H", -1) or -1)),
                "prompt_table_cap": int(caps.get("prompt_table_cap", caps.get("cap", 0) or 0)),
            }
        except Exception as e:
            print(f"[TRT-HTTP][WARN] /health failed ({e}); fallback caps: hidden_size=-1, prompt_table_cap=0")
            return {"hidden_size": -1, "prompt_table_cap": 0}

    def _post_json(self, path: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            r = requests.post(url, json=payload, timeout=timeout)
        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"POST {url} timed out: {e}") from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"POST {url} failed: {e}") from e

        if r.status_code != 200:
            try:
                err = r.json()
            except Exception:
                err = r.text
            raise RuntimeError(f"POST {url} -> HTTP {r.status_code}: {err}")

        try:
            return r.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"POST {url} returned non-JSON: {r.text[:512]}") from e
