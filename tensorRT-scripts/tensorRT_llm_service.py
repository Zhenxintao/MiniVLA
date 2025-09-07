#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import torch
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ========= Configuration =========
ENGINE_DIR = os.getenv("TRTLLM_ENGINE_DIR", "/home/cody/openvla-mini/export/qwen25-0_5b-trtllm/engine_v1013_sp256")
TOK_DIR    = os.getenv("TRTLLM_TOK_DIR",    "/home/cody/openvla-mini/export/qwen25-0_5b-trtllm/tokenizer")

# ========= TRT-LLM =========
try:
    from tensorrt_llm.runtime.model_runner import ModelRunner
except Exception as e:
    raise RuntimeError(f"Cannot import TensorRT-LLM: {e}")

# ========= HF tokenizer =========
from transformers import AutoTokenizer

app = FastAPI(title="TRT-LLM Service (prompt-tuning)", version="0.4.1")

_runner: Optional[ModelRunner] = None
_tok   = None


# ----------------------- Pydantic Model -----------------------
class IdsReq(BaseModel):
    input_ids: List[List[int]]             # [B, S]
    max_new_tokens: int = 7
    temperature: float = 0.0
    top_k: int = 1
    top_p: float = 1.0
    eos_id: Optional[int] = None
    pad_id: Optional[int] = None
    add_bos: bool = False
    force_no_eos: bool = False  

class IdsResp(BaseModel):
    ids: List[List[int]]
    time_ms: float

class MMReq(BaseModel):
    input_ids: List[int]              
    visual_embeds: List[List[float]]      
    max_new_tokens: int = 7
    temperature: float = 0.0
    top_k: int = 1
    top_p: float = 1.0
    force_no_eos: bool = True  

class MMResp(BaseModel):
    ids: List[int]
    time_ms: float


# ----------------------- Utility function -----------------------
def _to_padded_batch(seqs: List[List[int]], pad_id: int, add_bos: Optional[int] = None) -> np.ndarray:
    arr = []
    mx = 0
    for s in seqs:
        s2 = ([add_bos] + s) if (add_bos is not None) else s
        arr.append(s2)
        mx = max(mx, len(s2))
    out = np.full((len(arr), mx), int(pad_id), dtype=np.int32)
    for i, s in enumerate(arr):
        out[i, :len(s)] = np.asarray(s, dtype=np.int32)
    return out

def _to_2d_batch_ids(ids_obj) -> List[List[int]]:
    """
    Unify the IDs returned by TRT-LLM into 2D batches:
    - [B, beam, S] -> take beam=0 for each batch -> [B, S]
    - [B, S] -> original -> [B, S]
    - [S] -> wrap one layer -> [[S]]
    """
    t = torch.as_tensor(ids_obj)
    if t.dim() == 3:           # [B, beam, S]
        return t[:, 0, :].to(torch.int32).tolist()
    elif t.dim() == 2:         # [B, S]
        return t.to(torch.int32).tolist()
    elif t.dim() == 1:         # [S]
        return t.unsqueeze(0).to(torch.int32).tolist()
    else:
        raise ValueError(f"Unexpected ids rank {t.dim()} for output_ids")


# ----------------------- Lazy loading -----------------------
def _lazy_init():
    global _runner, _tok
    if _runner is None:
        try:
            if hasattr(ModelRunner, "from_dir"):
                _runner = ModelRunner.from_dir(ENGINE_DIR)
            else:
                _runner = ModelRunner(ENGINE_DIR)
        except Exception as e:
            raise RuntimeError(f"Failed to create ModelRunner from {ENGINE_DIR}: {e}")
        cap = int(getattr(_runner, "max_prompt_embedding_table_size", 0) or 0)
        hid = int(getattr(_runner, "hidden_size", -1))
        print(f"[INIT] ModelRunner ready | hidden_size={hid} | prompt_table_cap={cap}")

    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(TOK_DIR, trust_remote_code=True, use_fast=True)
        print(f"[INIT] Tokenizer ready from {TOK_DIR} "
              f"(pad={_tok.pad_token_id}, eos={_tok.eos_token_id}, bos={getattr(_tok,'bos_token_id',None)})")

    return {
        "runner": "ready" if _runner else "missing",
        "tokenizer": "ready" if _tok else "missing",
        "hidden_size": int(getattr(_runner, "hidden_size", -1)),
        "prompt_table_cap": int(getattr(_runner, "max_prompt_embedding_table_size", 0) or 0),
    }


# ----------------------- Health Check & Warm-up -----------------------
@app.get("/health")
def health():
    s = _lazy_init()
    return {"llm": s["runner"], "tokenizer": s["tokenizer"],
            "hidden_size": s["hidden_size"], "prompt_table_cap": s["prompt_table_cap"]}

@app.on_event("startup")
def _warmup():
    s = _lazy_init()
    if s["runner"] != "ready":
        return
    pad = int(_tok.pad_token_id or (_tok.eos_token_id or 0))
    eos = int(_tok.eos_token_id or pad)

    # text warmup
    try:
        arr = np.full((1, 8), pad, dtype=np.int32)
        bos = int(getattr(_tok, "bos_token_id", pad) or pad)
        arr[0, 0] = bos
        _ = _runner.generate(
            batch_input_ids=torch.from_numpy(arr),
            max_new_tokens=4, temperature=0.0, top_k=1, top_p=1.0,
            end_id=eos, pad_id=pad, return_dict=True,
        )
        print("[WARMUP] text generate OK")
    except Exception as e:
        print(f"[WARMUP][WARN] text: {e}")

    cap = int(getattr(_runner, "max_prompt_embedding_table_size", 0) or 0)
    H   = int(getattr(_runner, "hidden_size", 0) or 0)
    if cap > 0 and H > 0:
        try:
            L = min(4, cap)
            prompt_table = torch.zeros((1, L, H), dtype=torch.float16, device="cuda")
            input_ids = torch.full((1, 4), pad, dtype=torch.int32)
            _ = _runner.generate(
                batch_input_ids=input_ids,
                prompt_table=prompt_table,
                prompt_tasks="0",
                max_new_tokens=4,
                end_id=eos, pad_id=pad,
                temperature=0.0, top_k=1, top_p=1.0,
                return_dict=True,
            )
            print("[WARMUP] prompt_table generate OK")
        except Exception as e:
            print(f"[WARMUP][WARN] prompt_table: {e}")


# ----------------------- Text generation -----------------------
@app.post("/llm/generate_ids", response_model=IdsResp)
def generate_ids(req: IdsReq):
    _lazy_init()
    pad_id = req.pad_id if req.pad_id is not None else (_tok.pad_token_id if _tok.pad_token_id is not None else _tok.eos_token_id)
    eos_id = req.eos_id if req.eos_id is not None else (_tok.eos_token_id if _tok.eos_token_id is not None else pad_id)
    if pad_id is None or eos_id is None:
        raise HTTPException(400, "pad_id/eos_id missing and tokenizer does not define them.")

    add_bos_id = None
    if req.add_bos and getattr(_tok, "bos_token_id", None) is not None:
        add_bos_id = int(_tok.bos_token_id)

    np_batch = _to_padded_batch(req.input_ids, int(pad_id), add_bos_id)
    torch_batch_2d = torch.from_numpy(np_batch.astype(np.int32))

    t0 = time.time()
    try:
        out = _runner.generate(
            batch_input_ids=torch_batch_2d,
            max_new_tokens=int(req.max_new_tokens),
            temperature=float(req.temperature),
            top_k=int(req.top_k),
            top_p=float(req.top_p),
            end_id=int(eos_id),
            pad_id=int(pad_id),
            return_dict=True,
        )
    except Exception as e:
        raise HTTPException(500, f"trtllm.generate failed: {e}")
    dt = (time.time() - t0) * 1000.0

    if isinstance(out, dict):
        ids_obj = out.get("output_ids", None)
        if ids_obj is None:
            ids_obj = out.get("ids", None)
    else:
        ids_obj = getattr(out, "output_ids", None)
        if ids_obj is None:
            ids_obj = getattr(out, "ids", None)
    if ids_obj is None:
        raise HTTPException(500, "TRT-LLM returned no output ids")

    ids = _to_2d_batch_ids(ids_obj)    # Unified to [B, S]
    return IdsResp(ids=ids, time_ms=dt)


# ----------------------- Multimodal Generation -----------------------
@app.post("/llm/generate_mm", response_model=MMResp)
def generate_mm(req: MMReq):
    _lazy_init()

    cap = int(getattr(_runner, "max_prompt_embedding_table_size", 0))
    if cap <= 0:
        raise HTTPException(
            500,
            "Engine was built without soft prompt. Rebuild with --max_prompt_embedding_table_size.",
        )

    pad = int(_tok.pad_token_id or (_tok.eos_token_id or 0))
    eos = int(_tok.eos_token_id or pad)

    if req.force_no_eos:
        vocab_size = int(getattr(_tok, "vocab_size", 0) or 0)
        safe_end_id = max(vocab_size + 1024, 10_000_000)  
    else:
        safe_end_id = eos

    # Vision feature [Nv, H] -> prompt_table [B=1, Nv, H] (fp16, CUDA)
    H = int(getattr(_runner, "hidden_size", 0) or 0)
    vis = np.asarray(req.visual_embeds, dtype=np.float16)
    if vis.ndim != 2:
        raise HTTPException(400, f"visual_embeds must be [Nv, H], got shape {vis.shape}")
    Nv = int(vis.shape[0])
    if H and vis.shape[1] != H:
        raise HTTPException(400, f"hidden size mismatch: visual H={vis.shape[1]} vs engine H={H}")
    if Nv > cap:
        raise HTTPException(400, f"Nv={Nv} exceeds prompt_table_cap={cap}")

    prompt_table = torch.from_numpy(vis).to(device="cuda", dtype=torch.float16).unsqueeze(0).contiguous()
    prompt_tasks = "0"   # B=1

    # text ids
    input_ids = torch.tensor([req.input_ids], dtype=torch.int32)

    # generate
    t0 = time.time()
    try:
        out = _runner.generate(
            batch_input_ids=input_ids,
            max_new_tokens=int(req.max_new_tokens),
            temperature=float(req.temperature),
            top_k=int(req.top_k),
            top_p=float(req.top_p),
            end_id=int(safe_end_id),         
            pad_id=int(pad),
            return_dict=True,
            prompt_table=prompt_table,
            prompt_tasks=prompt_tasks,
        )
    except Exception as e:
        raise HTTPException(500, f"trtllm.generate failed: {e}")
    dt = (time.time() - t0) * 1000.0

    if isinstance(out, dict):
        ids_obj = out.get("output_ids", None)
        if ids_obj is None:
            ids_obj = out.get("ids", None)
    else:
        ids_obj = getattr(out, "output_ids", None)
        if ids_obj is None:
            ids_obj = getattr(out, "ids", None)
    if ids_obj is None:
        raise HTTPException(500, "TRT-LLM returned no output ids")

    if isinstance(ids_obj, torch.Tensor):
        t = ids_obj
    else:
        t = torch.as_tensor(ids_obj)
    if t.dim() == 3:
        seq = t[0, 0].int().tolist()
    elif t.dim() == 2:
        seq = t[0].int().tolist()
    else:
        seq = t.int().tolist()

    need = int(req.max_new_tokens)
    if len(seq) < need:
        seq = seq + [pad] * (need - len(seq))
    elif len(seq) > need:
        seq = seq[-need:]

    return MMResp(ids=seq, time_ms=dt)


# ----------------------- Start -----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8810")), workers=1)
