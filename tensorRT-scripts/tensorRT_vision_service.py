#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI HTTP server for:
1) Visual Encoding Inference using the TensorRT engine (VISION_ENGINE)

Dependencies (example):

    pip install fastapi uvicorn pydantic[dotenv] pillow numpy transformers

    # CUDA/TensorRT environment requires the following: tensorrt, pycuda (or cuda-python) runtime installed

Start:
    python server.py # or uvicorn server:app --host 0.0.0.0 --port 8000

Test:
    curl -X POST http://localhost:8000/vision/encode \
    -H 'Content-Type: application/json' \
    -d '{"image_b64":"<your_base64_image>"}'
"""

import base64
import io
import json
import os
import sys
import time
from typing import Optional, List, Dict, Any

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ====== Config ======
VISION_ENGINE = "/home/cody/tensorRT/vision_encoder_fp16.engine"

# ============ TensorRT & CUDA ============
TRT_AVAILABLE = True
try:
    import tensorrt as trt
    def _print_trt_env():
        try:
            import tensorrt as trt
            print(f"[TRT] Python package version = {trt.__version__}")
            print(f"[TRT] Python package file    = {trt.__file__}")
        except Exception as e:
            print(f"[TRT] cannot import tensorrt: {e}")

        # Print the nvinfer dynamic library path used by the current process
        try:
            import ctypes, ctypes.util
            path = ctypes.util.find_library("nvinfer")
            print(f"[TRT] ctypes.find_library('nvinfer') = {path}")
            if path:
                lib = ctypes.CDLL(path)
                print(f"[TRT] Loaded libnvinfer via ctypes: {lib._name}")
        except Exception as e:
            print(f"[TRT] probing nvinfer via ctypes failed: {e}")
except Exception as e:
    TRT_AVAILABLE = False
    trt = None
    print("[WARN] tensorrt not available", e, file=sys.stderr)

CUDART_AVAILABLE = True
try:
    from cuda.bindings import runtime as cudart
except Exception as e:
    CUDART_AVAILABLE = False
    cudart = None
    print("[WARN] cuda-python (cudart) not available", e, file=sys.stderr)

# ============ TensorRT-LLM Runtime ============
TRT_LLM_AVAILABLE = True
RUNNER_IMPL = None
try:
    from tensorrt_llm.runtime import ModelRunner as TRTLLMModelRunner
    RUNNER_IMPL = "ModelRunner"
except Exception:
    try:
        from tensorrt_llm.runtime import GenerationSession as TRTLLMGenerationSession  # type: ignore
        RUNNER_IMPL = "GenerationSession"
    except Exception as e:
        TRT_LLM_AVAILABLE = False
        print("[WARN] tensorrt-llm runtime not available:", e, file=sys.stderr)

# ============ Tokenizer ============
TOKENIZER_AVAILABLE = True
try:
    from transformers import AutoTokenizer
except Exception as e:
    TOKENIZER_AVAILABLE = False
    print("[WARN] transformers not available:", e, file=sys.stderr)


# -------------------- General Tools --------------------

def _cuda_err_code(ret) -> int:
    """
    Convert cuda-python return values ​​to int error codes:
    - May be int
    - May be (<cudaError_t.cudaSuccess: 0>,)
    - May also be <cudaError_t.cudaSuccess: 0>
    """
    # Tuple: Take the first
    if isinstance(ret, tuple):
        ret = ret[0]
    # Enumeration: Get .value
    if hasattr(ret, "value"):
        try:
            return int(ret.value)
        except Exception:
            pass
    # Number
    try:
        return int(ret)
    except Exception:
        return 0  



def _img_to_nchw_fp(image: Image.Image, size: int = 224,
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                    use_fp16: bool = True) -> np.ndarray:
    """Simple preprocessing: Resize->CenterCrop (proportional to the short side size)->NCHW->Normalize->FP16/FP32"""
    # Resize to the shortest side size, then crop the center to size x size
    w, h = image.size
    scale = size / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    image = image.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    image = image.crop((left, top, left + size, top + size))

    arr = np.asarray(image).astype(np.float32) / 255.0  # HWC, [0,1]
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = (arr - np.array(mean)) / np.array(std)
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # NCHW
    if use_fp16:
        arr = arr.astype(np.float16)
    return arr

# -------------------- TensorRT Vision Engine--------------------
class TRTImageEncoder:

    def __init__(self, engine_path: str):
        if not TRT_AVAILABLE or not CUDART_AVAILABLE:
            raise RuntimeError("Requires tensorrt + cuda-python (cudart) environment")
        if not os.path.exists(engine_path):
            raise FileNotFoundError(engine_path)

        logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Deserializing the TensorRT engine failed")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create IExecutionContext")

        # Collect I/O tensor names
        self.input_tensors: List[str] = []
        self.output_tensors: List[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)  # trt.TensorIOMode.{INPUT, OUTPUT}
            if mode == trt.TensorIOMode.INPUT:
                self.input_tensors.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_tensors.append(name)

        if not self.input_tensors or not self.output_tensors:
            raise RuntimeError("Unable to identify input/output tensors")

        expected = {"dino", "siglip"}
        if not expected.issubset(set(self.input_tensors)):
            print(f"[WARN] Engine input does not include {expected}，Actually: {self.input_tensors}", file=sys.stderr)

        # Default dynamic shape
        self.default_shape = (1, 3, 224, 224)
        for name in self.input_tensors:
            try:
                self.context.set_input_shape(name, self.default_shape)
            except Exception:
                pass

    @staticmethod
    def _np_dtype_from_trt_dtype(t) -> np.dtype:
        # Compatible with different enumeration constant names
        try:
            if t == trt.float32 or int(t) == int(trt.float32):
                return np.float32
        except Exception:
            pass
        try:
            if t == trt.float16 or int(t) == int(trt.float16):
                return np.float16
        except Exception:
            pass
        try:
            if t == trt.int8 or int(t) == int(trt.int8):
                return np.int8
        except Exception:
            pass
        try:
            if t == trt.int32 or int(t) == int(trt.int32):
                return np.int32
        except Exception:
            pass
        if hasattr(trt, "int64") and (t == trt.int64 or (hasattr(t, "__int__") and int(t) == int(trt.int64))):
            return np.int64
        if hasattr(trt, "bool") and (t == trt.bool or (hasattr(t, "__int__") and int(t) == int(trt.bool))):
            return np.bool_
        # Default
        return np.float32

    def infer(self, image_np: np.ndarray) -> np.ndarray:
        # 1) Set the dynamic shape of each input
        target_shape = tuple(image_np.shape)
        for name in self.input_tensors:
            cur = tuple(self.context.get_tensor_shape(name))
            if cur != target_shape:
                self.context.set_input_shape(name, target_shape)
                cur = tuple(self.context.get_tensor_shape(name))
                if cur != target_shape:
                    raise RuntimeError(f"Failed to set shape of input {name}: want={target_shape}, got={cur}")

        # 2) Prepare Host input (ensure dtype matching + C-contiguous)
        host_in: Dict[str, np.ndarray] = {}
        for name in self.input_tensors:
            np_dtype = self._np_dtype_from_trt_dtype(self.engine.get_tensor_dtype(name))
            arr = np.ascontiguousarray(image_np.astype(np_dtype, copy=False))
            host_in[name] = arr

        # 3) Infer output shape (after setting input shape)
        out_name = self.output_tensors[0]
        out_shape = tuple(self.context.get_tensor_shape(out_name))
        if any(d < 0 for d in out_shape):
            raise RuntimeError(f"The output {out_name} has an unknown shape: {out_shape}")
        out_np_dtype = self._np_dtype_from_trt_dtype(self.engine.get_tensor_dtype(out_name))
        host_out = np.empty(out_shape, dtype=out_np_dtype)

        # 4) CUDA resources: stream + device buffers
        # Create stream
        err_stream = cudart.cudaStreamCreate()
        if _cuda_err_code(err_stream) != 0:
            raise RuntimeError(f"cudaStreamCreate Fail, err={err_stream}")
        _, stream = err_stream  # cudaStreamCreate return (err, stream)

        d_in_ptrs: Dict[str, int] = {}
        try:
            # Output
            err_alloc_out = cudart.cudaMalloc(host_out.nbytes)
            if _cuda_err_code(err_alloc_out) != 0:
                raise RuntimeError(f"cudaMalloc(d_out,{host_out.nbytes}) fail, err={err_alloc_out}")
            _, d_out = err_alloc_out
            self.context.set_tensor_address(out_name, int(d_out))

            # Input + Asyn Copy
            for name, arr in host_in.items():
                err_alloc_in = cudart.cudaMalloc(arr.nbytes)
                if _cuda_err_code(err_alloc_in) != 0:
                    raise RuntimeError(f"cudaMalloc({name},{arr.nbytes}) fail, err={err_alloc_in}")
                _, dptr = err_alloc_in
                d_in_ptrs[name] = dptr

                err_copy_h2d = cudart.cudaMemcpyAsync(
                    dptr, arr.ctypes.data, arr.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream
                )
                if _cuda_err_code(err_copy_h2d) != 0:
                    raise RuntimeError(f"cudaMemcpyAsync(H2D,{name}) fail, err={err_copy_h2d}")

                self.context.set_tensor_address(name, int(dptr))

            # 5) Execute
            ok = self.context.execute_async_v3(stream)
            if not ok:
                raise RuntimeError("context.execute_async_v3 False")

            # Input D2H
            err_copy_d2h = cudart.cudaMemcpyAsync(
                host_out.ctypes.data, d_out, host_out.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream
            )
            if _cuda_err_code(err_copy_d2h) != 0:
                raise RuntimeError(f"cudaMemcpyAsync(D2H,output) fail, err={err_copy_d2h}")

            err_sync = cudart.cudaStreamSynchronize(stream)
            if _cuda_err_code(err_sync) != 0:
                raise RuntimeError(f"cudaStreamSynchronize fail, err={err_sync}")

            return host_out
        finally:
            try:
                for p in d_in_ptrs.values():
                    cudart.cudaFree(p)
            except Exception:
                pass
            try:
                if 'd_out' in locals():
                    cudart.cudaFree(d_out)
            except Exception:
                pass
            try:
                cudart.cudaStreamDestroy(stream)
            except Exception:
                pass

# -------------------- FastAPI --------------------
class VisionReq(BaseModel):
    image_b64: str  # Base64-encoded RGB image (jpg/png)
    tolist: bool = False  # If True, returns the embedding as a list; otherwise, returns the base64 binary.

class InferReq(BaseModel):
    prompt: str
    image_b64: Optional[str] = None
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9


# -------------------- App --------------------
app = FastAPI(title="OpenVLA-Mini TensorRT Service",
              version="0.1.0",
              description="Vision Encoder(TensorRT)")

_vision: Optional[TRTImageEncoder] = None

def lazy_init() -> Dict[str, Any]:
    global _vision
    status = {"vision": None, "llm": None}
    # Vision
    if _vision is None and os.path.exists(VISION_ENGINE) and TRT_AVAILABLE and CUDART_AVAILABLE:
        try:
            _vision = TRTImageEncoder(VISION_ENGINE)
            status["vision"] = "ready"
        except Exception as e:
            _vision = None
            status["vision"] = f"error: {e}"
    else:
        status["vision"] = "missing or not available"

    return status


@app.get("/health")
def health() -> Dict[str, Any]:
    status = lazy_init()
    return {
        "ok": (status.get("vision") == "ready"),
        "vision": status.get("vision"),
        "llm": status.get("llm"),
        "paths": {"VISION_ENGINE": VISION_ENGINE},
    }


@app.post("/vision/encode")
def vision_encode(req: VisionReq) -> Dict[str, Any]:
    status = lazy_init()
    if not isinstance(_vision, TRTImageEncoder):
        raise HTTPException(status_code=500, detail=f"Vision Engine is unavailable: {status.get('vision')}")

    try:
        img_bytes = base64.b64decode(req.image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image_b64")

    inp = _img_to_nchw_fp(img, size=224, use_fp16=True)
    t0 = time.time()
    feat = _vision.infer(inp)
    dt = time.time() - t0

    if req.tolist:
        payload = feat.ravel().tolist()
        return {"shape": list(feat.shape), "dtype": str(feat.dtype), "latency_ms": int(dt * 1000), "embedding": payload}
    else:
        raw = feat.tobytes()
        return {"shape": list(feat.shape), "dtype": str(feat.dtype), "latency_ms": int(dt * 1000), "embedding_b64": base64.b64encode(raw).decode("utf-8")}


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    _print_trt_env()
    uvicorn.run(app, host=host, port=port, reload=False)
