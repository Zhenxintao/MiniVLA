import json_numpy
json_numpy.patch()

import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path
from datetime import datetime

# -----------------------------
# GenerateConfig
# -----------------------------
@dataclass
class GenerateConfig:
    model_family: str = "prismatic"
    hf_token: Union[str, None] = None
    pretrained_checkpoint: Union[str, None] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    obs_history: int = 1
    use_wrist_image: bool = False
    task_suite_name: str = "libero_spatial"
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    prefix: str = ""
    use_wandb: bool = False
    wandb_project: str = "prismatic"
    wandb_entity: Optional[str] = None
    seed: int = 7
    unnorm_key: Optional[str] = None

# -----------------------------
# Loading models and utility functions
# -----------------------------
from experiments.robot.robot_utils import (
    get_model,
    get_action,
    set_seed_everywhere,
    invert_gripper_action,
    normalize_gripper_action,
    get_image_resize_size,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.libero.libero_utils import quat2axisangle

# -----------------------------
# Initialize configuration and model
# -----------------------------
cfg = GenerateConfig(
    pretrained_checkpoint="/home/cody/models/minivla-vq-libero90-prismatic/checkpoints/step-150000-epoch-67-loss=0.0934.pt",
    model_family="prismatic",  # or "openvla"
    hf_token="HF_TOKEN",
    task_suite_name="libero_90",
    center_crop=True,
    obs_history=1,
    use_wrist_image=False,
)
set_seed_everywhere(cfg.seed)
cfg.unnorm_key = cfg.task_suite_name

model = get_model(cfg)
processor = get_processor(cfg) if cfg.model_family == "openvla" else None
resize_size = get_image_resize_size(cfg)

# -----------------------------
# FastAPI Initialization
# -----------------------------
app = FastAPI()

# Input format description (compatible with json_numpy)
class InferenceInput(BaseModel):
    image: List[List[List[int]]]  # 3D RGB image
    state: List[float]            # 7D state vector
    instruction: str

@app.post("/act")
async def predict_action(request: Request):
    try:
        payload = await request.json()
        image = np.array(payload["image"], dtype=np.uint8)
        state = np.array(payload["state"], dtype=np.float32)
        instruction = payload["instruction"]

        # --- Save or display the image ---
        pil_image = Image.fromarray(image)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"./debug_images/input_image_{timestamp}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pil_image.save(save_path)

        observation = {
            "full_image": [image],
            "state": state
        }

        action = get_action(cfg, model, observation, instruction, processor=processor)
        action = normalize_gripper_action(action, binarize=True)

        if cfg.model_family in ["openvla", "prismatic"]:
            action = invert_gripper_action(action)
        
        print("[DEBUG] Predicted action:", action)
        return JSONResponse(content={"action": action.tolist()})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
