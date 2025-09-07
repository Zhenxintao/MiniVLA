import torch
import torch.onnx
import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))
from pathlib import Path
from prismatic.models.vlas.openvla import OpenVLA
from prismatic.models.load import load_vla

# ==== Config ====
ckpt_path = "/home/cody/models/minivla-vq-libero90-prismatic/checkpoints/step-150000-epoch-67-loss=0.0934.pt"
onnx_output_path = "./vision_encoder_fp16.onnx"
hf_token = "" # HF-TOKEN
use_half = True  # Export float16

# ==== Load ====
model: OpenVLA = load_vla(
    ckpt_path,
    hf_token=hf_token,
    load_for_training=False,
)
vision_encoder = model.vision_backbone.eval()

# ==== Build the wrapper ====
class VisionWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, dino, siglip):
        return self.encoder({"dino": dino, "siglip": siglip})

# ==== Set device and data type ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if use_half else torch.float32
wrapper = VisionWrapper(vision_encoder).eval().to(device, dtype=dtype)

# ==== ready to enter ====
dummy_dino = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)
dummy_siglip = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)

# ==== Export ONNX ====
torch.onnx.export(
    wrapper,
    (dummy_dino, dummy_siglip),
    onnx_output_path,
    input_names=["dino", "siglip"],
    output_names=["vision_features"],
    opset_version=17,
    dynamic_axes={
        "dino": {0: "batch"},
        "siglip": {0: "batch"},
        "vision_features": {0: "batch"},
    },
    do_constant_folding=True,
)

print(f"The visual module has been successfully exported as ONNX: {onnx_output_path}")
