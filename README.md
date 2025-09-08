# MiniVLA

<p align="center">
  <a href="https://huggingface.co/xintaozhen/MiniVLA">官方模型仓库 (Hugging Face)</a> |
  <a href="./openvla_mini_reproduction_guide.md">复现环境 (Reproduction Guide)</a> |
  <a href="https://libero-project.github.io/">LIBERO 项目</a>
</p>


MiniVLA is a modular and deployment-friendly Vision-Language-Action (VLA) framework, built on top of [OpenVLA-Mini](https://github.com/Stanford-ILIAD/openvla-mini) and optimized for **edge deployment** with TensorRT acceleration.

---

## 📦 Installation & Reproduction Guide

We provide a complete step-by-step installation and reproduction pipeline for **OpenVLA-Mini + Prismatic + LIBERO90**.  

### Quick Start

```bash
# Create conda environment
conda create -n minivla python=3.10 -y
conda activate minivla

# Install PyTorch + CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Clone repository
git clone https://github.com/Zhenxintao/MiniVLA.git
cd MiniVLA

# Install dependencies
pip install -e .
```

👉 For the **full detailed guide**, including CUDA setup, flash-attn, VQ-VAE, and LIBERO evaluation:  
📄 [Reproduction Guide](./openvla_mini_reproduction_guide.md)

---

## 📂 Model Weights & Checkpoints

All pretrained weights, TensorRT engines, and Hugging Face–compatible Qwen models are hosted on:  

👉 [Hugging Face: xintaozhen/MiniVLA](https://huggingface.co/xintaozhen/MiniVLA)

This includes:

- `models/`: checkpoints from [Stanford-ILIAD/minivla-vq-libero90-prismatic](https://huggingface.co/Stanford-ILIAD/minivla-vq-libero90-prismatic)  
- `qwen25-0_5b-trtllm/`: TensorRT-LLM formatted Qwen-0.5B  
- `tensorRT/`: Vision encoder ONNX & TensorRT engine  

---

## 🏗️ System Architecture

<p align="center">
  <img src="./Results/System_Architecture.svg" width="90%">
</p>


### Hybrid Acceleration

<p align="center">
  <img src="./Results/MiniVLA_Architecture.svg" width="90%">
</p>


---

## 📑 File Overview

This repository extends **OpenVLA-Mini** with TensorRT-based acceleration and modular deployment. The key files are:

### 1. TensorRT Microservices (`tensorRT-scripts/`)

- `tensorRT_llm_service.py` → Runs the **TensorRT-LLM service** for Qwen-0.5B.  
- `tensorRT_vision_service.py` → Runs the **TensorRT Vision Encoder service**.  
  ⚡ These correspond to the **two standalone microservices** in the architecture diagram.

### 2. VLA Deployment Scripts (`vla-scripts/`)

- `deploy_minivla.py` → Launches MiniVLA with a **FastAPI inference service** (`/act`), enabling image + language prompt based inference.  
- `export_vision_encoder_onnx.py` → Exports the vision encoder into **ONNX format** for TensorRT conversion.

### 3. Experiment Framework (`experiments/robot/`)

- `trt_backbone.py` → TensorRT-accelerated **vision backbone** replacement.  
- `trt_llm_backbone.py` → TensorRT-LLM **LLM backbone** replacement.  
- `openvla_utils.py` & `robot_utils.py` → Implement the **Router & Fallback mechanism**, managing local inference vs. accelerated services.

---

## 🔑 Key Contributions

- Built an **end-to-end online inference framework** with a FastAPI service (`/act`), transforming offline benchmark code into a **real-time deployable system**.  
- Reproduced a lightweight **OpenVLA-Mini** and proposed a **hybrid acceleration pipeline**.  
- Exported the **vision encoder** to TensorRT, reducing perception latency and GPU memory usage.  
- Improved **GPU memory efficiency**: reduced average utilization from ~67% to ~43%, and peak usage from ~85% to ~65%, making deployment feasible under 8 GB memory constraints (similar to Jetson-class devices).  
- Integrated **Qwen 2.5 0.5B** in Hugging Face and TensorRT-LLM formats.  
- Designed a **modular system architecture** with router & fallback for robustness.  
- Demonstrated efficient **edge-side VLA inference** on Jetson Orin Nano in LIBERO tasks, with only a moderate performance drop (5–10%).  

---

## 🖥️ Device & Performance

Target deployment: **Jetson Orin Nano (16 GB / 8 GB variants)**.  

For simulation and reproducibility, experiments were conducted on a **local workstation** equipped with:

- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)  
- **Driver / CUDA**: Driver 550.144.03, CUDA 12.4  
- **OS**: Ubuntu 22.04 LTS  

⚠️ **Note**: While the experiments were run on an RTX 4060 Laptop GPU (8 GB VRAM), the device mainly serves as a proxy to evaluate **memory constraints** comparable to Jetson Orin Nano. Absolute inference speed on Jetson devices may be slower due to lower computational power, but the memory utilization trends remain consistent.  

### GPU Memory Utilization (Long-Sequence Tasks)

| Model Variant                           | Avg. GPU Utilization | Peak GPU Utilization |
| --------------------------------------- | -------------------- | -------------------- |
| Original MiniVLA (PyTorch, no TRT)      | ~67%                 | ~85%                 |
| MiniVLA w/ TensorRT Vision Acceleration | ~43%                 | ~65%                 |

---

## 🎬 Results

We evaluated MiniVLA on **LIBERO desktop tasks**. Below are demonstrations:

<table>
<tr>
<td align="center">
  <b>Close the top drawer of the cabinet</b><br>
  <img src="./Results/success_close_the_top_drawer_of_the_cabinet.gif" width="220"><br>
  ✅ Success: 19/20
</td>
<td align="center">
  <b>Put the black bowl in the top drawer of the cabine</b><br>
  <img src="./Results/success_put_the_black_bowl_in_the_top_drawer_of_the_cabine.gif" width="220"><br>
  ✅ Success: 17/20
</td>
<td align="center">
  <b>Turn off the stove</b><br>
  <img src="./Results/success_turn_off_the_stove.gif" width="220"><br>
  ✅ Success: 16/20
</td>
<td align="center">
  <b>Close the bottom drawer of the cabinet</b><br>
  <img src="./Results/success_close_the_bottom_drawer_of_the_cabinet.gif" width="220"><br>
  ✅ Success: 18/20
</td>
<td align="center">
  <b>Close the top drawer of the cabinet and put the black bowl on top of it</b><br>
  <img src="./Results/success_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it.gif" width="220"><br>
  ✅ Success: 17/20
</td>
</tr>

<tr>
<td align="center">
  <b>Put the yellow and white mug to the front of the white mug</b><br>
  <img src="./Results/success_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug.gif" width="220"><br>
  ✅ Success: 18/20
</td>
<td align="center">
  <b>Pick up the black bowl on the left and put it in the tray</b><br>
  <img src="./Results/success_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray.gif" width="220"><br>
  ✅ Success: 19/20
</td>
<td align="center">
  <b>Pick up the book and place it in the left compartment of the caddy</b><br>
  <img src="./Results/success_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy.gif" width="220"><br>
  ✅ Success: 18/20
</td>
<td align="center">
  <b>Pick up the book in the middle and place it on the cabinet shelf</b><br>
  <img src="./Results/success_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf.gif" width="220"><br>
  ✅ Success: 17/20
</td>
<td align="center">
  <b>Pick up the book on the left and place it on top of the shelf</b><br>
  <img src="./Results/success_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf.gif" width="220"><br>
  ✅ Success: 16/20
</td>
</tr>
</table>


---

## 🔗 Related Links

- 📄 Hugging Face weights: [xintaozhen/MiniVLA](https://huggingface.co/xintaozhen/MiniVLA)  
- 🧑‍💻 Base repo: [Stanford-ILIAD/openvla-mini](https://github.com/Stanford-ILIAD/openvla-mini)  
