# 🧠 OpenVLA-Mini + Prismatic + LIBERO90 Reproduction Guide

> Project Source: [Stanford-ILIAD/openvla-mini](https://github.com/Stanford-ILIAD/openvla-mini)  
> Model Repository: [minivla-vq-libero90-prismatic](https://huggingface.co/Stanford-ILIAD/minivla-vq-libero90-prismatic)  
> VQ-VAE Repository: [pretrain_vq](https://huggingface.co/Stanford-ILIAD/pretrain_vq)  
> Guide adapted and maintained by: [Zhenxintao/MiniVLA](https://github.com/Zhenxintao/MiniVLA)  
> Related models and weights: [xintaozhen/MiniVLA](https://huggingface.co/xintaozhen/MiniVLA)

---

## 📦 1. Environment Setup

### ✅ Create Conda Environment
```bash
conda create -n minivla python=3.10 -y
conda activate minivla
```

### ✅ Install Dependencies
```bash
# Step 1: Install PyTorch and CUDA
# Note: safetensors must be version 0.4.3 (safetensors==0.4.3)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Step 2: Clone the main repository
git clone https://github.com/Stanford-ILIAD/openvla-mini.git
cd openvla-mini

# Step 3: Install project dependencies
pip install -e .

# Step 4: Install flash-attn (required for efficient inference)
# If you encounter CUDA_HOME related errors, please see Section 4 on environment variables
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation

# Step 5: Install LIBERO simulation platform
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .

# Step 6: Install LIBERO evaluation dependencies
# ⚠️ Ensure that 'robosuite' version is exactly 1.4.0
cd ../openvla-mini
pip install -r experiments/robot/libero/libero_requirements.txt
```

---

## 📥 2. Download Model Files

### ✅ Download Prismatic Checkpoints

```bash
# Step 1: Create a models folder to store checkpoints
cd /root/models

# Step 2: Use git lfs + huggingface-cli to download the entire repo
sudo apt update
sudo apt install git git-lfs
pip install huggingface_hub

# Login to Hugging Face
huggingface-cli login

git lfs install # only needed once

# '/Stanford-ILIAD/minivla-vq-libero90-prismatic' is one of the pretrained models
# You can browse Hugging Face for others
git clone https://huggingface.co/Stanford-ILIAD/minivla-vq-libero90-prismatic

cd minivla-vq-libero90-prismatic
git lfs pull # download weights
# ⚠️ If TLS handshake fails, try switching to an OpenSSL version of Git
```

### ✅ Download VQ-VAE Model Files

```bash
mkdir -p /root/openvla-mini/vq
huggingface-cli download Stanford-ILIAD/pretrain_vq   --local-dir /root/openvla-mini/vq   --local-dir-use-symlinks False

# If huggingface-cli fails, manually download and upload to server:
# https://huggingface.co/Stanford-ILIAD/pretrain_vq
```

Expected file structure:

```bash
/root/models/minivla-vq-libero90-prismatic/checkpoints/*.pt
/root/openvla-mini/vq/pretrain_vq+mx-libero_90+fach-7+ng-7+nemb-128+nlatent-512/{checkpoints/model.pt, config.json}
```

---

## 🧩 3. Install VQ-VAE Inference Code (Action Tokenizer)

```bash
cd /root/openvla-mini
git clone https://github.com/jayLEE0301/vq_bet_official vqvae
cd vqvae
pip install -e .
```

Ensure structure looks like this (note the `__init__.py`):

```
openvla-mini/
└── vqvae/
    ├── vqvae/
    │   ├── __init__.py
    │   ├── vqvae.py
    │   └── vqvae_utils.py
```

---

## 🔐 4. Environment Variables (if needed)

```bash
huggingface-cli login
# or:
export HF_TOKEN=your_token_here

# Cache & data directories
export PRISMATIC_DATA_ROOT=/root/.cache/prismatic_data
export LIBERO_DATA_ROOT=/root/.cache/libero_data
export HF_HUB_CACHE=/root/.cache/huggingface
export TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets

# CUDA environment variables
echo 'export CUDA_HOME=/usr/local/cuda-12.4' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# CUDA 12.4 download & install
Browser: https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
Terminal: wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-87AE4CD8-keyring.gpg /usr/share/keyrings/

sudo apt update
sudo apt install -y cuda
```

---

## ▶️ 5. Run LIBERO Evaluation Script

```bash
cd /root/openvla-mini

python experiments/robot/libero/run_libero_eval.py   --model_family prismatic   --pretrained_checkpoint /home/cody/models/minivla-vq-libero90-prismatic/checkpoints/step-150000-epoch-67-loss=0.0934.pt   --task_suite_name libero_90   --center_crop True   --hf_token HF_TOKEN   --num_trials_per_task 20
```

---

## 📁 6. Results & Output Files

- Evaluation videos will be saved to:
  ```
  ./rollouts/YYYY_MM_DD/episode-xx--task=xxx.mp4
  ```

- Logs will be saved to:
  ```
  ./experiments/Logs/EVAL-*.txt
  ```

---

## ⚠️ Notes

- All model downloads rely on `git-lfs`. Ensure it is properly installed and configured.  
- If your network is slow, consider manually downloading Hugging Face files and uploading them to your server.  
- If the VQ model is missing or structure is incorrect, you may encounter:
  ```
  AssertionError: Missing VQ VAE path...
  ```
