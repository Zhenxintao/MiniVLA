## 🧠 OpenVLA-Mini + Prismatic + LIBERO90 复现流程手册

> 项目来源: [Stanford-ILIAD/openvla-mini](https://github.com/Stanford-ILIAD/openvla-mini)  
> 模型仓库: [minivla-vq-libero90-prismatic](https://huggingface.co/Stanford-ILIAD/minivla-vq-libero90-prismatic)  
> VQ-VAE仓库: [pretrain_vq](https://huggingface.co/Stanford-ILIAD/pretrain_vq)  
> 本手册整理与适配: [Zhenxintao/MiniVLA](https://github.com/Zhenxintao/MiniVLA)  
> 相关模型与权重: [xintaozhen/MiniVLA](https://huggingface.co/xintaozhen/MiniVLA)

---

## 📦 1. 准备环境

### ✅ 创建 Conda 虚拟环境
```bash
conda create -n minivla python=3.10 -y
conda activate minivla
```

### ✅ 安装依赖
```bash
# Step 1: 安装 PyTorch 与 CUDA（safetensors对应需要为0.4.3 safetensors==0.4.3）
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Step 2: 克隆主仓库
git clone https://github.com/Stanford-ILIAD/openvla-mini.git
cd openvla-mini

# Step 3: 安装项目依赖
pip install -e .

# Step 4: 安装 flash-attn（用于高效推理，必要），如果安装时存在CUDA_HOME有关错误，请看第四部分的环境变量设置
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation

# Step 5: 安装 LIBERO 仿真平台
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .

# Step 6: 安装 LIBERO 评估脚本依赖 注意: 安装依赖后检查一下'robosuite'的版本，'robosuite' 需要为 robosuite==1.4.0
cd ../openvla-mini
pip install -r experiments/robot/libero/libero_requirements.txt

```

---

## 📥 2. 下载模型文件

### ✅ 下载 Prismatic 检查点

```bash
# Step 1: 首先创建一个models文件夹用于存放model
cd /root/models

# Step 2: 使用 git lfs + huggingface-cli 下载整个仓库
sudo apt update
sudo apt install git git-lfs
pip install huggingface_hub

# 以上git-lfs 以及 huggingface_hub 安装完成后进行login
huggingface-cli login

git lfs install # 只需要初始化一次

# '/Stanford-ILIAD/minivla-vq-libero90-prismatic' 这只是我用的一个其中一个预训练的model，可以在huggingface 查看其他model
git clone https://huggingface.co/Stanford-ILIAD/minivla-vq-libero90-prismatic

cd minivla-vq-libero90-prismatic
git lfs pull # 开始下载权重文件 注意: 如果遇到TLS握手过程失败，可以尝试切换到openssl版本的Git

```

### ✅ 下载 VQ-VAE 模型文件

```bash
mkdir -p /root/openvla-mini/vq
huggingface-cli download Stanford-ILIAD/pretrain_vq \
  --local-dir /root/openvla-mini/vq \
  --local-dir-use-symlinks False
# 如果huggingface-cli 命令下载出现异常，可以手动下载并上传到服务器，huggingface地址为: https://huggingface.co/Stanford-ILIAD/pretrain_vq
```

最终应该有：

```bash
/root/models/minivla-vq-libero90-prismatic/checkpoints/*.pt
/root/openvla-mini/vq/pretrain_vq+mx-libero_90+fach-7+ng-7+nemb-128+nlatent-512/{checkpoints/model.pt, config.json}
```

---

## 🧩 3. 安装 VQ-VAE 推理代码（用于 Action Tokenizer）

```bash
cd /root/openvla-mini
git clone https://github.com/jayLEE0301/vq_bet_official vqvae
cd vqvae
pip install -e .
```

确保结构如下（注意 `__init__.py`）：

```
openvla-mini/
└── vqvae/
    ├── vqvae/
    │   ├── __init__.py
    │   ├── vqvae.py
    │   └── vqvae_utils.py
```

---

## 🔐 4. 环境变量设置（如果需要）

```bash
huggingface-cli login
# 或：
export HF_TOKEN=your_token_here

# 环境变量设置
export PRISMATIC_DATA_ROOT=/root/.cache/prismatic_data
export LIBERO_DATA_ROOT=/root/.cache/libero_data
export HF_HUB_CACHE=/root/.cache/huggingface
export TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets

# CUDA 环境变量设置
echo 'export CUDA_HOME=/usr/local/cuda-12.4' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# CUDA12.4 下载及安装
浏览器下载: https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
终端下载: wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-87AE4CD8-keyring.gpg /usr/share/keyrings/

sudo apt update
sudo apt install -y cuda

```

---

## ▶️ 5. 运行 LIBERO 评估脚本

```bash
cd /root/openvla-mini

python experiments/robot/libero/run_libero_eval.py \
  --model_family prismatic \
  --pretrained_checkpoint /home/cody/models/minivla-vq-libero90-prismatic/checkpoints/step-150000-epoch-67-loss=0.0934.pt \
  --task_suite_name libero_90 \
  --center_crop True \
  --hf_token HF_TOKEN \
  --num_trials_per_task 20
```

---

## 📁 6. 结果与输出文件

- 每个任务会保存评估视频于：
  ```
  ./rollouts/YYYY_MM_DD/episode-xx--task=xxx.mp4
  ```

- 日志默认保存在：
  ```
  ./experiments/Logs/EVAL-*.txt
  ```

---

## ⚠️ 注意事项

- 所有模型下载过程依赖 `git-lfs`，必须正确安装并配置。
- 如果网络缓慢建议使用本地下载 HuggingFace 文件再上传。
- 若缺失 VQ 模型或模型结构异常，将出现：
  ```
  AssertionError: Missing VQ VAE path...
  ```

