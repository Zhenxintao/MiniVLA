

# MiniVLA

<p align="center">
  <a href="https://huggingface.co/xintaozhen/MiniVLA">Hugging Face</a> |
  <a href="#reproduction-guide">Guía de reproduccióncción</a> |
  <a href="#results">Resultados</a>
</p>


MiniVLA es un framework VLA (Visión-Lenguaje-Acción) ligero, modular y fácil de implementar.  
Está construido sobre [OpenVLA-Mini](https://github.com/Stanford-ILIAD/openvla-mini), y optimizado adicionalmente para **despliegue en edge** a través de una canalización de aceleración híbrida.  

Características clave:
- ⚡ **Aceleración con TensorRT**: Vision Encoder y LLM (Qwen-0.5B) exportados a motores ONNX / TensorRT, reduciendo la latencia y el uso de memoria GPU.  
- 🖥️ **Diseño orientado a edge**: validado en dispositivos con 8 GB de VRAM (RTX 4060 Laptop GPU) como proxy de Jetson Orin Nano, logrando un uso eficiente de la memoria e inferencia en tiempo real.  
- 🌐 **Servicio en línea basado en FastAPI**: transforma scripts de evaluación fuera de línea en una API de inferencia reutilizable (`/act`), permitiendo el control interactivo.  
- 🔄 **Canalización híbrida PyTorch + TensorRT**: garantiza compatibilidad y mecanismo de respaldo (fallback) mientras maximiza los beneficios de la aceleración.  

---

## 📦 Guía de Instalación y Reproducción

Esta sección proporciona instrucciones paso a paso para instalar **MiniVLA** y configurar la canalización de aceleración híbrida (TensorRT + PyTorch) para el **despliegue en edge** (por ejemplo, Jetson Orin Nano o GPUs de clase 8 GB).  

La canalización incluye:
- Instalación del proyecto MiniVLA y sus dependencias  
- Habilitación de la aceleración TensorRT para el Vision Encoder y el LLM  
- Preparación de LIBERO para la evaluación de tareas  

### Inicio Rápido

```bash
# Step 1: Create conda environment
conda create -n minivla python=3.10 -y
conda activate minivla

# Step 2: Install PyTorch + CUDA (note: safetensors==0.4.3 is required)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Step 3: Clone repository
git clone https://github.com/Zhenxintao/MiniVLA.git
cd MiniVLA

# Install dependencies
pip install -e .

# Step 4: Install flash-attn (for efficient inference, required)
# ⚠️ If you encounter CUDA_HOME errors, please check environment variable setup in Section 4.
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation

# Step 5: Install LIBERO simulation platform
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .

# Step 6: Install LIBERO evaluation dependencies
# ⚠️ Check that 'robosuite' version is exactly 1.4.0
cd ../MiniVLA
pip install -r experiments/robot/libero/libero_requirements.txt

# Step 7: (Optional) Enable Hybrid Acceleration with TensorRT
# Download the pre-exported TensorRT vision encoder engine from Hugging Face:
# https://huggingface.co/xintaozhen/MiniVLA
# After that Run the TensorRT Vision Encoder microservice
cd tensorRT-scripts
python tensorRT_vision_service.py

# Run LIBERO evaluation script
python experiments/robot/libero/run_libero_eval.py \
  --model_family prismatic \
  --pretrained_checkpoint /{Your model checkpoint path} \
  --task_suite_name libero_90 \
  --center_crop True \
  --hf_token HF_TOKEN \
  --num_trials_per_task 20


```
<a id="reproduction-guide"></a>
👉 Si te encuentras con problemas complejos de entorno, o deseas **reproducir OpenVLA-Mini desde cero** (incluyendouyendondo la configuración de CUDA, flash-attn y variables de entorno), consulta nuestras guías de reproducción detalladas:  
- 📄 [Versión en inglés](./Results/openvla_mini_reproduction_guide_en.md)  
- 📄 [Versión en chino](./Results/openvla_mini_reproduction_guide.md)  

---

## 📂 Pesos y Checkpoints del Modelo

Todos los pesos preentrenados, motores TensorRT y modelos Qwen compatibles con Hugging Face se alojan en:  

👉 [Hugging Face: xintaozhen/MiniVLA](https://huggingface.co/xintaozhen/MiniVLA)

Esto incluye:

- `models/`: checkpoints de [Stanford-ILIAD/minivla-vq-libero90-prismatic](https://huggingface.co/Stanford-ILIAD/minivla-vq-libero90-prismatic)  
- `qwen25-0_5b-trtllm/`: Qwen-0.5B en formato TensorRT-LLM  
- `tensorRT/`: ONNX del Vision encoder y motor TensorRT  

---

## 🏗️ Arquitectura del Sistema

<p align="center">
  <img src="./Results/System_Architecture.svg" width="90%">
</p>


### Aceleración Híbrida

<p align="center">
  <img src="./Results/MiniVLA_Architecture.svg" width="90%">
</p>


---

## 📑 Descripción de los Archivos

Este repositorio extiende **OpenVLA-Mini** con aceleración basada en TensorRT y un despliegue modular. Los archivos clave son:

### 1. Microservicios de TensorRT (`tensorRT-scripts/`)

- `tensorRT_llm_service.py` → Ejecuta el **servicio TensorRT-LLM** para Qwen-0.5B.  
- `tensorRT_vision_service.py` → Ejecuta el **servicio Vision Encoder de TensorRT**.  
  ⚡ Estos corresponden a los **dos microservicios independientes** en el diagrama de arquitectura.

### 2. Scripts de Despliegue VLA (`vla-scripts/`)

- `deploy_minivla.py` → Inicia MiniVLA con un **servicio de inferencia FastAPI** (`/act`), permitiendo inferencia basada en imágenes y prompts de lenguaje.  
- `export_vision_encoder_onnx.py` → Exporta el vision encoder al **formato ONNX** para la conversión a TensorRT.

### 3. Marco de Trabajo Experimental (`experiments/robot/`)

- `trt_backbone.py` → **Backbone visual** acelerado por TensorRT.  
- `trt_llm_backbone.py` → **Backbone LLM** acelerado por TensorRT-LLM.  
- `openvla_utils.py` & `robot_utils.py` → Implementan el **mecanismo de enrutamiento y respaldo (Router & Fallback)**, gestionando la inferencia local frente a los servicios acelerados.

---

## 🔑 Contribuciones Clave

- Construcción de un **marco de inferencia en línea de extremo a extremo** con un servicio FastAPI (`/act`), transformando código de benchmarks fuera de línea en un **sistema desplegable en tiempo real**.  
- Reproducción de un **OpenVLA-Mini** ligero y propuesta de una **canalización de aceleración híbrida**.  
- Exportación del **vision encoder** a TensorRT, reduciendo la latencia de percepción y el uso de memoria GPU.  
- Mejora de la **eficiencia de la memoria GPU**: se redujo el uso promedio de ~67% a ~43%, y el picouso /pico de ~85% a ~65%, haciendo el despliegue factible bajo restricciones de 8 GB de memoria (similar a dispositivos tipo Jetson).  
- Integración de **Qwen 2.5 0.5B** en formatos Hugging Face y TensorRT-LLM.  
- Diseño de una **arquitectura de sistema modular** con enrutamiento y respaldo para mayor robustez.  
- Demostración de una **inferencia VLA en edge /eficiente** en Jetson Orin Nano en tareas LIBERO, con una caída moderada de rendimiento (5–10%).  

---

## 🖥️ Dispositivo y Rendimiento

Despliegue objetivo: **Jetson Orin Nano (variantes de 16 GB / 8 GB)**.  

Para simulación y reproducibilidad, los experimentos se realizaron en una **estación de trabajo local** equipada con:

- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)  
- **Controlador / CUDA**: Controlador 550.144.03, CUDA 12.4  
- **SO**: Ubuntu 22.04 LTS  

⚠️ **Nota**: Aunque los experimentos se ejecutaron en una RTX 4060 Laptop GPU (8 GB VRAM), el dispositivo /sirve principalmente como un sustituto para evaluar **restricciones de memoria** comparables a las de Jetson Orin Nano. La velocidad de inferencia absoluta en dispositivos Jetson puede ser más lenta debido a una /menor potencia computacional, pero las tendencias de uso de memoria permanecen /consistentes.  

---

<a id="results"></a>
## 📊 Resultados Experimentales

Evaluamos exhaustivamente **MiniVLA** bajo diferentes estrategias de aceleración (Baseline de PyTorch vs. Híbrido TensorRT).  
Los experimentos cubren el **uso de memoria GPU**, el **desglose de latencia** y un **estudio de ablation**.

### 1. Uso de Memoria GPU

| Variante del modelo               | Uso de memoria | Uso promedio de GPU | Uso /pico de GPU |
|-----------------------------|--------------|----------------|----------------|
| Baseline MiniVLA (PyTorch)  | 4115/8188MiB | ~67%           | ~80%           |
| MiniVLA (TRT Vision)        | 3892/8188MiB | ~41%           | ~57%           |
| MiniVLA (TRT Vision + LLM)  | 3292/8188MiB | ~35%           | ~50%           |

➡️ **Observación**: El TensorRT solo de la visión reduce la memoria en ~223 MiB y disminuye el uso /pico de GPU en ~23%.  
La aceleración completa Vision+LLM logra el uso de memoria más bajo (3292 MiB), pero produce salidas inválidas (0% de éxito).  

---

### 2. Desglose de Latencia (ms)

| Módulo            | Baseline | Híbrido (Visión) | Híbrido (Visión+LLM) |
|-------------------|----------|-----------------|----------------------|
| Preprocesamiento de imagen  | 15       | 15              | 15                   |
| Vision Encoder    | 138      | 47              | 47                   |
| Inferencia LLM     | 202      | 202             | 120                  |
| Decodificación de acción   | 10       | 10              | 10                   |
| **Fin a fin**    | **365**  | **274**         | **192**              |

➡️ **Observación**: La aceleración del vision encoder reduce la latencia en un –65,9% (138 → 47 ms).  
La latencia general mejora en un –24,9% (365 → 274 ms).  
La aceleración adicional del LLM reduce la latencia a 192 ms, pero produce resultados inválidos.  

---

### 3. Estudio de Ablation

| Configuración | Latencia (ms) | Uso de GPU /pico (%) | Memoria (MiB) | Éxito (%) |
|---------------|--------------|---------------------|--------------|-------------|
| A: Baseline (PyTorch) | 365 | 80 | 4115 | 80.0 |
| B: Solo Visión (TRT)  | 274 | 57 | 3892 | 75.5 |
| C: Visión+LLM (TRT)   | 192 | 50 | 3292 | 0.0 |

➡️ **Observación**: La mejor relación eficiencia-efectividad se logra con **TensorRT solo de visión**.  
La aceleración completa Vision+LLM obtiene la menor latencia, pero produce salidas inválidas (0% de éxito).  

---

## 🎬 Resultados de las Tareas

Evaluamos MiniVLA en **tareas de escritorio de LIBERO**. A continuación se muestran demostraciones:

<table>
<tr>
<td align="center">
  <b>Cerrar el cajón superior del mueble</b><br>
  <img src="./Results/success_close_the_top_drawer_of_the_cabinet.gif" width="220"><br>
  ✅ Éxito (Original): 20/20 <br>
  ⚡ Éxito (TensorRT Híbrido): 19/20
</td>
<td align="center">
  <b>Colocar el cu negro en el cajón superior del mueble</b><br>
  <img src="./Results/success_put_the_black_bowl_in_the_top_drawer_of_the_cabine.gif" width="220"><br>
  ✅ Éxito (Original): 14/20 <br>
  ⚡ Éxito (TensorRT Híbrido): 12/20
</td>
<td align="center">
  <b>Apagar la estufa</b><br>
  <img src="./Results/success_turn_off_the_stove.gif" width="220"><br>
  ✅ Éxito (Original): 14/20 <br>
  ⚡ Éxito (TensorRT Híbrido): 14/20
</td>
<td align="center">
  <b>Cerrar el cajón inferior del mueble</b><br>
  <img src="./Results/success_close_the_bottom_drawer_of_the_cabinet.gif" width="220"><br>
  ✅ Éxito (Original): 16/20 <br>
  ⚡ Éxito (TensorRT Híbrido): 16/20
</td>
<td align="center">
  <b>Cerrar el cajón superior del mueble y colocar el cu negro encima de él</b><br>
  <img src="./Results/success_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it.gif" width="220"><br>
  ✅ Éxito (Original): 13/20 <br>
  ⚡ Éxito (TensorRT Híbrido): 11/20
</td>
</tr>

<tr>
<td align="center">
  <b>Colocar la taza amarilla y blanca al frente de la taza blanca</b><br>
  <img src="./Results/success_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug.gif" width="220"><br>
  ✅ Éxito (Original): 16/20 <br>
  ⚡ Éxito (TensorRT Híbrido): 14/20
</td>
<td align="center">
  <b>Recoger el cu negro de la izquierda y ponerlo en la bandeja</b><br>
  <img src="./Results/success_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray.gif" width="220"><br>
  ✅ Éxito (Original): 20/20 <br>
  ⚡ Éxito (TensorRT Híbrido): 20/20
</td>
<td align="center">
  <b>Recoger el libro y colocarlo en el compartimento izquierdo del organizador</b><br>
  <img src="./Results/success_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy.gif" width="220"><br>
  ✅ Éxito (Original): 14/20 <br>
  ⚡ Éxito (TensorRT Híbrido): 13/20
</td>
<td align="center">
  <b>Recoger el libro del centro y colocarlo en el estante del mueble</b><br>
  <img src="./Results/success_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf.gif" width="220"><br>
  ✅ Éxito (Original): 15/20 <br>
  ⚡ Éxito (TensorRT Híbrido): 15/20
</td>
<td align="center">
  <b>Recoger el libro de la izquierda y colocarlo encima del estante</b><br>
  <img src="./Results/success_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf.gif" width="220"><br>
  ✅ Éxito (Original): 18/20 <br>
  ⚡ Éxito (TensorRT Híbrido): 17/20
</td>
</tr>
</table>


---

## 🔗 Enlaces Relacionados

- 📄 Pesos de Hugging Face: [xintaozhen/MiniVLA](https://huggingface.co/xintaozhen/MiniVLA)  
- 🧑‍💻 Repositorio base: [Stanford-ILIAD/openvla-mini](https://github.com/Stanford-ILIAD/openvla-mini)
