# Kandinsky 5 Pro – NVIDIA Blackwell-Oriented Enhancements

This is an **experimental fork** of the official [Kandinsky 5](https://github.com/kandinskylab/kandinsky-5) repository, focused on:

- **NVIDIA Blackwell GPUs** (focused on RTX PRO 6000 Blackwell at the moment)
- **High-quality video generation** with Kandinsky 5 Pro T2V
- **Performance optimizations** using NVIDIA’s Transformer Engine (TE) and FP8 kernels (waiting for full NVFP4 support)

The goal of this fork is to explore “turbo” compute backends for Kandinsky Pro video models on modern NVIDIA hardware (starting with FP8 on Blackwell, with NVFP4 kept as an experimental path).

> ⚠️ **Status:** Work in progress.  
> This fork is currently aimed at **power users with NVIDIA Blackwell GPUs** and the NVIDIA NGC PyTorch Docker image.

---

## Supported Environment (NGC Docker Only)

This fork **officially supports** running inside the NVIDIA NGC PyTorch container:

- **Base image (recommended):**  
  `nvcr.io/nvidia/pytorch:25.10-py3`
- **Hardware target:**  
  NVIDIA Blackwell GPUs (RTX PRO 6000 Blackwell, etc.)
- **OS layout:**  
  - Windows 11 + WSL2 Ubuntu + Docker Desktop  
  - or a native Linux + Docker + NVIDIA Container Toolkit setup

Other environments (bare-metal venvs, non-NGC containers, older GPUs) are considered **community / experimental** and may work, but are not the focus of this fork.

---

## Quickstart (Blackwell + NGC)

1. **Clone this fork**

   ```bash
   git clone https://github.com/ferrex-aurakvena/k5pro-blackwell.git k5pro-blackwell
   cd k5pro-blackwell
   ```

2. **Start the NGC PyTorch container**

   The repo includes a convenience script that:

   * Pulls `nvcr.io/nvidia/pytorch:25.10-py3` (once — about 10 GB download)
   * Mounts the repo at `/workspace/k5`
   * Sets safe IPC / ulimit settings for PyTorch

   ```bash
   ./run_ngc.sh
   ```

   You should now be in a shell inside the container with your repo mounted.

   > Tip: you can override the image tag if needed:
   >
   > ```bash
   > NGC_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3 ./run_ngc.sh
   > ```

3. **Install Python dependencies inside the container**

   Inside the container:

   ```bash
   pip install --upgrade pip
   pip install -r requirements-core.txt
   ```

   The container already ships:

   * `torch`, `torchvision`, `torchaudio`
   * CUDA, cuDNN
   * `transformer_engine` (TE)

   `requirements-core.txt` only adds the **high-level Python deps** needed by the fork (diffusers, transformers, etc.).

4. **Check TE / FP8 / NVFP4 capability**

   Still inside the container:

   ```bash
   python test_ngc.py
   ```

   You’ll see a summary like:

   ```text
   === NGC / PyTorch / Transformer Engine capability check ===

   PyTorch version: 2.9.0a0+...
   CUDA device:     NVIDIA RTX PRO 6000 Blackwell Max-Q ...
   SM capability:   12.0
   torch.version.cuda: 13.0

   Transformer Engine: present (version: 2.8.0+...)

   === Summary ===
   Transformer Engine version: 2.8.0+...

   TE FP8 status:   SUPPORTED
     Details:       FP8 recipe and kernel path ran successfully

   NVFP4 status:    EXPERIMENTAL
     Details:       NVFP4 recipe present, but this GPU is RTX Blackwell (SM12x).
                    Current NGC TE build is known to emit FP4 PTX warnings here;
                    skipping NVFP4 kernel test and treating NVFP4 as experimental.
   ```

   Interpretation:

   * **TE FP8: SUPPORTED** → you can safely enable a **TE-FP8 backend** in the fork
   * **NVFP4: EXPERIMENTAL** → symbols exist, but **RTX Blackwell NVFP4** is not yet fully supported in the NGC TE build and should be treated as research-only

5. **Run the actual pipeline**

   Once capability checks are green, you can run whatever CLI entrypoints this fork exposes (T2V scripts, config-driven generation, etc.). The exact commands will evolve as the fork matures.

   For now, this README only defines the **environment & capability story**; the model-side enhancements (TE-FP8 DiT blocks, scheduler tweaks, VAE improvements, etc.) are being layered in on top of this foundation.

---

## Design Philosophy

This fork is opinionated:

* **Blackwell-first**
  We optimize for users on NVIDIA’s latest GPUs. Older architectures may still run, but they’re not the priority.

* **NGC-first**
  We rely on NVIDIA’s own PyTorch/TE container to avoid “dependency hell” and fragile local CUDA installs.

* **Transformer Engine for FP8**
  FP8 support is handled via TE (not bare PyTorch float8) for better kernels, scaling recipes, and hardware alignment on Hopper/Ada/Blackwell.

* **NVFP4 as experimental**
  NVFP4 is present in TE 2.8+, but full support on RTX Blackwell (SM12x) is still catching up. This fork keeps the NVFP4 plumbing visible, but will not treat it as a stable backend until NVIDIA’s TE/NGC stack explicitly supports it for SM12x/SM121.

---

## Upstream Project (Original Kandinsky 5)

This fork is built on top of the excellent work from the original Kandinsky team.
For model details, research writeup, and broader ecosystem integrations, please refer to the upstream project:

<h1>Please see the <a href="https://github.com/kandinskylab/kandinsky-5">original Kandinsky 5 repository</a>:</h1>
<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/KANDINSKY_LOGO_1_WHITE.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/KANDINSKY_LOGO_1_BLACK.png">
    <img alt="Kandinsky 5 logo" src="https://user-images.githubusercontent.com/25423296/163456779-a8556205-d0a5-45e2-ac17-42d089e3c3f8.png">
  </picture>
</div>

<div align="center">
  <a href="https://habr.com/ru/companies/sberbank/articles/951800/">Habr</a> |
  <a href="https://kandinskylab.ai/">Project Page</a> |
  <a href="https://arxiv.org/abs/2511.14993">Technical Report</a> |
  🤗 <a href="https://huggingface.co/collections/kandinskylab/kandinsky-50-video-lite">Video Lite</a> /
  <a href="https://huggingface.co/collections/kandinskylab/kandinsky-50-video-pro">Video Pro</a> /
  <a href="https://huggingface.co/collections/kandinskylab/kandinsky-50-image-lite">Image Lite</a> |
  <a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/kandinsky5">🤗 Diffusers</a> |
  <a href="https://github.com/kandinskylab/kandinsky-5/blob/main/comfyui/README.md">ComfyUI</a>
</div>

<h1>Kandinsky 5.0: A family of diffusion models for Video &amp; Image generation</h1>

