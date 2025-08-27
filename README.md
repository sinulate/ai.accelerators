# Win-Accel Kits (PyTorch 2.8 · CUDA 12.9 · Py3.13)

Super simple, **one-file installs** for a fast CUDA stack on Windows.
Two preset profiles:

* **`accel-py313.txt`** – base accelerators for DL/ML.
* **`accel-py313-comfyui.txt`** – everything in base **+** ComfyUI ecosystem.

Both target **PyTorch 2.8 (CUDA 12.9)** on **Python 3.13** and include local wheels for your custom kernels.

---

## What’s inside

* PyTorch 2.8 + cu129: `torch`, `torchvision`, `torchaudio`, `torchsde`
* Kernel/graph compilers: `triton-windows`, `xformers`, `accelerate`
* Your attention accelerators:

  * Local wheels (put these in `./wheels/`):

    * `flash_attn-2.8.3-cp313-cp313-win_amd64.whl`
    * `sageattn3-1.0.0-cp313-cp313-win_amd64.whl`
  * Prebuilt SageAttention v2 (ABI3; works Py3.9–3.13):

    * `https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post1/sageattention-2.2.0+cu129torch2.8.0-cp39-abi3-win_amd64.whl`
* Useful libs for DL/diffusion: `numpy`, `einops`, `safetensors`, `packaging`, `ninja`, `transformers`, `diffusers`, `sentencepiece`, `tokenizers`
* **ComfyUI profile** also adds `comfyui-embedded-docs`, `comfyui_frontend_package`, `comfyui_workflow_templates`, plus common video/net deps.

---

## Quick start

> ✅ Requires Windows 10/11 x64, **Python 3.13** (Conda recommended), and an NVIDIA GPU/driver compatible with CUDA 12.9.

1. **Clone this repo** and place your two wheels under `./wheels/`:

```
wheels/
  flash_attn-2.8.3-cp313-cp313-win_amd64.whl
  sageattn3-1.0.0-cp313-cp313-win_amd64.whl
```

2. **Create / activate** a Python 3.13 environment (example with Conda):

```bat
conda create -n py313 python=3.13 -y
conda activate py313
```

3. **Install one profile**:

**Base accelerators**

```bat
pip install -r accel-py313.txt
```

**ComfyUI accelerators**

```bat
pip install -r accel-py313-comfyui.txt
```

That’s it. The files already set the PyTorch CUDA index (`--extra-index-url`) and point to your local wheels.

---

## Verify the install

Minimal sanity checks:

```py
import torch
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU:", torch.cuda.get_device_name())
```

FlashAttention 2 import:

```py
import flash_attn
print("flash_attn OK")
```

SageAttention 3 smoke test (tiny shapes):

```py
import torch as t
from sageattention import sageattn
q=k=v=t.randn(2,256,16,64, device='cuda', dtype=t.float16)
out = sageattn(q,k,v, tensor_layout='HND', is_causal=False)
print("sageattn3 OK", tuple(out.shape))
```

---

## GPU compatibility (your wheels)

Your custom builds include kernels for **SM 80, 90, 100, 120**.
This broadly covers modern NVIDIA families (e.g., Ampere/Hopper/Blackwell). If you see:

> `CUDA error: no kernel image is available for execution on the device`

…your GPU’s compute capability isn’t one of the compiled targets (e.g., some Ada 40-series are **sm\_89**). In that case, rebuild with the right `-gencode` for your card (and ideally include PTX, e.g. `code=[sm_XY,compute_XY]`) or open an issue and we can add it.

---

## Using the prebuilt SageAttention v2 (optional)

The requirements files already include a direct wheel URL that works across Python 3.9–3.13:

```
https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post1/sageattention-2.2.0+cu129torch2.8.0-cp39-abi3-win_amd64.whl
```

This is handy for environments where you don’t want to pin to `cp313`.

---

## ComfyUI notes

The **ComfyUI** profile installs common extras used by nodes and templates. After installing the profile:

```bat
# if you haven't already
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
python main.py
```

(Adjust paths if you keep ComfyUI elsewhere—this repo only provides the Python deps.)

---

## Troubleshooting

* **Wrong linker or “/usr/bin/link extra operand”**
  You’re likely running from Git Bash. Use plain **Command Prompt (cmd.exe)** or PowerShell for installs.
* **No compiler needed**
  For **using** these wheels, you don’t need Visual Studio Build Tools. (You only need them if you’re rebuilding.)
* **Out of memory during builds**
  If you ever rebuild from source, limit NVCC workers (e.g., `--threads 2`) and reduce the number of `-gencode` targets, then re-add once stable.

---

## License

This repo is just dependency glue + your build artifacts. Libraries retain their own licenses.

---

Happy accel-ing 🚀
