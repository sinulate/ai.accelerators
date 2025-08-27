# accel-py313 — Windows CUDA 12.9 accelerator packs (Py 3.13, Torch 2.8)

Drop-in requirements files to set up **fast GPU inference/training** on Windows with **PyTorch 2.8.0 + CUDA 12.9**.
Focus order: **Blackwell (5000) → Ada (4000) → Ampere (3000)**. Works on sm\_120 / sm\_100 / sm\_90 / sm\_80.

---

## Quick install (CMD)

> Open **Command Prompt (cmd.exe)** in an activated Conda env running **Python 3.13**.

**Core accelerators**

```bat
pip install -r https://raw.githubusercontent.com/sinulate/accel-py313/main/accel-py313.txt
```

**ComfyUI bundle** (includes Core + Comfy specifics)

```bat
pip install -r https://raw.githubusercontent.com/sinulate/accel-py313/main/accel-py313-comfyui.txt
```

### What you get

* **PyTorch 2.8.0+cu129**, **torchvision**, **torchaudio**

* **Triton-Windows** *prebuilt by* **@woct0rdho**

* **Flash Attention 2** — *my* prebuilt Blackwell-optimized wheel

* **Sage Attention 2** — *prebuilt by* **@woct0rdho**

* **Sage Attention 3** — *my* prebuilt Blackwell-optimized wheel

* **Sparge Attention** — *prebuilt by* **@woct0rdho**

* Useful deps for ML/Diffusion stacks (einops, safetensors, packaging, ninja, transformers, etc.)

---

## Direct wheel links (from this release)

If you want to install just the wheels:

```bat
pip install --no-deps torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
pip install --no-deps triton-windows<3.5
pip install --no-deps https://github.com/sinulate/accel-py313/releases/download/v0.1.0/flash_attn-2.8.3-cp313-cp313-win_amd64.whl
pip install --no-deps https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post2/sageattention-2.2.0+cu128torch2.8.0.post2-cp39-abi3-win_amd64.whl
pip install --no-deps https://github.com/sinulate/accel-py313/releases/download/v0.1.0/sageattn3-1.0.0-cp313-cp313-win_amd64.whl
pip install --no-deps https://github.com/woct0rdho/SpargeAttn/releases/download/v0.1.0-windows.post1/spas_sage_attn-0.1.0+cu128torch2.8.0.post1-cp39-abi3-win_amd64.whl
```

> **Credit:** SageAttention 2 (+ Sparge Attention) wheels referenced by the txt files are built and published by **[@woct0rdho](https://github.com/woct0rdho)**. Please ⭐ their repos.

---

## Sanity checks (CMD one-liners)

**PyTorch + CUDA + device**

```bat
python -c "import torch as t; print('Torch', t.__version__, 'CUDA', t.version.cuda); print('GPU[0]:', t.cuda.get_device_name(0)); print('SM:', t.cuda.get_device_capability(0))"
```

**Triton (Windows)**

```bat
python -c "import triton, torch as t; print('Triton', triton.__version__); print('CUDA available:', t.cuda.is_available())"
```

**Flash Attention 2**

```bat
python -c "import torch as t; from flash_attn.flash_attn_interface import flash_attn_func; q=t.randn(2,256,16,64,device='cuda',dtype=t.float16); o=flash_attn_func(q,q,q,0.0,False,False); print('FA2 OK, shape:', tuple(o.shape))"
```

**Sage Attention 2**

```bat
python -c "import torch as t; from sageattention import sageattn as s2; q=k=v=t.randn(2,256,16,64,device='cuda',dtype=t.float16); o=s2(q,k,v,tensor_layout='HND',is_causal=False); print('SA2 OK, shape:', tuple(o.shape))"
```

**Sage Attention 3 (Blackwell-only)**

```bat
python -c "import torch as t; from sageattn.api import sageattn_blackwell as s3; q=k=v=t.randn(2,256,16,64,device='cuda',dtype=t.float16); o=s3(q,k,v,tensor_layout='HND',is_causal=False); print('SA3 OK, shape:', tuple(o.shape))"
```

**Sparge Attention**

```bat
python -c "import importlib, pkgutil; cand=['spas_sage_attn','sparge_attn','spargeattention']; name=next((n for n in cand if pkgutil.find_loader(n)), None); print('Sparge OK, module:', name or 'NOT FOUND')"
```

---

## Tips & Troubleshooting

* Use **cmd.exe** (not PowerShell) for these one-liners.
* If you see `ModuleNotFoundError: flash_attn_2_cuda` or `DLL load failed`, make sure you:

  1. installed from the txt files above **and**
  2. import `torch` before other CUDA extensions in the same process.
* You need **Visual C++ build tools** + **CUDA 12.9 runtime** (installed with your NVIDIA driver for 550+).

---

## License & Credits

* This repo provides requirements files and a couple of Windows wheels I built (FlashAttention2, SageAttention3).
* **SageAttention 2** and **Sparge Attention** wheels are by **@woct0rdho** — all credit to them for their awesome Windows builds.
  Please visit and ⭐ their repositories.

---

**Enjoy the speed!** If these files helped, consider opening issues/PRs with tweaks for more GPUs or stacks.
