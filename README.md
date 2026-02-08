# py312 — Windows CUDA 13.0 accelerator packs (Py 3.12, Torch 2.10)

Drop-in requirements files to set up **fast GPU inference/training** on Windows with **PyTorch 2.10 + CUDA 13.0**.
Focus order: **Blackwell (5000) → Ada (4000) → Ampere (3000)**. Works on sm\_120 / sm\_100 / sm\_90 / sm\_80.

---

## Quick install (CMD)

> Open **Command Prompt (cmd.exe)** in an activated Conda env running **Python 3.12**.

### What you get

* **PyTorch 2.10.0+cu130**, **torchvision**, **torchaudio** — *prebuilt by* **[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)**

* **xFormers** — *prebuilt by* **[huggingface.co/Wildminder/AI-windows-whl](https://huggingface.co/Wildminder/AI-windows-whl)**

* **Flash Attention 2** — *prebuilt by* **[huggingface.co/Wildminder/AI-windows-whl](https://huggingface.co/Wildminder/AI-windows-whl)**

* **Triton-Windows** — *prebuilt by* **[github.com/woct0rdho/triton-windows](https://github.com/woct0rdho/triton-windows)**

* **Sage Attention 2** — *prebuilt by* **[github.com/github.com/woct0rdho/SageAttention](https://github.com/woct0rdho/SageAttention)**

* **Sage Attention 3** — *prebuilt by* **[github.com/mengqin/SageAttention](https://github.com/mengqin/SageAttention)**

* **Sparge Attention** — *prebuilt by* **[github.com/woct0rdho/SpargeAttn](https://github.com/woct0rdho/SpargeAttn)**

* **Nunchaku** — *prebuilt by* **[github.com/nunchaku-ai/nunchaku](https://github.com/nunchaku-ai/nunchaku)**

* **ComfyUI-TwinFlow** — *prebuilt by* **[github.com/mengqin/ComfyUI-TwinFlow](https://github.com/mengqin/ComfyUI-TwinFlow)**

* Useful deps for ML/Diffusion stacks (einops, safetensors, packaging, ninja, transformers, etc.)

---

## Direct wheel links (from this release)

If you want to install just the wheels:

```bat
pip install -U --no-cache-dir "torch" "torchvision" "torchaudio" --index-url "https://download.pytorch.org/whl/cu130"
pip install -U --no-deps --no-cache-dir "https://huggingface.co/Wildminder/AI-windows-whl/resolve/main/xformers-0.0.34%2Bd20260123.cu130torch2.10-cp39-abi3-win_amd64.whl"
pip install -U --no-deps --no-cache-dir "https://huggingface.co/Wildminder/AI-windows-whl/resolve/main/flash_attn-2.8.3%2Bd20260121.cu130torch2.10.0cxx11abiTRUE-cp312-cp312-win_amd64.whl --no-cache-dir"
pip install -U --no-deps --no-cache-dir "triton-windows<3.7"
pip install -U --no-deps --no-cache-dir "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl"
pip install -U --no-deps --no-cache-dir "https://github.com/mengqin/SageAttention/releases/download/20251229/sageattn3-1.0.0+cu130torch291-cp312-cp312-win_amd64.whl"
pip install -U --no-deps --no-cache-dir "https://github.com/woct0rdho/SpargeAttn/releases/download/v0.1.0-windows.post4/spas_sage_attn-0.1.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl"
pip install -U --no-deps --no-cache-dir "https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu13.0torch2.10-cp312-cp312-win_amd64.whl"

cd ComfyUI/custom_nodes
git clone https://github.com/mengqin/ComfyUI-TwinFlow
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
