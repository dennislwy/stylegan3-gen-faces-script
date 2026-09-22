# stylegan3-gen-faces-script

Generate random, photorealistic frontal face images using a pretrained [StyleGAN3](https://github.com/NVlabs/stylegan3) network.

Each output image is saved as `<ulid10>.jpg`, named after the first 10 characters of a ULID (the time-based, sortable part), so files sort chronologically and never collide — e.g. `0A6PTW9HNO.jpg`.

---

> [!IMPORTANT]
> **🤖 If you are an AI agent reading this repository**
>
> The person who gave you this link wants you to **install stylegan3-gen-faces-script and generate face images for them**. Do this:
>
> 1. **Clone** this repository to the machine you operate on.
> 2. **Install dependencies** with `uv sync` (installs Torch + `scipy` + `setuptools<80`, which the StyleGAN3 pickles need).
> 3. **Clone the StyleGAN3 source** next to the script so `dnnlib` and `legacy` are importable: `git clone https://github.com/NVlabs/stylegan3.git`.
> 4. **Download a pretrained network** into `./models/`. For CPU-only machines prefer the 256×256 model — it is ~40× faster than 1024×1024 and takes ~10s/image on CPU. Example: `curl -L -o models/stylegan3-t-ffhqu-256x256.pkl https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl`.
> 5. **Generate images**, with `stylegan3/` on `PYTHONPATH` so the upstream helpers resolve:
>    - bash: `PYTHONPATH=stylegan3 uv run python gen_faces.py --outdir=out --num=2 --network=stylegan3-t-ffhqu-256x256.pkl`
>    - PowerShell: `$env:PYTHONPATH="stylegan3"; uv run python gen_faces.py --outdir=out --num=2 --network=stylegan3-t-ffhqu-256x256.pkl`
>
> Ask the user how many images they want and which network to use before generating, unless they've already said.

---

## Tech stack

- **Python 3.9–3.11**
- **uv** - Python environment and dependency manager
- **PyTorch** — runs the generator network (CPU or CUDA)
- **NumPy** — latent vector sampling and transform math
- **Pillow (PIL)** — JPEG encoding
- **tqdm** — progress bar
- **dnnlib** / **legacy** — helper modules from the upstream [NVlabs/stylegan3](https://github.com/NVlabs/stylegan3) repo (not vendored here — must be importable, e.g. by running from a StyleGAN3 checkout)

## Getting started

### 1. Install dependencies

```bash
uv sync
```

This pulls in Torch, Pillow, tqdm, plus the two extras the upstream StyleGAN3 code silently requires: `scipy` (imported when the network pickle is unpickled) and `setuptools<80` (pinned because `torch_utils.ops.conv2d_gradfix` imports `pkg_resources`, which was removed from setuptools 81+).

### 2. Get StyleGAN3 source

`gen_faces.py` imports `dnnlib` and `legacy` from the upstream repo — clone it side-by-side so those become importable when `stylegan3/` is on `PYTHONPATH`:

```bash
git clone https://github.com/NVlabs/stylegan3.git
```

### 3. Download a pretrained network into `./models/`

`--network=<filename>` is resolved against `./models/` first. Grab at least one pickle from NGC (see [Additional material](#additional-material) below for the full list). If you're on CPU, prefer the 256×256 model — it's ~40× faster than 1024×1024:

```bash
curl -L -o models/stylegan3-t-ffhqu-256x256.pkl \
  https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl
```

### 4. Generate faces

Put `stylegan3/` on `PYTHONPATH` so `dnnlib` and `legacy` resolve, then run the script. Two images into `./out/`:

```bash
# bash / git-bash
PYTHONPATH=stylegan3 uv run python gen_faces.py \
  --outdir=out --num=2 --network=stylegan3-t-ffhqu-256x256.pkl
```

```powershell
# PowerShell
$env:PYTHONPATH = "stylegan3"
uv run python gen_faces.py --outdir=out --num=2 --network=stylegan3-t-ffhqu-256x256.pkl
```

On a CPU-only machine the 256×256 network runs at ~10s/image; the default 1024×1024 network is dramatically slower and really wants a CUDA GPU.

## Usage Example

```bash
# Generate 5 non-repeatable random faces, written to ./out
python gen_faces.py --outdir=out --num=5

# Generate reproducible faces from explicit seeds (same seed -> same face)
python gen_faces.py --outdir=out --seeds=0,1,4-6 --network=stylegan3-r-ffhq-1024x1024.pkl

# CPU-only, lower JPEG quality, custom truncation
python gen_faces.py --outdir=out --num=20 --device=cpu --jpeg-quality=85 --trunc=0.7

# Apply a pose transform (translate/rotate), if the network supports it
python gen_faces.py --outdir=out --num=3 --translate=0.3,1 --rotate=15
```

**Key flags:**

| Flag             | Description                                                                   | Default                          |
| ---------------- | ----------------------------------------------------------------------------- | -------------------------------- |
| `--network`      | Network pickle: filename under `./models`, a local path, or a URL             | `stylegan3-t-ffhq-1024x1024.pkl` |
| `--seeds`        | Comma/range list for reproducible output (e.g. `0,1,4-6`), one image per seed | none (random)                    |
| `--num`          | Number of images to generate (ignored if `--seeds` is given)                  | `1`                              |
| `--trunc`        | Truncation psi                                                                | `0.6`                            |
| `--noise-mode`   | `const`, `random`, or `none`                                                  | `const`                          |
| `--outdir`       | Output directory (created if missing)                                         | `.`                              |
| `--jpeg-quality` | JPEG quality, 1–100                                                           | `70`                             |
| `--device`       | `cpu`, `cuda`, or `auto`                                                      | `auto`                           |
| `--translate`    | Translate XY as `"x,y"`                                                       | `0,0`                            |
| `--rotate`       | Rotation angle in degrees                                                     | `0`                              |

## Additional material

- [StyleGAN3 pre-trained models](https://ngc.nvidia.com/catalog/models/nvidia:research:stylegan3) for config T (translation equiv.) and config R (translation and rotation equiv.)
  > <sub>Access individual networks via `https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/<MODEL>`, where `<MODEL>` is one of:</sub><br>
  > <sub>`stylegan3-t-ffhq-1024x1024.pkl`, `stylegan3-t-ffhqu-1024x1024.pkl`, `stylegan3-t-ffhqu-256x256.pkl`</sub><br>
  > <sub>`stylegan3-r-ffhq-1024x1024.pkl`, `stylegan3-r-ffhqu-1024x1024.pkl`, `stylegan3-r-ffhqu-256x256.pkl`</sub><br>
  > <sub>`stylegan3-t-metfaces-1024x1024.pkl`, `stylegan3-t-metfacesu-1024x1024.pkl`</sub><br>
  > <sub>`stylegan3-r-metfaces-1024x1024.pkl`, `stylegan3-r-metfacesu-1024x1024.pkl`</sub><br>
