# stylegan3-gen-faces-script

Generate random, photorealistic frontal face images using a pretrained [StyleGAN3](https://github.com/NVlabs/stylegan3) network.

Each output image is saved as `<ulid10>.jpg`, named after the first 10 characters of a ULID (the time-based, sortable part), so files sort chronologically and never collide — e.g. `0A6PTW9HNO.jpg`.

## Tech stack

- **Python 3**
- **PyTorch** — runs the generator network (CPU or CUDA)
- **NumPy** — latent vector sampling and transform math
- **Pillow (PIL)** — JPEG encoding
- **tqdm** — progress bar
- **dnnlib** / **legacy** — helper modules from the upstream [NVlabs/stylegan3](https://github.com/NVlabs/stylegan3) repo (not vendored here — must be importable, e.g. by running from a StyleGAN3 checkout or adding it to `PYTHONPATH`)

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
