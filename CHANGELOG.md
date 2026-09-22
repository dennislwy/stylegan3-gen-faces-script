# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- With `--seeds`, output files are now named `seed-<seed>.jpg` (e.g. `seed-1.jpg`) instead of getting a ULID name, so a rerun with the same seed overwrites the same file. Unseeded runs (`--num`) keep the `<ulid10>.jpg` naming.

## [1.0.1](https://github.com/dennislwy/stylegan3-gen-faces-script/releases/tag/1.0.1) - 2026-09-22

### Added

- Torch is now opt-in via mutually-exclusive `cpu` / `cuda` extras routed through the matching PyTorch wheel index (`download.pytorch.org/whl/cpu` and `/cu124`). Install with `uv sync --extra cpu` or `uv sync --extra cuda` — CPU-only machines no longer pull the ~2 GB CUDA wheels.
- Added `ninja` to project dependencies to speed up StyleGAN3's C++/CUDA extension builds.

### Changed

- Pinned `torch==2.6.0` (previously `>=2.0`) for reproducible installs across the `cpu` and `cuda` extras.

## [1.0.0](https://github.com/dennislwy/stylegan3-gen-faces-script/releases/tag/1.0.0) - 2026-09-22

### Added

- `--network` now auto-downloads the requested pickle from NGC when it is not present in `./models/`. Matches names with or without the `.pkl` extension against the NGC file listing; skips the network call when the value is already a URL or a local path.
- `gen_faces.py` CLI for generating face images from a pretrained StyleGAN3 network. Supports `--num` (non-repeatable, OS-entropy-seeded) and `--seeds` (deterministic, bit-compatible with upstream `gen_images.py`) generation modes.
- Generation flags: `--outdir`, `--network`, `--trunc`, `--noise-mode`, `--jpeg-quality`, `--device` (`cpu`/`cuda`/`auto`), `--translate`, `--rotate`.
- Output filenames use a 10-character ULID prefix (Crockford Base32 of the current-ms timestamp) so files sort chronologically and never collide.
- `--list-models`: prints available StyleGAN3 pickles on NGC and marks any already present in `./models/`.
- `--list-models --json`: emits a machine-readable JSON array (name, sizeBytes, url, downloaded).
- `--download-model NAME`: fetches a pickle from NGC into `./models/` with a `tqdm` progress bar; writes to `<name>.pkl.part` and atomically renames on success. Accepts the name with or without `.pkl`.
- `--network` accepts a bare filename with or without the `.pkl` extension; resolves against `./models/` first, then falls back to a URL passthrough for `dnnlib.util.open_url`.
- Auto-adds a sibling `stylegan3/` checkout to `sys.path`, so `dnnlib` and `legacy` are importable without setting `PYTHONPATH`.
- `uv`-based project configuration (`pyproject.toml`, `uv.lock`).
- Google-style docstrings across every function; inline comments for non-obvious design decisions (`G_ema` vs `G`, inverse transform for `synthesis.input.transform`, `RandomState` for seed bit-compat, `.part` atomic rename, tensor denormalisation, etc.).
- Getting Started guide and AI-agent onboarding block in `README.md`.

### Changed

- Unknown `--network` names now fail fast with a clear `SystemExit` pointing at `--list-models`, instead of propagating a raw `FileNotFoundError` from `dnnlib.util.open_url`.
- Type annotations migrated from `typing.List/Optional/Tuple/Union` to PEP 585 built-in generics (`list`, `tuple`, `dict`) and PEP 604 union syntax (`X | Y`), enabled on Python 3.9/3.10 via `from __future__ import annotations`. The `typing` module is no longer imported.

### Fixed

- Added `scipy` to project dependencies — StyleGAN3 network pickles import it at unpickle time.
- Pinned `setuptools<80` — upstream `torch_utils.ops.conv2d_gradfix` imports `pkg_resources`, which was removed from setuptools 81+.
