# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/), 
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-22

### Added

- `gen_faces.py` CLI for generating face images from a pretrained StyleGAN3
  network. Supports `--num` (non-repeatable, OS-entropy-seeded) and `--seeds`
  (deterministic, bit-compatible with upstream `gen_images.py`) generation
  modes.
- Generation flags: `--outdir`, `--network`, `--trunc`, `--noise-mode`,
  `--jpeg-quality`, `--device` (`cpu`/`cuda`/`auto`), `--translate`,
  `--rotate`.
- Output filenames use a 10-character ULID prefix (Crockford Base32 of the
  current-ms timestamp) so files sort chronologically and never collide.
- `--list-models`: prints available StyleGAN3 pickles on NGC and marks any
  already present in `./models/`.
- `--list-models --json`: emits a machine-readable JSON array (name,
  sizeBytes, url, downloaded).
- `--download-model NAME`: fetches a pickle from NGC into `./models/`
  with a `tqdm` progress bar; writes to `<name>.pkl.part` and atomically
  renames on success. Accepts the name with or without `.pkl`.
- `--network` accepts a bare filename with or without the `.pkl`
  extension; resolves against `./models/` first, then falls back to a
  URL passthrough for `dnnlib.util.open_url`.
- Auto-adds a sibling `stylegan3/` checkout to `sys.path`, so `dnnlib`
  and `legacy` are importable without setting `PYTHONPATH`.
- `uv`-based project configuration (`pyproject.toml`, `uv.lock`).
- Google-style docstrings across every function; inline comments for
  non-obvious design decisions (`G_ema` vs `G`, inverse transform for
  `synthesis.input.transform`, `RandomState` for seed bit-compat,
  `.part` atomic rename, tensor denormalisation, etc.).
- Getting Started guide and AI-agent onboarding block in `README.md`.

### Changed

- Type annotations migrated from `typing.List/Optional/Tuple/Union` to
  PEP 585 built-in generics (`list`, `tuple`, `dict`) and PEP 604 union
  syntax (`X | Y`), enabled on Python 3.9/3.10 via
  `from __future__ import annotations`. The `typing` module is no longer
  imported.

### Fixed

- Added `scipy` to project dependencies — StyleGAN3 network pickles
  import it at unpickle time.
- Pinned `setuptools<80` — upstream `torch_utils.ops.conv2d_gradfix`
  imports `pkg_resources`, which was removed from setuptools 81+.

[Unreleased]: https://github.com/dennislwy/stylegan3-gen-faces-script/compare/HEAD...HEAD
