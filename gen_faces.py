"""Generate random face images with a pretrained StyleGAN3 network.

Wraps the upstream https://github.com/NVlabs/stylegan3 generator behind a small
CLI. Two modes of image generation:

* ``--seeds`` — deterministic, one image per seed. Matches upstream
  ``gen_images.py``'s ``np.random.RandomState(seed)`` scheme so the same seed
  yields the same face.
* ``--num`` — non-repeatable, OS-entropy-seeded latents (``secrets.randbits``).

Each output file is named after the first 10 characters of a ULID (the
time-based, sortable part), e.g. ``0A6PTW9HNO.jpg`` — so files sort
chronologically and collisions within a millisecond are avoided by a retry.

Convenience subcommands (each exits before touching Torch):

* ``--list-models``   — fetch NGC file list; print available pickles.
* ``--download-model NAME`` — download a pickle into ``./models/``.
"""
# Enable PEP 604 (``X | Y``) union syntax on Python 3.9/3.10 by making all
# annotations lazy strings. Together with PEP 585 built-in generics (``list``,
# ``tuple``, ``dict``) this removes the need for ``from typing import ...``.
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import time
import urllib.request

NGC_FILES_URL = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files'

# Make a sibling `stylegan3/` checkout importable without requiring PYTHONPATH.
_STYLEGAN3_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan3')
if os.path.isdir(_STYLEGAN3_DIR) and _STYLEGAN3_DIR not in sys.path:
    sys.path.insert(0, _STYLEGAN3_DIR)

import dnnlib
import legacy
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

_CROCKFORD32 = '0123456789ABCDEFGHJKMNPQRSTVWXYZ'  # ULID alphabet (no I, L, O, U)


def parse_range(s: str | list) -> list[int]:
    """Expand a comma/range spec into an explicit list of integers.

    Accepts a string like ``'1,2,5-10'`` and returns ``[1, 2, 5, 6, 7, 8, 9, 10]``.
    Ranges are inclusive on both ends. If ``s`` is already a list, it is
    returned unchanged so this function can be reused as an ``argparse``
    ``type`` and also called on already-parsed input.

    Args:
        s: Either the raw CLI string, or an already-materialised list of ints.

    Returns:
        A list of integers in the order they appear in the input.
    """
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


def parse_vec2(s: str | tuple[float, float]) -> tuple[float, float]:
    """Parse a 2-vector of the form ``'a,b'`` into ``(float, float)``.

    Used as the ``argparse`` type for ``--translate``. If a tuple is passed
    (already-parsed value), it is returned unchanged.

    Args:
        s: Either the raw CLI string ``'x,y'`` or an already-parsed tuple.

    Returns:
        The parsed ``(x, y)`` tuple.

    Raises:
        argparse.ArgumentTypeError: If ``s`` does not split into exactly two
            comma-separated numeric parts.
    """
    if isinstance(s, tuple):
        return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise argparse.ArgumentTypeError(f'cannot parse 2-vector {s}')


def jpeg_quality_type(s: str) -> int:
    """Argparse validator for ``--jpeg-quality``: an int in ``[1, 100]``.

    Args:
        s: The raw CLI string.

    Returns:
        The validated integer quality value.

    Raises:
        argparse.ArgumentTypeError: If ``s`` is not an integer, or is outside
            the ``[1, 100]`` range that Pillow's JPEG encoder accepts.
    """
    v = int(s)
    if not 1 <= v <= 100:
        raise argparse.ArgumentTypeError('--jpeg-quality must be between 1 and 100')
    return v


def make_transform(translate: tuple[float, float], angle: float) -> np.ndarray:
    """Build a 3×3 affine matrix for StyleGAN3's ``synthesis.input.transform``.

    The result composes a rotation by ``angle`` (degrees) with a translation by
    ``translate``. The caller is expected to invert this matrix before copying
    it into the network — StyleGAN3 stores the *inverse* image-space transform.

    Args:
        translate: ``(x, y)`` translation in the network's input frame.
        angle: Rotation angle, in degrees.

    Returns:
        A ``(3, 3)`` ``np.ndarray`` in the standard [R | t; 0 0 1] layout.
    """
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


def ulid10() -> str:
    """Return the first 10 characters of a ULID.

    Those 10 characters encode the current millisecond timestamp in Crockford
    Base32 (the sortable, time-based prefix of a full ULID). The 50 bits of
    precision cover ~35,000 years, so lexicographic order matches chronological
    order for any realistic timestamp.

    Two calls made within the same millisecond will return the same value;
    callers that need uniqueness should use :func:`unique_ulid10_path` instead.

    Returns:
        A 10-character Crockford Base32 string, e.g. ``'01M33T914B'``.
    """
    ms = time.time_ns() // 1_000_000
    chars = [''] * 10
    for i in range(9, -1, -1):
        chars[i] = _CROCKFORD32[ms & 0x1F]
        ms >>= 5
    return ''.join(chars)


def humanize_duration(seconds: float) -> str:
    """Format a duration as ``'<d>days <h>hrs <m>mins <s>secs'``.

    Leading units that are zero are dropped, but every unit below the largest
    non-zero one is kept — so ``60`` → ``'1mins 0secs'`` and ``3661`` →
    ``'1hrs 1mins 1secs'``. Days are only shown for runs of >= 24 hours.

    Args:
        seconds: Elapsed time in seconds; fractional values are rounded to the
            nearest whole second.

    Returns:
        A human-readable string.
    """
    total = int(round(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f'{days}days')
    if hours or parts:
        parts.append(f'{hours}hrs')
    if mins or parts:
        parts.append(f'{mins}mins')
    parts.append(f'{secs}secs')
    return ' '.join(parts)


def unique_ulid10_path(outdir: str, ext: str) -> str:
    """Return a not-yet-existing ``<ulid10>.<ext>`` path under ``outdir``.

    On the rare same-millisecond collision (two calls within the same 1 ms
    tick), the function sleeps 1 ms so the next :func:`ulid10` tick differs,
    then retries. In the current generator loop this only matters when the
    per-image time falls below 1 ms.

    Args:
        outdir: Directory the file will live in (must already exist).
        ext: Extension without the leading dot (e.g. ``'jpg'``).

    Returns:
        An absolute or relative path suitable for immediate use with
        ``PIL.Image.save``.
    """
    while True:
        path = os.path.join(outdir, f'{ulid10()}.{ext}')
        if not os.path.exists(path):
            return path
        time.sleep(0.001)


def resolve_device(device: str) -> torch.device:
    """Turn the ``--device`` CLI value into a ``torch.device``.

    ``'auto'`` picks CUDA if available, otherwise CPU. ``'cuda'`` is treated
    as a hard requirement — the process exits rather than silently falling
    back to CPU, so users don't discover their run took hours on the wrong
    hardware.

    Args:
        device: One of ``'auto'``, ``'cpu'``, or ``'cuda'``.

    Returns:
        The concrete ``torch.device`` to run the generator on.

    Raises:
        SystemExit: If the user explicitly asked for ``'cuda'`` but no CUDA
            GPU is available.
    """
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('--device=cuda requested but no CUDA GPU is available.')
    return torch.device(device)


def _models_dir() -> str:
    """Return the absolute path to the ``./models/`` directory next to this script.

    Resolved from ``__file__`` so the location is stable regardless of the
    caller's current working directory.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


def _fetch_ngc_pkls() -> list[dict]:
    """Fetch the NGC file listing and keep only top-level ``*.pkl`` entries.

    The NGC bundle also contains ``metrics/inception-*.pkl`` and
    ``metrics/vgg16.pkl``, which are training/metric helpers and not usable
    generator networks; the ``'/' not in f['path']`` guard excludes them.

    Returns:
        A list of ``{'path': str, 'sizeInBytes': int, ...}`` dicts (raw NGC
        entries), sorted by name for stable, human-scannable output.
    """
    with urllib.request.urlopen(NGC_FILES_URL, timeout=30) as resp:
        data = json.load(resp)
    return sorted(
        (f for f in data.get('modelFiles', [])
         if f['path'].endswith('.pkl') and '/' not in f['path']),
        key=lambda f: f['path'],
    )


def list_models(as_json: bool = False) -> None:
    """Print available StyleGAN3 pickles on NGC to stdout.

    Contacts NGC to get the canonical list, then reports each pickle's name
    and size. Local ``./models/`` is checked so already-downloaded pickles
    can be flagged (``*`` marker in tabular mode, ``downloaded: true`` in
    JSON mode).

    Args:
        as_json: If True, emit a JSON array (name, sizeBytes, url, downloaded)
            suitable for scripting/agents. Otherwise, emit a padded table plus
            a summary line.
    """
    files = _fetch_ngc_pkls()
    models_dir = _models_dir()

    if as_json:
        payload = [
            {
                'name': f['path'],
                'sizeBytes': f['sizeInBytes'],
                'url': f'{NGC_FILES_URL}/{f["path"]}',
                'downloaded': os.path.isfile(os.path.join(models_dir, f['path'])),
            }
            for f in files
        ]
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write('\n')
        return

    print(f'{"":2}  {"NAME":40} {"SIZE":>8}')
    for f in files:
        name = f['path']
        size_mb = f['sizeInBytes'] / (1024 * 1024)
        mark = '*' if os.path.isfile(os.path.join(models_dir, name)) else ' '
        print(f'{mark:2}  {name:40} {size_mb:>6.1f}M')
    present = sum(1 for f in files if os.path.isfile(os.path.join(models_dir, f['path'])))
    print(f'\n({present} of {len(files)} present in ./models; * = already downloaded)')


def download_model(name: str) -> None:
    """Fetch a StyleGAN3 pickle from NGC into ``./models/``.

    Downloads to ``<name>.pkl.part`` first, then atomically renames on
    success. This guarantees a killed or failed download never leaves a
    truncated file that would later look valid to :func:`resolve_network`.

    Args:
        name: The pickle name, with or without the ``.pkl`` extension
            (e.g. ``'stylegan3-r-ffhqu-256x256'``).

    Raises:
        urllib.error.HTTPError: If NGC returns a non-2xx status (typically
            404 for a misspelled name).
    """
    if not name.endswith('.pkl'):
        name = name + '.pkl'
    models_dir = _models_dir()
    os.makedirs(models_dir, exist_ok=True)
    dest = os.path.join(models_dir, name)
    if os.path.isfile(dest):
        # No --force flag: assume anything already at the destination is
        # intact (the .part → dest rename below ensures that invariant).
        print(f'Already downloaded: {dest}')
        return

    url = f'{NGC_FILES_URL}/{name}'
    print(f'Downloading {url}')
    with urllib.request.urlopen(url, timeout=60) as resp:
        # Content-Length may be absent (chunked encoding) — pass None so tqdm
        # falls back to an untotaled progress bar rather than showing 0%.
        total = int(resp.headers.get('Content-Length') or 0)
        tmp = dest + '.part'
        with open(tmp, 'wb') as out, tqdm(
            total=total or None, unit='B', unit_scale=True, unit_divisor=1024, desc=name
        ) as bar:
            # 64 KiB chunks: large enough to keep syscall overhead trivial,
            # small enough to give the progress bar frequent updates.
            while chunk := resp.read(1 << 16):
                out.write(chunk)
                bar.update(len(chunk))
    os.replace(tmp, dest)
    print(f'Saved to {dest}')


def resolve_network(network: str) -> str:
    """Turn a ``--network`` argument into something ``dnnlib.util.open_url`` can load.

    Resolution order:
        1. If the value already looks like a URL or points at an existing
           file, return it unchanged.
        2. Otherwise treat it as a bare name and probe ``./models/<name>``.
           If the caller omitted the ``.pkl`` extension, ``./models/<name>.pkl``
           is also tried.
        3. If still not found, query NGC's file listing. When the name matches
           a known pickle, download it into ``./models/`` and return that path.
        4. Fall through and return the raw value so ``dnnlib.util.open_url``
           can attempt it as a URL — preserves upstream behaviour and lets
           users pass a raw NGC URL if they want.

    Args:
        network: The raw value of ``--network``.

    Returns:
        A URL or an absolute/relative filesystem path.
    """
    if '://' in network or os.path.isfile(network):
        return network
    models_dir = _models_dir()
    candidates = (network,) if network.endswith('.pkl') else (network, network + '.pkl')
    for candidate in candidates:
        local = os.path.join(models_dir, candidate)
        if os.path.isfile(local):
            return local

    # Not found locally: check NGC and auto-download if it's a known pickle.
    pkl_name = network if network.endswith('.pkl') else network + '.pkl'
    try:
        available = {f['path'] for f in _fetch_ngc_pkls()}
    except Exception as e:
        print(f'warn: could not query NGC listing ({e}); passing "{network}" through as-is.')
        return network
    if pkl_name in available:
        print(f'Network "{pkl_name}" not found in {models_dir}; fetching from NGC...')
        download_model(pkl_name)
        return os.path.join(models_dir, pkl_name)
    raise SystemExit(
        f'--network "{network}": not a URL, no such file locally, and not '
        f'listed on NGC. Run "gen_faces.py --list-models" to see valid names.'
    )


def parse_args() -> argparse.Namespace:
    """Define and parse the CLI.

    Returns:
        The ``argparse.Namespace`` produced by ``ArgumentParser.parse_args``.
    """
    parser = argparse.ArgumentParser(
        description='Generate face images and save them as <ulid10>.jpg.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:

  # 5 non-repeatable random faces
  python gen_faces.py --outdir=out --num=5

  # Reproducible faces from explicit seeds (same seed -> same face, like gen_images.py)
  python gen_faces.py --outdir=out --seeds=0,1,4-6 --network=stylegan3-r-ffhq-1024x1024
''')
    parser.add_argument('--network', dest='network_pkl', type=str, default='stylegan3-r-ffhqu-256x256',
                         help='Network pickle: filename under ./models, a local path, or a URL. (default: %(default)s)')
    parser.add_argument('--seeds', type=parse_range, default=None,
                         help="List/range of seeds for reproducible output (e.g. '0,1,4-6'), one image per seed. "
                              'Omit for non-repeatable random faces (use --num to set the count).')
    parser.add_argument('--num', dest='num_images', type=int, default=1,
                         help='Number of images to generate. Ignored if --seeds is given (len(seeds) is used instead). (default: %(default)s)')
    parser.add_argument('--trunc', dest='truncation_psi', type=float, default=0.6,
                         help='Truncation psi. (default: %(default)s)')
    parser.add_argument('--noise-mode', choices=['const', 'random', 'none'], default='const',
                         help='Noise mode. (default: %(default)s)')
    parser.add_argument('--outdir', type=str, default='./out',
                         help='Output directory (created if missing). (default: %(default)s)')
    parser.add_argument('--jpeg-quality', dest='jpeg_quality', type=jpeg_quality_type, default=70,
                         help='JPEG quality, 1-100. (default: %(default)s)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                         help='Device to run on. (default: %(default)s)')
    parser.add_argument('--translate', type=parse_vec2, default='0,0', metavar='VEC2',
                         help='Translate XY-coordinate (e.g. "0.3,1"). (default: %(default)s)')
    parser.add_argument('--rotate', type=float, default=0, metavar='ANGLE',
                         help='Rotation angle in degrees. (default: %(default)s)')
    parser.add_argument('--list-models', action='store_true',
                         help='Fetch the NGC file list, print available StyleGAN3 pickles (marking any already in ./models), then exit.')
    parser.add_argument('--json', action='store_true',
                         help='With --list-models, emit machine-readable JSON instead of a table.')
    parser.add_argument('--download-model', metavar='NAME', default=None,
                         help='Download a pickle from NGC into ./models/ (the .pkl extension is optional), then exit.')
    return parser.parse_args()


def main():
    """CLI entry point: dispatch to a subcommand or run the generator loop.

    Subcommand flags (``--list-models``, ``--download-model``) short-circuit
    before any Torch/dnnlib work so they run fast and don't need a GPU or a
    downloaded pickle. The default path loads the requested network, builds
    latents according to ``--seeds`` / ``--num``, and writes one JPEG per
    output into ``--outdir``.
    """
    args = parse_args()
    if args.list_models:
        list_models(as_json=args.json)
        return
    if args.download_model:
        download_model(args.download_model)
        return
    os.makedirs(args.outdir, exist_ok=True)
    dev = resolve_device(args.device)
    network_pkl = resolve_network(args.network_pkl)

    print(f'Device: {dev}')
    print(f'Loading network "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        # G_ema (exponential-moving-average generator) is what StyleGAN3
        # ships for inference — the plain 'G' is only used during training.
        G = legacy.load_network_pkl(f)['G_ema'].to(dev)
    # FFHQ-style pickles are unconditional (c_dim == 0), so this is an empty
    # label tensor — kept because G.forward() always requires the argument.
    label = torch.zeros([1, G.c_dim], device=dev)

    translate = args.translate
    rotate = args.rotate
    if hasattr(G.synthesis, 'input'):
        # synthesis.input.transform stores the *inverse* image-space affine,
        # so we invert make_transform() before assigning.
        m = np.linalg.inv(make_transform(translate, rotate))
        G.synthesis.input.transform.copy_(torch.from_numpy(m))
    elif translate != (0, 0) or rotate != 0:
        # Config-T (translation-equivariant) networks have the transform
        # attribute; older / non-equivariant configs don't. Warn rather than
        # silently ignore so the user notices the flag was a no-op.
        print('warn: --translate/--rotate ignored; network has no synthesis.input transform.')

    # With --seeds: deterministic, one image per seed (matches gen_images.py's RandomState(seed) scheme).
    # Without: non-repeatable, OS-entropy-seeded latents, --num images.
    seed_list: list[int | None] = list(args.seeds) if args.seeds is not None else [None] * args.num_images

    start_time = time.perf_counter()
    for seed in tqdm(seed_list, desc='Generating', unit='img'):
        if seed is not None:
            # RandomState (legacy API) is required for bit-for-bit
            # compatibility with upstream gen_images.py — swapping to
            # default_rng here would produce different faces for the same seed.
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim).astype(np.float32)).to(dev)
        else:
            # secrets.randbits gives a full 128-bit OS-entropy seed, so
            # consecutive same-second runs cannot collide.
            rng = np.random.default_rng(secrets.randbits(128))
            z = torch.from_numpy(rng.standard_normal((1, G.z_dim), dtype=np.float32)).to(dev)
        img = G(z, label, truncation_psi=args.truncation_psi, noise_mode=args.noise_mode)
        # StyleGAN3 emits NCHW in [-1, 1]; convert to HWC uint8 [0, 255] for PIL.
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_np = img[0].cpu().numpy()
        path = unique_ulid10_path(args.outdir, 'jpg')
        PIL.Image.fromarray(img_np, 'RGB').save(path, 'JPEG', quality=args.jpeg_quality)
    elapsed = time.perf_counter() - start_time
    img_per_sec = len(seed_list) / elapsed if elapsed > 0 else 0.0

    print(f"Done. {len(seed_list)} image(s) written to '{args.outdir}' "
          f"(took {humanize_duration(elapsed)}, {img_per_sec:.2f} img/s)")


if __name__ == '__main__':
    main()
