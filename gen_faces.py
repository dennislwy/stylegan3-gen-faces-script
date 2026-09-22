"""Generate random frontal face images with a pretrained StyleGAN3 (https://github.com/NVlabs/stylegan3) network.

Each output file is named after the first 10 characters of a ULID (the
time-based, sortable part), e.g. 0A6PTW9HNO.jpg.
"""
import argparse
import os
import re
import secrets
import time
from typing import List, Optional, Tuple, Union

import dnnlib
import legacy
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

_CROCKFORD32 = '0123456789ABCDEFGHJKMNPQRSTVWXYZ'  # ULID alphabet (no I, L, O, U)


def parse_range(s: Union[str, List]) -> List[int]:
    """Parse a comma separated list of numbers or ranges, e.g. '1,2,5-10' -> [1,2,5,6,7,8,9,10]."""
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


def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    """Parse a floating point 2-vector of syntax 'a,b', e.g. '0.3,1' -> (0.3, 1.0)."""
    if isinstance(s, tuple):
        return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise argparse.ArgumentTypeError(f'cannot parse 2-vector {s}')


def jpeg_quality_type(s: str) -> int:
    v = int(s)
    if not 1 <= v <= 100:
        raise argparse.ArgumentTypeError('--jpeg-quality must be between 1 and 100')
    return v


def make_transform(translate: Tuple[float, float], angle: float) -> np.ndarray:
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
    """First 10 chars of a ULID: Crockford base32 of the current ms timestamp."""
    ms = time.time_ns() // 1_000_000
    chars = [''] * 10
    for i in range(9, -1, -1):
        chars[i] = _CROCKFORD32[ms & 0x1F]
        ms >>= 5
    return ''.join(chars)


def humanize_duration(seconds: float) -> str:
    """Format seconds as e.g. '1hrs 2mins 23secs', extending to days for long runs."""
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
    """ULID10-based filename, retrying on the rare same-millisecond collision."""
    while True:
        path = os.path.join(outdir, f'{ulid10()}.{ext}')
        if not os.path.exists(path):
            return path
        time.sleep(0.001)


def resolve_device(device: str) -> torch.device:
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('--device=cuda requested but no CUDA GPU is available.')
    return torch.device(device)


def resolve_network(network: str) -> str:
    """Accept a bare filename (looked up under ./models), a local path, or a URL."""
    if '://' in network or os.path.isfile(network):
        return network
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', network)
    if os.path.isfile(local):
        return local
    return network  # let dnnlib.util.open_url try it as-is (e.g. NGC URL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate face images and save them as <ulid10>.jpg.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:

  # 5 non-repeatable random faces
  python gen_faces.py --outdir=out --num=5

  # Reproducible faces from explicit seeds (same seed -> same face, like gen_images.py)
  python gen_faces.py --outdir=out --seeds=0,1,4-6 --network=stylegan3-r-ffhq-1024x1024.pkl
''')
    parser.add_argument('--network', dest='network_pkl', type=str, default='stylegan3-t-ffhq-1024x1024.pkl',
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
    parser.add_argument('--outdir', type=str, default='.',
                         help='Output directory (created if missing). (default: current directory)')
    parser.add_argument('--jpeg-quality', dest='jpeg_quality', type=jpeg_quality_type, default=70,
                         help='JPEG quality, 1-100. (default: %(default)s)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                         help='Device to run on. (default: %(default)s)')
    parser.add_argument('--translate', type=parse_vec2, default='0,0', metavar='VEC2',
                         help='Translate XY-coordinate (e.g. "0.3,1"). (default: %(default)s)')
    parser.add_argument('--rotate', type=float, default=0, metavar='ANGLE',
                         help='Rotation angle in degrees. (default: %(default)s)')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dev = resolve_device(args.device)
    network_pkl = resolve_network(args.network_pkl)

    print(f'Device: {dev}')
    print(f'Loading network "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(dev)
    label = torch.zeros([1, G.c_dim], device=dev)

    translate = args.translate
    rotate = args.rotate
    if hasattr(G.synthesis, 'input'):
        m = np.linalg.inv(make_transform(translate, rotate))
        G.synthesis.input.transform.copy_(torch.from_numpy(m))
    elif translate != (0, 0) or rotate != 0:
        print('warn: --translate/--rotate ignored; network has no synthesis.input transform.')

    # With --seeds: deterministic, one image per seed (matches gen_images.py's RandomState(seed) scheme).
    # Without: non-repeatable, OS-entropy-seeded latents, --num images.
    seed_list: List[Optional[int]] = list(args.seeds) if args.seeds is not None else [None] * args.num_images

    start_time = time.perf_counter()
    for seed in tqdm(seed_list, desc='Generating', unit='img'):
        if seed is not None:
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim).astype(np.float32)).to(dev)
        else:
            rng = np.random.default_rng(secrets.randbits(128))
            z = torch.from_numpy(rng.standard_normal((1, G.z_dim), dtype=np.float32)).to(dev)
        img = G(z, label, truncation_psi=args.truncation_psi, noise_mode=args.noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_np = img[0].cpu().numpy()
        path = unique_ulid10_path(args.outdir, 'jpg')
        PIL.Image.fromarray(img_np, 'RGB').save(path, 'JPEG', quality=args.jpeg_quality)
    elapsed = time.perf_counter() - start_time
    img_per_sec = len(seed_list) / elapsed if elapsed > 0 else 0.0

    print(f"Done. {len(seed_list)} image(s) written to '{args.outdir}' "
          f"(took {humanize_duration(elapsed)}, {img_per_sec:.2f}img/s)")


if __name__ == '__main__':
    main()
