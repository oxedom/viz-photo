from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import open_clip
import umap


def list_images(root: Path, exts: set[str] = {'.jpg', '.jpeg', '.png', '.webp'}) -> list[Path]:
    """Collect image paths under root."""
    return sorted([p for p in root.rglob('*') if p.suffix.lower() in exts])


def load_model(arch: str, ckpt: str) -> tuple[torch.nn.Module, callable, str]:
    """Load OpenCLIP model + matching preprocess, choose device."""
    model, preprocess = open_clip.create_model_from_pretrained(arch, pretrained=ckpt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval().to(device)
    return model, preprocess, device


def embed_paths(
    paths: list[Path],
    model: torch.nn.Module,
    preprocess: callable,
    device: str,
    batch_size: int = 32
) -> np.ndarray:
    """Return L2-normalized embeddings: float32 array [N, D]."""
    zs: list[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch_size), desc='embedding'):
            batch_paths = paths[i:i + batch_size]
            xs: list[torch.Tensor] = []
            for p in batch_paths:
                x = preprocess(Image.open(p).convert('RGB')).unsqueeze(0)     # [1,3,H,W]
                xs.append(x)
            xb = torch.cat(xs, dim=0).to(device)                              # [B,3,H,W]
            zb = model.encode_image(xb)                                       # [B,D]
            zb = zb / zb.norm(dim=-1, keepdim=True)
            zs.append(zb.cpu().numpy().astype('float32'))
    return np.concatenate(zs, axis=0)                                         # [N,D]


def reduce_umap(
    vectors: np.ndarray,
    components: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42
) -> np.ndarray:
    """UMAP to k-D; normalized to [-1,1] per axis."""
    reducer = umap.UMAP(n_components=components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
    y = reducer.fit_transform(vectors).astype('float32')
    y -= y.min(axis=0, keepdims=True)
    span = y.max(axis=0, keepdims=True)
    span[span == 0] = 1.0
    y /= span
    y = (y - 0.5) * 2.0
    return y


def make_thumbs(paths: list[Path], out_dir: Path, size: int = 64, fill: str = 'black') -> list[Path]:
    """Create square thumbnails; return paths in same order as inputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs: list[Path] = []
    for p in tqdm(paths, desc='thumbnails'):
        img = Image.open(p).convert('RGB')
        img.thumbnail((size, size), Image.LANCZOS)
        canvas = Image.new('RGB', (size, size), fill)
        ox = (size - img.width) // 2
        oy = (size - img.height) // 2
        canvas.paste(img, (ox, oy))
        tp = out_dir / f'{p.stem}_thumb.jpg'
        canvas.save(tp, quality=85, optimize=True)
        thumbs.append(tp)
    return thumbs


def pack_atlas(
    thumbs: list[Path],
    tile_size: int,
    gutter: int,
    atlas_max: int = 4096
) -> dict:
    """Pack thumbs on a fixed grid atlas. Return atlas PIL image + per-tile UV info."""
    tiles_per_axis = atlas_max // (tile_size + 2 * gutter)
    capacity = tiles_per_axis * tiles_per_axis
    if len(thumbs) > capacity:
        raise ValueError(f'too many thumbs for atlas: {len(thumbs)} > {capacity}')

    atlas = Image.new('RGB', (atlas_max, atlas_max), 'black')
    tiles: list[dict] = []
    for i, t in enumerate(tqdm(thumbs, desc='atlas pack')):
        x = (i % tiles_per_axis) * (tile_size + 2 * gutter) + gutter
        y = (i // tiles_per_axis) * (tile_size + 2 * gutter) + gutter
        im = Image.open(t).convert('RGB')
        atlas.paste(im, (x, y))

        u0 = x / atlas_max
        v0 = y / atlas_max
        us = tile_size / atlas_max
        vs = tile_size / atlas_max

        tiles.append({
            'tileIndex': i,
            'uvOffset': [float(u0), float(v0)],
            'uvScale':  [float(us), float(vs)]
        })

    return {'atlas_image': atlas, 'tiles': tiles, 'tiles_per_axis': tiles_per_axis}


def save_manifest(
    paths: list[Path],
    tiles: list[dict],
    out_dir: Path,
    tile_size: int,
    atlas_size: int,
    gutter: int
) -> Path:
    """Write manifest.json describing each image and its atlas UVs."""
    items: list[dict] = []
    for i, p in enumerate(paths):
        items.append({
            'id': p.stem,
            'name': p.name,
            'tileIndex': tiles[i]['tileIndex'],
            'uvOffset': tiles[i]['uvOffset'],
            'uvScale':  tiles[i]['uvScale']
        })
    manifest = {
        'version': 1,
        'imageCount': len(paths),
        'tileSize': tile_size,
        'atlasSize': atlas_size,
        'gutter': gutter,
        'items': items
    }
    path = out_dir / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    return path


def build_dataset(
    images_root: Path,
    out_root: Path,
    arch: str = 'ViT-B-16',
    ckpt: str = 'laion2b_s34b_b88k',
    tile_size: int = 64,
    gutter: int = 2,
    batch_size: int = 32,
    do_2d: bool = True
) -> dict:
    """End-to-end: embeddings → UMAP(3D/2D) → thumbs → atlas → manifest + binaries."""
    out_root.mkdir(parents=True, exist_ok=True)

    paths = list_images(images_root)
    if not paths:
        raise RuntimeError(f'no images found under {images_root}')

    model, preprocess, device = load_model(arch, ckpt)

    embs = embed_paths(paths, model, preprocess, device, batch_size=batch_size)      # [N,D]
    np.save(out_root / 'embeddings.npy', embs)

    coords3d = reduce_umap(embs, components=3)
    coords3d.tofile(out_root / 'coords3d.bin')

    if do_2d:
        coords2d = reduce_umap(embs, components=2)
        coords2d.tofile(out_root / 'coords2d.bin')

    thumbs = make_thumbs(paths, out_root / 'thumbs', size=tile_size)
    atlas_info = pack_atlas(thumbs, tile_size=tile_size, gutter=gutter)
    atlas_path = out_root / 'thumbs_atlas.png'
    atlas_info['atlas_image'].save(atlas_path, optimize=True)

    manifest_path = save_manifest(
        paths=paths,
        tiles=atlas_info['tiles'],
        out_dir=out_root,
        tile_size=tile_size,
        atlas_size=atlas_info['atlas_image'].size[0],
        gutter=gutter
    )

    print(f'images: {len(paths)} | emb shape: {embs.shape} | atlas: {atlas_path.name} | out: {out_root}')
    return {
        'embeddings': str(out_root / 'embeddings.npy'),
        'coords3d': str(out_root / 'coords3d.bin'),
        'coords2d': str(out_root / 'coords2d.bin') if do_2d else '',
        'atlas': str(atlas_path),
        'manifest': str(manifest_path)
    }


if __name__ == '__main__':
    project = Path(__file__).resolve().parent
    images_dir = project / 'images'
    out_dir = project / 'out'
    build_dataset(images_dir, out_dir)
