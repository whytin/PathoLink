#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Virchow2 inference for PathoLink.

Extract cell-level image embeddings from HEST h5ad.

Author: Weitian Huang
Email: cswthuang@scut.edu.cn
"""
import argparse
import os

import numpy as np
import scanpy as sc
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Virchow2 inference for HEST cell patches")
    parser.add_argument("--h5ad", type=str, required=True, help="HEST h5ad (with .obsm['img'])")
    parser.add_argument("--output", type=str, required=True, help="output .npy for img_emb [N, 2560]")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ref_he", type=str, default=None, help="reference H&E for Reinhard normalization (optional)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Virchow2
    print("Loading Virchow2...")
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model = model.eval().to(device)
    transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    # Load HEST
    print(f"Loading h5ad: {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    if "img" not in adata.obsm:
        raise KeyError("No 'img' in adata.obsm")
    images = adata.obsm["img"]
    N = images.shape[0]
    print(f"{N} cells loaded")

    # Optional: Reinhard normalization
    mean_ref, std_ref = None, None
    if args.ref_he is not None:
        import histomicstk as htk
        import skimage.io

        im_reference = skimage.io.imread(args.ref_he)[:, :, :3]
        mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(im_reference)
        print("Using Reinhard normalization")

    # Batch inference
    batch_imgs = []
    embeddings_list = []

    def flush_batch():
        if not batch_imgs:
            return
        batch = torch.stack(batch_imgs, 0).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type=args.device, dtype=torch.float16):
            output = model(batch)
        class_token = output[:, 0]
        patch_tokens = output[:, 5:]  # skip register tokens
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        embedding = embedding.to(torch.float16).cpu().numpy()
        embeddings_list.append(embedding)
        batch_imgs.clear()

    for i in tqdm(range(N), desc="Embedding cells"):
        im = images[i]
        if mean_ref is not None and std_ref is not None:
            import histomicstk as htk
            im = htk.preprocessing.color_normalization.reinhard(im, mean_ref, std_ref)
        image = transforms(Image.fromarray(im))
        batch_imgs.append(image)
        if len(batch_imgs) >= args.batch_size:
            flush_batch()

    flush_batch()
    embeddings = np.concatenate(embeddings_list, axis=0).astype(np.float32)  # [N, 2560]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, embeddings)
    print(f"Saved Virchow2 embeddings to {args.output}, shape={embeddings.shape}")


if __name__ == "__main__":
    main()
