import argparse
import os
from typing import Optional

import numpy as np
import scanpy as sc
from scipy.sparse import issparse


def load_array(path: str, key: Optional[str] = None):
    '''Load from .npy or .npz'''
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        if key is None:
            raise ValueError(f"For .npz file {path}, please provide --*_key argument.")
        npz = np.load(path)
        if key not in npz:
            raise KeyError(f"Key '{key}' not found in npz file {path}. Available: {list(npz.keys())}")
        arr = npz[key]
    else:
        raise ValueError(f"Unsupported file extension for {path}, expected .npy or .npz")
    return np.asarray(arr)


def build_npz(
    h5ad_path: str,
    img_emb_path: str,
    gene_emb_path: str,
    cell_type_path: str,
    output_path: str,
    img_emb_key: Optional[str] = None,
    gene_emb_key: Optional[str] = None,
    cell_type_key: Optional[str] = None,
):
    # Load expr from h5ad
    adata = sc.read_h5ad(h5ad_path)
    X = adata.X
    if issparse(X):
        X = X.toarray()
    expr = np.asarray(X, dtype=np.float32)  # [N, G]
    N_expr, G_expr = expr.shape
    print(f"Loaded h5ad: {h5ad_path}, shape={expr.shape}")

    # Load embeddings
    img_emb = load_array(img_emb_path, img_emb_key).astype(np.float32)
    gene_emb = load_array(gene_emb_path, gene_emb_key).astype(np.float32)
    cell_type = load_array(cell_type_path, cell_type_key).astype(np.int64)

    print(f"img_emb: {img_emb.shape}, gene_emb: {gene_emb.shape}, cell_type: {cell_type.shape}")

    # Validate
    if img_emb.shape[0] != N_expr:
        raise ValueError(f"img_emb N mismatch")
    if gene_emb.shape[0] != N_expr:
        raise ValueError(f"gene_emb N={gene_emb.shape[0]} != expr N={N_expr}")
    if gene_emb.shape[1] != G_expr:
        raise ValueError(f"gene_emb G={gene_emb.shape[1]} != expr G={G_expr}")
    if cell_type.shape[0] != N_expr:
        raise ValueError(f"cell_type N={cell_type.shape[0]} != expr N={N_expr}")

    # 3) save to npz
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        img_emb=img_emb,
        gene_emb=gene_emb,
        expr=expr,
        cell_type=cell_type,
    )
    print(f"Saved PathoLink npz to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PathoLink training npz from HEST h5ad + precomputed Virchow2/scGPT embeddings",
    )
    parser.add_argument("--h5ad", type=str, required=True, help="HEST cell x gene h5ad (expr)")
    parser.add_argument("--img_emb", type=str, required=True, help="Virchow2 cell-level image embeddings (.npy or .npz)")
    parser.add_argument("--gene_emb", type=str, required=True, help="scGPT cell-gene embeddings (.npy or .npz)")
    parser.add_argument("--cell_type", type=str, required=True, help="cell type labels (.npy or .npz)")
    parser.add_argument("--output", type=str, required=True, help="output .npz path for PathoLink training")

    parser.add_argument("--img_emb_key", type=str, default=None, help="key inside img_emb .npz (if using .npz)")
    parser.add_argument("--gene_emb_key", type=str, default=None, help="key inside gene_emb .npz (if using .npz)")
    parser.add_argument("--cell_type_key", type=str, default=None, help="key inside cell_type .npz (if using .npz)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_npz(
        h5ad_path=args.h5ad,
        img_emb_path=args.img_emb,
        gene_emb_path=args.gene_emb,
        cell_type_path=args.cell_type,
        output_path=args.output,
        img_emb_key=args.img_emb_key,
        gene_emb_key=args.gene_emb_key,
        cell_type_key=args.cell_type_key,
    )


if __name__ == "__main__":
    main()
