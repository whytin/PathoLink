"""
PathoLink 结果可视化脚本

功能：
- 空间转录数据可视化
- UMAP 降维可视化
- 聚类结果可视化
- SSIM 计算与可视化
- PCC (Pearson 相关系数) 计算与可视化
"""
import argparse
import os
from typing import Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def visualize_spatial(
    adata: sc.AnnData,
    genes: list,
    output_dir: str,
    spot_size: float = 1.0,
    save_prefix: str = "spatial",
):
    """可视化空间转录数据"""
    print(f"Visualizing {len(genes)} genes spatially...")
    os.makedirs(output_dir, exist_ok=True)
    
    for gene in tqdm(genes, desc="Spatial plots"):
        if gene not in adata.var_names:
            print(f"Warning: {gene} not in adata.var_names, skipping")
            continue
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        sc.pl.spatial(
            adata,
            color=gene,
            spot_size=spot_size,
            ax=ax,
            show=False,
            title=f"{gene} expression",
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{save_prefix}_{gene}.png"), dpi=150, bbox_inches="tight")
        plt.close()
    
    print(f"Saved spatial plots to {output_dir}")


def visualize_umap(
    adata: sc.AnnData,
    color_by: list,
    output_dir: str,
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    save_prefix: str = "umap",
):
    """UMAP 降维可视化"""
    print("Computing UMAP...")
    os.makedirs(output_dir, exist_ok=True)
    
    if "X_pca" not in adata.obsm:
        print("Computing PCA...")
        sc.pp.pca(adata, n_comps=50)
    
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=50)
    sc.tl.umap(adata, min_dist=min_dist)
    
    for color in color_by:
        if color not in adata.obs.columns and color not in adata.var_names:
            print(f"Warning: {color} not found, skipping")
            continue
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        sc.pl.umap(adata, color=color, ax=ax, show=False, title=f"UMAP colored by {color}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{save_prefix}_{color}.png"), dpi=150, bbox_inches="tight")
        plt.close()
    
    print(f"Saved UMAP plots to {output_dir}")


def visualize_clustering(
    adata: sc.AnnData,
    output_dir: str,
    resolution: float = 1.0,
    save_prefix: str = "cluster",
):
    """聚类结果可视化（Leiden + spatial + UMAP）"""
    print(f"Computing Leiden clustering (resolution={resolution})...")
    os.makedirs(output_dir, exist_ok=True)
    
    if "neighbors" not in adata.uns:
        print("Computing neighbors...")
        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata, n_comps=50)
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
    
    sc.tl.leiden(adata, resolution=resolution, key_added="leiden")
    
    # UMAP 聚类
    if "X_umap" not in adata.obsm:
        sc.tl.umap(adata)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    sc.pl.umap(adata, color="leiden", ax=ax, show=False, title="UMAP - Leiden clusters")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{save_prefix}_umap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # Spatial 聚类
    if "spatial" in adata.obsm:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        sc.pl.spatial(adata, color="leiden", ax=ax, show=False, title="Spatial - Leiden clusters")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{save_prefix}_spatial.png"), dpi=150, bbox_inches="tight")
        plt.close()
    
    print(f"Saved clustering plots to {output_dir}")


def compute_ssim(
    expr_true: np.ndarray,
    expr_pred: np.ndarray,
    output_path: Optional[str] = None,
) -> Dict[str, float]:
    """计算 SSIM (Structural Similarity Index)
    
    Args:
        expr_true: [N, G] ground truth
        expr_pred: [N, G] predictions
        output_path: 保存 SSIM 分布图
    
    Returns:
        dict with 'ssim_mean', 'ssim_std', 'ssim_per_gene'
    """
    print("Computing SSIM...")
    N, G = expr_true.shape
    assert expr_pred.shape == (N, G), f"Shape mismatch: {expr_pred.shape} != {expr_true.shape}"
    
    # Normalize to [0, 1] for SSIM
    expr_true_norm = (expr_true - expr_true.min()) / (expr_true.max() - expr_true.min() + 1e-12)
    expr_pred_norm = (expr_pred - expr_pred.min()) / (expr_pred.max() - expr_pred.min() + 1e-12)
    
    ssim_per_gene = []
    for g in tqdm(range(G), desc="SSIM per gene"):
        # SSIM expects 2D images; treat each gene as 1D signal reshaped
        true_vec = expr_true_norm[:, g].reshape(-1, 1)
        pred_vec = expr_pred_norm[:, g].reshape(-1, 1)
        
        # For 1D, we compute correlation-like SSIM
        s = ssim(true_vec, pred_vec, data_range=1.0, win_size=min(7, N))
        ssim_per_gene.append(s)
    
    ssim_per_gene = np.array(ssim_per_gene)
    ssim_mean = float(np.mean(ssim_per_gene))
    ssim_std = float(np.std(ssim_per_gene))
    
    print(f"SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.hist(ssim_per_gene, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(ssim_mean, color="red", linestyle="--", label=f"Mean: {ssim_mean:.4f}")
        ax.set_xlabel("SSIM")
        ax.set_ylabel("Frequency")
        ax.set_title(f"SSIM Distribution (mean={ssim_mean:.4f}, std={ssim_std:.4f})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved SSIM plot to {output_path}")
    
    return {
        "ssim_mean": ssim_mean,
        "ssim_std": ssim_std,
        "ssim_per_gene": ssim_per_gene.tolist(),
    }


def compute_pcc(
    expr_true: np.ndarray,
    expr_pred: np.ndarray,
    output_path: Optional[str] = None,
    gene_names: Optional[list] = None,
) -> Dict[str, float]:
    """计算 Pearson 相关系数 (PCC)
    
    Args:
        expr_true: [N, G] ground truth
        expr_pred: [N, G] predictions
        output_path: 保存 PCC 分布图
        gene_names: 基因名列表
    
    Returns:
        dict with 'pcc_mean', 'pcc_std', 'pcc_per_gene'
    """
    print("Computing PCC...")
    N, G = expr_true.shape
    assert expr_pred.shape == (N, G), f"Shape mismatch: {expr_pred.shape} != {expr_true.shape}"
    
    pcc_per_gene = []
    for g in tqdm(range(G), desc="PCC per gene"):
        true_vec = expr_true[:, g]
        pred_vec = expr_pred[:, g]
        
        if np.std(true_vec) < 1e-12 or np.std(pred_vec) < 1e-12:
            pcc_per_gene.append(0.0)
        else:
            corr, _ = pearsonr(true_vec, pred_vec)
            pcc_per_gene.append(corr if not np.isnan(corr) else 0.0)
    
    pcc_per_gene = np.array(pcc_per_gene)
    pcc_mean = float(np.mean(pcc_per_gene))
    pcc_std = float(np.std(pcc_per_gene))
    
    print(f"PCC: {pcc_mean:.4f} ± {pcc_std:.4f}")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Histogram
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(pcc_per_gene, bins=50, edgecolor="black", alpha=0.7)
        axes[0].axvline(pcc_mean, color="red", linestyle="--", label=f"Mean: {pcc_mean:.4f}")
        axes[0].set_xlabel("PCC")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title(f"PCC Distribution (mean={pcc_mean:.4f}, std={pcc_std:.4f})")
        axes[0].legend()
        
        # Top/Bottom genes
        top_k = min(20, G)
        sorted_idx = np.argsort(pcc_per_gene)
        bottom_idx = sorted_idx[:top_k]
        top_idx = sorted_idx[-top_k:][::-1]
        
        if gene_names is not None:
            bottom_genes = [gene_names[i] for i in bottom_idx]
            top_genes = [gene_names[i] for i in top_idx]
        else:
            bottom_genes = [f"Gene_{i}" for i in bottom_idx]
            top_genes = [f"Gene_{i}" for i in top_idx]
        
        axes[1].barh(range(top_k), pcc_per_gene[top_idx], color="green", alpha=0.7, label="Top genes")
        axes[1].set_yticks(range(top_k))
        axes[1].set_yticklabels(top_genes, fontsize=8)
        axes[1].set_xlabel("PCC")
        axes[1].set_title(f"Top {top_k} genes by PCC")
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved PCC plot to {output_path}")
    
    return {
        "pcc_mean": pcc_mean,
        "pcc_std": pcc_std,
        "pcc_per_gene": pcc_per_gene.tolist(),
    }


def visualize_gene_correlation_heatmap(
    expr_true: np.ndarray,
    expr_pred: np.ndarray,
    output_path: str,
    gene_names: Optional[list] = None,
    n_genes: int = 50,
):
    """可视化真实与预测表达的基因相关性热图"""
    print(f"Computing gene correlation heatmap (top {n_genes} genes)...")
    N, G = expr_true.shape
    
    # 计算每个基因的 variance，选 top-k
    var_true = np.var(expr_true, axis=0)
    top_idx = np.argsort(var_true)[-n_genes:]
    
    expr_true_sub = expr_true[:, top_idx]
    expr_pred_sub = expr_pred[:, top_idx]
    
    # 计算 correlation matrix
    corr_true = np.corrcoef(expr_true_sub.T)
    corr_pred = np.corrcoef(expr_pred_sub.T)
    
    if gene_names is not None:
        labels = [gene_names[i] for i in top_idx]
    else:
        labels = [f"G{i}" for i in top_idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.heatmap(corr_true, ax=axes[0], cmap="coolwarm", vmin=-1, vmax=1, cbar=True, square=True)
    axes[0].set_title("True Gene Correlation")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    sns.heatmap(corr_pred, ax=axes[1], cmap="coolwarm", vmin=-1, vmax=1, cbar=True, square=True)
    axes[1].set_title("Predicted Gene Correlation")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    diff = corr_pred - corr_true
    sns.heatmap(diff, ax=axes[2], cmap="bwr", vmin=-1, vmax=1, cbar=True, square=True)
    axes[2].set_title("Difference (Pred - True)")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved gene correlation heatmap to {output_path}")


def visualize_scatter_pred_vs_true(
    expr_true: np.ndarray,
    expr_pred: np.ndarray,
    output_path: str,
    n_samples: int = 5000,
):
    """散点图：真实 vs 预测表达"""
    print("Plotting scatter: true vs pred...")
    N, G = expr_true.shape
    
    # 随机采样避免过多点
    if N * G > n_samples:
        idx = np.random.choice(N * G, n_samples, replace=False)
        true_flat = expr_true.flatten()[idx]
        pred_flat = expr_pred.flatten()[idx]
    else:
        true_flat = expr_true.flatten()
        pred_flat = expr_pred.flatten()
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.scatter(true_flat, pred_flat, s=1, alpha=0.3, c="blue")
    ax.plot([true_flat.min(), true_flat.max()], [true_flat.min(), true_flat.max()], "r--", lw=2, label="y=x")
    ax.set_xlabel("True Expression")
    ax.set_ylabel("Predicted Expression")
    ax.set_title("True vs Predicted Expression")
    ax.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter plot to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PathoLink 结果可视化")
    
    # 数据路径
    parser.add_argument("--h5ad", type=str, help="HEST h5ad 文件（用于空间/UMAP/聚类可视化）")
    parser.add_argument("--expr_true", type=str, help="ground truth expression .npy [N, G]")
    parser.add_argument("--expr_pred", type=str, help="predicted expression .npy [N, G]")
    parser.add_argument("--gene_names", type=str, default=None, help="gene names .txt（每行一个基因名）")
    parser.add_argument("--output_dir", type=str, default="visualizations")
    
    # 可视化选项
    parser.add_argument("--spatial_genes", type=str, nargs="+", help="空间可视化的基因列表")
    parser.add_argument("--umap_color", type=str, nargs="+", default=["leiden"], help="UMAP 着色变量")
    parser.add_argument("--clustering_resolution", type=float, default=1.0)
    parser.add_argument("--compute_ssim", action="store_true", help="计算 SSIM")
    parser.add_argument("--compute_pcc", action="store_true", help="计算 PCC")
    parser.add_argument("--gene_corr_heatmap", action="store_true", help="绘制基因相关性热图")
    parser.add_argument("--scatter_plot", action="store_true", help="绘制 true vs pred 散点图")
    
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load gene names
    gene_names = None
    if args.gene_names is not None:
        with open(args.gene_names, "r") as f:
            gene_names = [line.strip() for line in f if line.strip()]
    
    # ========== h5ad 相关可视化 ==========
    if args.h5ad is not None:
        print(f"Loading h5ad: {args.h5ad}")
        adata = sc.read_h5ad(args.h5ad)
        
        # 空间转录可视化
        if args.spatial_genes is not None:
            visualize_spatial(
                adata,
                args.spatial_genes,
                os.path.join(args.output_dir, "spatial"),
            )
        
        # UMAP
        visualize_umap(
            adata,
            args.umap_color,
            os.path.join(args.output_dir, "umap"),
        )
        
        # 聚类
        visualize_clustering(
            adata,
            os.path.join(args.output_dir, "clustering"),
            resolution=args.clustering_resolution,
        )
    
    # ========== 表达预测评估 ==========
    if args.expr_true is not None and args.expr_pred is not None:
        print(f"Loading expression arrays...")
        expr_true = np.load(args.expr_true).astype(np.float32)
        expr_pred = np.load(args.expr_pred).astype(np.float32)
        
        print(f"expr_true: {expr_true.shape}, expr_pred: {expr_pred.shape}")
        
        # SSIM
        if args.compute_ssim:
            ssim_res = compute_ssim(
                expr_true,
                expr_pred,
                os.path.join(args.output_dir, "ssim_distribution.png"),
            )
            import json
            with open(os.path.join(args.output_dir, "ssim_results.json"), "w") as f:
                json.dump(ssim_res, f, indent=2)
        
        # PCC
        if args.compute_pcc:
            pcc_res = compute_pcc(
                expr_true,
                expr_pred,
                os.path.join(args.output_dir, "pcc_distribution.png"),
                gene_names=gene_names,
            )
            import json
            with open(os.path.join(args.output_dir, "pcc_results.json"), "w") as f:
                json.dump(pcc_res, f, indent=2)
        
        # 基因相关性热图
        if args.gene_corr_heatmap:
            visualize_gene_correlation_heatmap(
                expr_true,
                expr_pred,
                os.path.join(args.output_dir, "gene_correlation_heatmap.png"),
                gene_names=gene_names,
            )
        
        # 散点图
        if args.scatter_plot:
            visualize_scatter_pred_vs_true(
                expr_true,
                expr_pred,
                os.path.join(args.output_dir, "scatter_true_vs_pred.png"),
            )
    
    print(f"\nAll visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
