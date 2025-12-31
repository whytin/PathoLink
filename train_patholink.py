import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from MoE import MoELayer


class PathoLinkDataset(Dataset):
    """预计算 embedding 的数据集。

    期望 .npz 至少包含:
    - img_emb: [N, D_img]
    - gene_emb: [N, G, D_gene]
    - expr: [N, G]
    - cell_type: [N] int
    """

    def __init__(self, path: str):
        super().__init__()
        npz = np.load(path)
        self.img_emb = npz["img_emb"].astype("float32")
        self.gene_emb = npz["gene_emb"].astype("float32")
        self.expr = npz["expr"].astype("float32")
        self.cell_type = npz["cell_type"].astype("int64")
        assert (
            self.img_emb.shape[0]
            == self.gene_emb.shape[0]
            == self.expr.shape[0]
            == self.cell_type.shape[0]
        ), "样本数不一致"
        assert (
            self.gene_emb.shape[1] == self.expr.shape[1]
        ), "gene 维度不一致"

    def __len__(self):
        return self.img_emb.shape[0]

    def __getitem__(self, idx):
        return {
            "img_emb": torch.from_numpy(self.img_emb[idx]),
            "gene_emb": torch.from_numpy(self.gene_emb[idx]),
            "expr": torch.from_numpy(self.expr[idx]),
            "cell_type": torch.tensor(self.cell_type[idx], dtype=torch.long),
        }


class PathoLinkModel(nn.Module):
    """PathoLink 多任务模型：图像 embedding + gene id embedding -> MoE 对齐 + 表达重建 + 细胞类型分类。"""

    def __init__(
        self,
        img_dim: int,
        gene_emb_dim: int,
        num_genes: int,
        num_cell_types: int,
        moe_hidden_dim: int = 1024,
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
    ):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, gene_emb_dim)
        self.gene_id_emb = nn.Parameter(torch.randn(num_genes, gene_emb_dim) * 0.02)
        self.moe = MoELayer(
            input_dim=gene_emb_dim,
            hidden_dim=moe_hidden_dim,
            output_dim=gene_emb_dim,
            num_experts=num_experts,
        )
        self.num_experts_per_tok = num_experts_per_tok
        self.decoder = nn.Linear(gene_emb_dim, 1)
        self.cell_head = nn.Linear(gene_emb_dim, num_cell_types)

    def forward(self, img_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            img_emb: [N, D_img]

        Returns:
            gene_feat: [N, G, D_gene] 经过 MoE 的每个细胞-基因表征
            expr_pred: [N, G] 预测的基因表达
            cell_logits: [N, num_cell_types] 细胞类型 logits
        """
        N = img_emb.size(0)
        img_feat = self.img_proj(img_emb)  # [N, D_gene]
        cell_logits = self.cell_head(img_feat)  # [N, num_cell_types]

        # broadcast 到 [N, G, D_gene]
        img_feat_exp = img_feat.unsqueeze(1)  # [N, 1, D_gene]
        G = self.gene_id_emb.size(0)
        gene_id = self.gene_id_emb.unsqueeze(0).expand(N, G, -1)  # [N, G, D_gene]
        moe_input = img_feat_exp + gene_id  # [N, G, D_gene]

        gene_feat = self.moe(moe_input, num_experts_per_tok=self.num_experts_per_tok)  # [N, G, D_gene]
        expr_pred = self.decoder(gene_feat).squeeze(-1)  # [N, G]
        return gene_feat, expr_pred, cell_logits


class MMDLoss(nn.Module):
    def __init__(self, sigma: float = 1.0, max_samples: Optional[int] = None):
        super().__init__()
        self.sigma = sigma
        self.max_samples = max_samples

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算两个分布之间的 MMD。

        Args:
            x: [N, G, D]
            y: [N, G, D]
        """
        x = x.reshape(-1, x.size(-1))
        y = y.reshape(-1, y.size(-1))

        if self.max_samples is not None and (
            x.size(0) > self.max_samples or y.size(0) > self.max_samples
        ):
            idx_x = torch.randperm(x.size(0), device=x.device)[: self.max_samples]
            idx_y = torch.randperm(y.size(0), device=y.device)[: self.max_samples]
            x = x[idx_x]
            y = y[idx_y]

        xx = self._gaussian_kernel(x, x)
        yy = self._gaussian_kernel(y, y)
        xy = self._gaussian_kernel(x, y)

        mmd = xx.mean() + yy.mean() - 2.0 * xy.mean()
        return mmd

    def _gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist_sq = (diff * diff).sum(-1)
        return torch.exp(-dist_sq / (2.0 * (self.sigma ** 2) + 1e-12))


def build_dataloaders(
    npz_path: str,
    batch_size: int,
    val_ratio: float = 0.1,
    num_workers: int = 4,
) -> tuple[DataLoader, Optional[DataLoader], int, int, int, int, int]:
    dataset = PathoLinkDataset(npz_path)
    N = dataset.img_emb.shape[0]
    D_img = dataset.img_emb.shape[1]
    _, G, D_gene = dataset.gene_emb.shape
    num_cell_types = int(np.unique(dataset.cell_type).shape[0])

    if val_ratio > 0:
        val_size = int(N * val_ratio)
        train_size = N - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_set = dataset
        val_loader = None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, N, G, D_img, D_gene, num_cell_types


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mmd_loss_fn: "MMDLoss",
    expr_loss_fn: nn.Module,
    cls_loss_fn: nn.Module,
    lambda_mmd: float,
    lambda_expr: float,
    lambda_cls: float,
) -> Dict[str, float]:
    model.train()
    total = {"loss": 0.0, "mmd": 0.0, "expr": 0.0, "cls": 0.0}
    nbatches = 0

    for batch in tqdm(loader, desc="train", leave=False):
        img_emb = batch["img_emb"].to(device)
        gene_emb = batch["gene_emb"].to(device)
        expr = batch["expr"].to(device)
        cell_type = batch["cell_type"].to(device)

        optimizer.zero_grad()
        gene_feat, expr_pred, cell_logits = model(img_emb)

        loss_mmd = mmd_loss_fn(gene_feat, gene_emb)
        loss_expr = expr_loss_fn(expr_pred, expr)
        loss_cls = cls_loss_fn(cell_logits, cell_type)

        loss = lambda_mmd * loss_mmd + lambda_expr * loss_expr + lambda_cls * loss_cls
        loss.backward()
        optimizer.step()

        total["loss"] += loss.item()
        total["mmd"] += loss_mmd.item()
        total["expr"] += loss_expr.item()
        total["cls"] += loss_cls.item()
        nbatches += 1

    for k in total:
        total[k] /= max(nbatches, 1)

    return total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
    mmd_loss_fn: "MMDLoss",
    expr_loss_fn: nn.Module,
    cls_loss_fn: nn.Module,
    lambda_mmd: float,
    lambda_expr: float,
    lambda_cls: float,
) -> Optional[Dict[str, float]]:
    if loader is None:
        return None
    model.eval()
    total = {"loss": 0.0, "mmd": 0.0, "expr": 0.0, "cls": 0.0}
    nbatches = 0

    for batch in tqdm(loader, desc="val", leave=False):
        img_emb = batch["img_emb"].to(device)
        gene_emb = batch["gene_emb"].to(device)
        expr = batch["expr"].to(device)
        cell_type = batch["cell_type"].to(device)

        gene_feat, expr_pred, cell_logits = model(img_emb)

        loss_mmd = mmd_loss_fn(gene_feat, gene_emb)
        loss_expr = expr_loss_fn(expr_pred, expr)
        loss_cls = cls_loss_fn(cell_logits, cell_type)
        loss = lambda_mmd * loss_mmd + lambda_expr * loss_expr + lambda_cls * loss_cls

        total["loss"] += loss.item()
        total["mmd"] += loss_mmd.item()
        total["expr"] += loss_expr.item()
        total["cls"] += loss_cls.item()
        nbatches += 1

    for k in total:
        total[k] /= max(nbatches, 1)

    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PathoLink training script (Virchow2 + scGPT embeddings)")
    parser.add_argument("--train_npz", type=str, required=True, help="包含 img_emb, gene_emb, expr, cell_type 的 .npz 文件路径")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="从训练集中划分验证集的比例，0 表示不划分")
    parser.add_argument("--num_workers", type=int, default=4)

    # MoE 超参数
    parser.add_argument("--moe_hidden_dim", type=int, default=1024)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--num_experts_per_tok", type=int, default=2)

    # loss 权重
    parser.add_argument("--lambda_mmd", type=float, default=1.0)
    parser.add_argument("--lambda_expr", type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=1.0)

    # MMD 参数
    parser.add_argument("--mmd_sigma", type=float, default=1.0)
    parser.add_argument("--mmd_max_samples", type=int, default=8192, help="MMD 计算时最多采样的向量数，<=0 表示不采样")

    parser.add_argument("--output_dir", type=str, default="outputs/patholink")
    parser.add_argument("--save_every", type=int, default=10, help="每多少个 epoch 保存一次权重")

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, N, G, D_img, D_gene, num_cell_types = build_dataloaders(
        args.train_npz,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
    )
    print(
        f"Loaded data from {args.train_npz}: N={N}, G={G}, D_img={D_img}, D_gene={D_gene}, num_cell_types={num_cell_types}"
    )

    model = PathoLinkModel(
        img_dim=D_img,
        gene_emb_dim=D_gene,
        num_genes=G,
        num_cell_types=num_cell_types,
        moe_hidden_dim=args.moe_hidden_dim,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    mmd_loss_fn = MMDLoss(
        sigma=args.mmd_sigma,
        max_samples=None if args.mmd_max_samples <= 0 else args.mmd_max_samples,
    )
    expr_loss_fn = nn.MSELoss()
    cls_loss_fn = nn.CrossEntropyLoss()

    best_val = float("inf") if val_loader is not None else None

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            mmd_loss_fn,
            expr_loss_fn,
            cls_loss_fn,
            args.lambda_mmd,
            args.lambda_expr,
            args.lambda_cls,
        )
        print(
            "  train - loss: {:.4f}, mmd: {:.4f}, expr: {:.4f}, cls: {:.4f}".format(
                train_stats["loss"],
                train_stats["mmd"],
                train_stats["expr"],
                train_stats["cls"],
            )
        )

        val_stats = evaluate(
            model,
            val_loader,
            device,
            mmd_loss_fn,
            expr_loss_fn,
            cls_loss_fn,
            args.lambda_mmd,
            args.lambda_expr,
            args.lambda_cls,
        )
        if val_stats is not None:
            print(
                "  val   - loss: {:.4f}, mmd: {:.4f}, expr: {:.4f}, cls: {:.4f}".format(
                    val_stats["loss"],
                    val_stats["mmd"],
                    val_stats["expr"],
                    val_stats["cls"],
                )
            )
            if val_stats["loss"] < best_val:
                best_val = val_stats["loss"]
                ckpt_path = os.path.join(args.output_dir, "best.pt")
                torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)
                print(f"  saved best checkpoint to {ckpt_path}")

        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch}.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)
            print(f"  saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
