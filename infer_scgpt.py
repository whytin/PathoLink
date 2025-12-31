"""scGPT inference for PathoLink.

Extract gene embeddings and cell type labels.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

# Adjust import paths if needed
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scGPT"))

from scgpt.data_collator import DataCollator
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.utils import load_pretrained


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="scGPT inference for PathoLink")
    parser.add_argument("--h5ad", type=str, required=True, help="HEST h5ad (cell x gene)")
    parser.add_argument("--model_dir", type=str, required=True, help="scGPT model directory (with vocab.json, args.json, best_model.pt)")
    parser.add_argument("--gene_col", type=str, default="index", help="column in adata.var for gene names")
    parser.add_argument("--output_gene_emb", type=str, required=True, help="output .npy for gene_emb [N, G, D_gene]")
    parser.add_argument("--output_cell_type", type=str, required=True, help="output .npy for cell_type [N] int")
    parser.add_argument("--max_length", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_fast_transformer", action="store_true", help="use flash-attn if available")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load h5ad
    print(f"Loading h5ad: {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    if args.gene_col == "index":
        adata.var["index"] = adata.var.index
    else:
        assert args.gene_col in adata.var, f"gene_col {args.gene_col} not in adata.var"

    # Load scGPT model & vocab
    model_dir = Path(args.model_dir)
    vocab_file = model_dir / "vocab.json"
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    adata.var["id_in_vocab"] = [vocab[gene] if gene in vocab else -1 for gene in adata.var[args.gene_col]]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    print(f"Matched {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocab of size {len(vocab)}")
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    vocab.set_default_index(vocab["<pad>"])
    genes = adata.var[args.gene_col].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)

    # Build model
    print("Building model...")
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=model_configs["dropout"],
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=args.use_fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=False,
    )
    load_pretrained(model, torch.load(model_file, map_location=device), verbose=False)
    model.to(device)
    model.eval()

    # Prepare dataset
    count_matrix = adata.X
    if hasattr(count_matrix, "toarray"):
        count_matrix = count_matrix.toarray()
    count_matrix = np.asarray(count_matrix, dtype=np.float32)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, count_matrix_, gene_ids_):
            self.count_matrix = count_matrix_
            self.gene_ids = gene_ids_

        def __len__(self):
            return len(self.count_matrix)

        def __getitem__(self, idx):
            row = self.count_matrix[idx]
            values = row
            genes = self.gene_ids
            genes = np.insert(genes, 0, vocab["<cls>"])
            values = np.insert(values, 0, model_configs["pad_value"])
            genes = torch.from_numpy(genes).long()
            values = torch.from_numpy(values).float()
            return {"id": idx, "genes": genes, "expressions": values}

    dataset = Dataset(count_matrix, gene_ids)
    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[model_configs["pad_token"]],
        pad_value=model_configs["pad_value"],
        do_mlm=False,
        do_binning=True,
        max_length=args.max_length,
        sampling=True,
        keep_first_n_tokens=1,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=collator,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), args.batch_size),
        pin_memory=True,
    )

    # Inference: get all-gene embeddings [N, max_length, embsize]
    N = len(dataset)
    D_gene = model_configs["embsize"]
    gene_embeddings = np.zeros((N, args.max_length, D_gene), dtype=np.float32)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        count = 0
        for data_dict in tqdm(data_loader, desc="Embedding"):
            input_gene_ids = data_dict["gene"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[model_configs["pad_token"]])
            embeddings = model._encode(
                input_gene_ids,
                data_dict["expr"].to(device),
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
            )
            embeddings = embeddings.cpu().numpy()  # [B, max_length, embsize]
            gene_embeddings[count : count + len(embeddings)] = embeddings
            count += len(embeddings)

    # Normalize
    gene_embeddings = gene_embeddings / (np.linalg.norm(gene_embeddings, axis=2, keepdims=True) + 1e-12)

    # Trim to actual gene count G
    G = len(gene_ids)
    gene_embeddings = gene_embeddings[:, 1 : G + 1, :]  # skip cls token

    # TODO: Replace with actual cell type prediction
    cell_type = np.zeros(N, dtype=np.int64)
    print(f"Warning: cell_type all zeros, need real classifier")

    # Save
    os.makedirs(os.path.dirname(args.output_gene_emb) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_cell_type) or ".", exist_ok=True)
    np.save(args.output_gene_emb, gene_embeddings)
    np.save(args.output_cell_type, cell_type)
    print(f"Saved gene_emb to {args.output_gene_emb}, shape={gene_embeddings.shape}")
    print(f"Saved cell_type to {args.output_cell_type}, shape={cell_type.shape}")


if __name__ == "__main__":
    main()
