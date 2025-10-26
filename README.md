# PathoLink

**PathoLink** is a multimodal learning framework designed to bridge **histopathological images** and **spatial/single-cell omics data**.  
It leverages **Transformer architectures** and a **Mixture-of-Experts (MoE)** mechanism to enable scalable, interpretable **cross-modal prediction** and **morphology–molecular alignment**.

> 🧬 This repository is the official implementation accompanying our paper *“PathoLink: Bridging Histopathology and Omics via Mixture-of-Experts Transformer”*, which is **currently under review**.  
> The code is under **active development**, and we will continue updating this repository with new modules and pretrained checkpoints — **stay tuned!**

---

## 🌟 Overview

In conventional models, predicting molecular expression (`Y`) from histology (`X`) is constrained by modality-specific redundancy.  
**PathoLink** introduces a synergistic information framework that learns a shared latent representation `Z` bridging both domains — reducing conditional uncertainty and enhancing predictive power:

\[
H(Y|X,Z) < H(Y|X)
\]

This design enables interpretable feature learning and robust cross-modal generation between tissue morphology and omics signals.

---

## 🧩 Repository Structure

```
PathoLink/
├── Virchow2/              # Vision backbone (e.g., patch-level encoder)
├── build/                 # Compiled extensions and build scripts
├── cuda/                  # CUDA kernels for GPU acceleration
├── dataset/HEST/          # Data loading and preprocessing (H&E + spatial transcriptomics)
├── doc/                   # Documentation and supplementary notes
├── fastmoe.egg-info/      # FastMoE package metadata
├── fmoe/                  # Mixture-of-Experts (FastMoE) implementation
├── scGPT/                 # scGPT integration for single-cell/omics representation
├── tests/                 # Unit tests and reproducibility scripts
├── MoE.py                 # Mixture of Experts model wrapper
├── transformer.py         # Transformer architecture definition
└── requirements.txt       # Dependencies
```

---

## ⚙️ Installation

```bash
git clone https://github.com/whytin/PathoLink.git
cd PathoLink

# (Recommended) Create a conda environment
conda create -n patholink python=3.9
conda activate patholink

# Install dependencies
pip install -r requirements.txt
```

---


## 🚀 Usage

## 🧠 Model Architecture

PathoLink integrates three main components:

1. **Virchow2** — Vision encoder for histopathological patches.  
2. **scGPT** — Pretrained omics encoder for molecular representation.  
3. **MoE-Transformer Fusion** — Multi-stage Transformer enhanced with **Mixture-of-Experts** routing for scalable cross-modal fusion and prediction.

The framework follows a synergistic information flow:
```
H&E Image (X) → Latent Bridge (Z) → Omics Prediction (Y)
```

---

## 🧪 Development Status

- [x] Core model implementation 
- [x] scGPT integration
- [ ] Full training scripts
- [ ] Pretrained checkpoints
- [ ] Documentation and visualization tools

> 🔧 *The codebase is being actively updated. Please watch or star the repository to get the latest updates.*

---

> 📢 *PathoLink is under active development — follow the repository for upcoming updates, pretrained models, and experiment results!*
