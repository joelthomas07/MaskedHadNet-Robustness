# MaskedHadNet-Robustness: Tensor Completion via Generalized Hadamard Products

A robust implementation and extension of the **HadNet** architecture, specifically designed for image recovery and tensor completion in the presence of extreme data loss. This project benchmarks the efficacy of a **Masked Loss** approach against standard analytical and neural tensor decomposition methods.

---

## 🚀 Project Overview
**MaskHadNet** is an unsupervised neural tensor decomposition model. It is an extension of the work by **Panchal, Sahoo, and Behera** in their paper, *"Tensor decomposition and neural architectures through generalized Hadamard products."*

While the original HadNet is designed for tensor approximation, this project introduces a binary mask into the loss function to enable **robust tensor completion**. By ignoring missing or corrupted pixels during optimization, the model leverages its low-rank Hadamard bottleneck to infer the underlying signal rather than memorizing the corruption.

---

## 🧠 Core Methodology
The model approximates a 3D target tensor $X$ as the element-wise product of two factorized components:

$$X \approx (W_1 \boxdot H_1) \odot (W_2 \boxdot H_2)$$

*   **$\boxdot$ (Generalized Hadamard Product):** Slice-wise matrix multiplication.
*   **$\odot$ (Hadamard Product):** Element-wise multiplication.
*   **Masked Loss:** The Frobenius norm is calculated only over observed indices $\Omega$:
    $$\mathcal{L} = \| \text{Mask} \odot (X_{true} - X_{pred}) \|_F$$

---

## 📊 Experimental Results
The benchmarks compare **MaskHadNet** against **TSVD** (Truncated SVD), **TBCD** (Tensor Block Coordinate Descent), and standard **HadNet**.

### Key Findings:
*   **Resilience to Loss:** At **80% data loss**, MaskHadNet maintains a PSNR of **~44.1 dB**, whereas the standard HadNet crashes to **~4.8 dB**.
*   **Method Comparison:** In random dropout tests, MaskHadNet consistently outperforms baselines by a margin of **+6 dB to +30 dB** depending on the missing rate.
*   **Adaptability:** The model successfully recovers data across various corruption types, including:
    *   **Random Mask:** Pixel-level dropout (sensor noise).
    *   **Block Mask:** Spatial occlusions (physical barriers).
    *   **Slice Mask:** Channel-wise failure (broken sensors).

---

## 📂 Repository Structure
The results are categorically stored in the `outputs/` directory:
```text
├── maskhadnet.py        # Core implementation and experiment script
├── README.md            # Project documentation
└── outputs/
    ├── random_mask/      # Heatmaps for pixel-level dropout
    ├── block_mask/       # Heatmaps for spatial occlusions
    ├── slice_mask/       # Heatmaps for channel-wise failure
    └── comparisons/      # PSNR vs. Missing Rate line graphs

## 📜 Acknowledgments

This research implementation and robustness study is built upon the mathematical foundations established in:

> **Panchal, S., Sahoo, J. K., & Behera, R.**
> *"Tensor decomposition and neural architectures through generalized Hadamard products"*
> Research conducted at **IISc Bangalore** and **BITS Pilani, K.K. Birla Goa Campus**.

### My Contributions:
*   **Masked Loss Extension**: Developed the `MaskHadNet` variant by integrating a binary-weighted Frobenius norm to enable true tensor completion.
*   **Robustness Benchmarking**: Conducted a phase-analysis study across three specific failure modes: Random, Block, and Slice-wise corruption.
*   **Analytical Comparison**: Evaluated neural methods against traditional analytical baselines (TSVD and TBCD) to prove the superior inductive bias of Hadamard-based neural layers.

---
Special thanks to Prof. JK Sahoo under whom I did this project. He and his co-authors did innovative work on the generalized Hadamard product and its application to neural architectures, which I was able to extend on to
an analysis when missing data is introduced into the idea.
