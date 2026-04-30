# MaskedHadNet-Robustness: Tensor Completion via Generalized Hadamard Products

A robust extension of the **HadNet** architecture, specifically designed for tensor completion and recovery in the presence of extreme data loss. This project addresses a key future research direction identified by Panchal et al. (2025) by benchmarking the efficacy of a **Masked Loss** approach against fundamental tensor decomposition methods.

## 1. Project Overview

**MaskHadNet** is an unsupervised neural tensor decomposition model. It is an extension of the work by **Panchal, Sahoo, and Behera** in their paper, *"Tensor decomposition and neural architectures through generalized Hadamard products"* (2025).

While the original HadNet is optimized for low-rank tensor approximation, this project introduces a binary mask into the loss function to enable **robust tensor completion**. By ignoring corrupted or missing pixels during optimization, the model leverages its low-rank Hadamard bottleneck to infer the underlying signal—successfully fulfilling the authors' call for future research into incomplete and corrupted data analysis.

## 2. Core Methodology

The model approximates a target tensor $\mathcal{X}$ as the element-wise product of two factorized components using the **Generalized Hadamard Product ($\square$)**:

$$\mathcal{X} \approx (\mathcal{W}_1 \square \mathcal{H}_1) \circ (\mathcal{W}_2 \square \mathcal{H}_2)$$

*   **$\square$ (Generalized Hadamard Product):** Implemented via the Hadamard transformation operator $T$, which enables block-wise multiplication across tensor slices.
*   **$\circ$ (Hadamard Product):** Standard element-wise multiplication.
*   **Masked Loss Function:** To facilitate completion, the Frobenius norm is calculated exclusively over observed indices $\Omega$:
$$\mathcal{L} = \| \text{Mask} \odot (\mathcal{X}_{true} - \mathcal{X}_{pred}) \|_F$$

## 3. Experimental Results

The benchmarks compare MaskHadNet against three primary baselines defined in the source paper: **TSVD** (Tensor Singular Value Decomposition), **TBCD** (Tensor Block-Coordinate Descent), and standard **HadNet**.

### Quantitative Analysis and Key Findings

A phase-space analysis across varying Tensor Ranks (2 to 40) and Missing Data Rates (10% to 80%) reveals a stark contrast in the inductive biases of these methods:

*   **Catastrophic Overfitting vs. Extreme Resilience:** At 80% sparsity, standard HadNet experiences total signal collapse (PSNR ~4.2–5.0 dB) as it converges on the corruption. MaskHadNet maintains high-fidelity reconstruction (37.9–44.1 dB), proving the masked loss successfully decouples signal from noise.
*   **The Model Complexity Paradox:** Increasing rank inversely impacts standard approximation on corrupted data. MaskHadNet inverts this: at low sparsity (10%), higher ranks capture finer detail (peak 47.8 dB), while at high sparsity (80%), lower ranks act as a regularizer to prevent data hallucination.
*   **Analytical Baseline Benchmarking:** Analytical baselines like **TSVD** (~15.1 dB) and **TBCD** (~16.8 dB) marginally outperform standard HadNet because they utilize initial mean-filling. However, MaskHadNet establishes a superior performance tier (~41.7–45.3 dB), proving neural optimization with masked loss is strictly superior to iterative analytical descent for completion tasks.
*   **Spatial and Channel Recovery:** MaskHadNet successfully "inpaints" large spatial blocks (Block Mask) and reconstructs entirely missing color channels (Slice Mask) by learning cross-slice mathematical correlations.

### Performance Summary

| Mask Type | Corruption Scenario | HadNet (Standard) | MaskedHadNet (Extension) | Improvement (dB) |
| :--- | :--- | :--- | :--- | :--- |
| **1. Random Mask** | Pixel-level Dropout | ~15.02 dB | ~21.30 dB | +6.28 dB |
| **2. Block Mask** | Spatial Occlusion | ~6.80 dB | ~44.10 dB | +37.30 dB |
| **3. Slice Mask** | Channel Failure | ~5.00 dB | ~44.10 dB | +39.10 dB |

## 4. Acknowledgments

This research implementation and robustness study is built upon the mathematical foundations established in:

> **Panchal, A., Sahoo, J. K., & Behera, R.** 
> *"Tensor decomposition and neural architectures through generalized Hadamard products"* 
> Department of Mathematics, BITS Pilani KK Birla Goa Campus.

Special thanks to **Prof. J.K. Sahoo**, under whose guidance this project was conducted. The innovative work on generalized Hadamard products provided the essential foundation for this analysis of model robustness in missing data scenarios.

## 5. Repository Structure
The results are stored categorically in the `outputs/` directory:
├── maskhadnet.py        # Core implementation and experiment script
├── README.md            # Project documentation
└── outputs/
    ├── random_mask/      # Heatmaps for pixel-level dropout
    ├── block_mask/       # Heatmaps for spatial occlusions
    ├── slice_mask/       # Heatmaps for channel-wise failure
    └── comparisons/      # PSNR vs. Missing Rate line graphs
