# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
"""
MaskHadNet: Robustness of HadNet/TSVD/TBCD to Missing Image Data
Extension of: "Tensor decomposition and neural architectures through
generalized Hadamard products" (Panchal, Sahoo, Behera)

"""
import matplotlib.pyplot as plt
import numpy as np
import requests
import io
from PIL import Image

# 1. OUTPUTTING TEST IMAGE

def get_test_image(size=64):
    try:
        r = requests.get("https://picsum.photos/seed/hadnet/128/128", timeout=5)
        img = Image.open(io.BytesIO(r.content)).convert("RGB").resize((size, size))
        return np.array(img).astype(np.float32) / 255.0
    except Exception as e:
        print(f"URL failed, using fallback: {e}")
        x = np.linspace(0, 1, size)
        xx, yy = np.meshgrid(x, x)
        img = np.stack([np.sin(np.pi*xx), np.cos(np.pi*yy), xx*yy], axis=2)
        return (img - img.min()) / (img.max() - img.min())

# 2. RUN AND DISPLAY
X_np = get_test_image(64)

plt.figure(figsize=(4, 4))
plt.imshow(X_np)
plt.title("Original Test Image")
plt.axis('off')
plt.show()

# ─────────────────────────────────────────────
# CONTINUE WITH EXPERIMENT 
# ─────────────────────────────────────────────

# Now switch to Agg for non-interactive saving of experiment results
matplotlib.use("Agg")

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# 1. HADAMARD TRANSFORM & PRODUCT
# ─────────────────────────────────────────────

def T_transform(A):
    """Stacks frontal slices vertically (paper eq. 3)."""
    n1, k, n3 = A.shape
    return A.permute(2, 0, 1).reshape(n1 * n3, k)


def T_inverse(C, n1, n3):
    """C : (n1*n3, n2) -> (n1, n2, n3)"""
    n2 = C.shape[1]
    return C.reshape(n3, n1, n2).permute(1, 2, 0)


def hadamard_product(A, B):
    """
    Generalised Hadamard product A ⊡ B (paper Def 2.3)
    Implemented via slice-wise batch matrix multiplication.
    """
    n1, k, n3 = A.shape
    _, n2, _ = B.shape

    # Block multiply: for each slice i, C_i = A_i @ B_i
    A_slices = A.permute(2, 0, 1)  # (n3, n1, k)
    B_slices = B.permute(2, 0, 1)  # (n3, k, n2)
    C_slices = torch.bmm(A_slices, B_slices)
    return C_slices.permute(1, 2, 0) # (n1, n2, n3)


# ─────────────────────────────────────────────
# 2. ROBUST SVD INITIALISATION
# ─────────────────────────────────────────────

def svd_init(X_np, rank, mask_np=None):
    """
    Initialises W1,H1,W2,H2.
    If mask is provided, fills missing entries with the mean of observed pixels
    to prevent zero-bias in the initial decomposition.
    """
    m, n, p = X_np.shape

    # Robust Step: Mean-fill missing data
    X_init = X_np.copy()
    if mask_np is not None:
        obs_mean = X_np[mask_np > 0].mean() if np.any(mask_np > 0) else 0.5
        X_init[mask_np == 0] = obs_mean

    W1, H1, W2, H2 = [np.zeros((dim, rank, p)) if i % 2 == 0 else np.zeros((rank, dim, p))
                      for i, dim in enumerate([m, n, m, n])]

    for i in range(p):
        Xi = X_init[:, :, i]
        Xs = np.sqrt(np.abs(Xi))
        Xsign = Xs * np.sign(Xi)

        U1, s1, Vt1 = np.linalg.svd(Xs, full_matrices=False)
        U2, s2, Vt2 = np.linalg.svd(Xsign, full_matrices=False)

        sq1 = np.sqrt(s1[:rank])
        sq2 = np.sqrt(s2[:rank])

        W1[:, :len(sq1), i] = U1[:, :rank] * sq1
        H1[:len(sq1), :, i] = sq1[:, None] * Vt1[:rank]
        W2[:, :len(sq2), i] = U2[:, :rank] * sq2
        H2[:len(sq2), :, i] = sq2[:, None] * Vt2[:rank]

    return W1, H1, W2, H2


# ─────────────────────────────────────────────
# 3. TSVD & TBCD (Analytical Baselines)
# ─────────────────────────────────────────────

def tsvd(X_np, rank):
    m, n, p = X_np.shape
    X_approx = np.zeros_like(X_np)
    for i in range(p):
        U, s, Vt = np.linalg.svd(X_np[:, :, i], full_matrices=False)
        r = min(rank, len(s))
        X_approx[:, :, i] = (U[:, :r] * s[:r]) @ Vt[:r]
    return X_approx


def tbcd(X_np, rank, mask_np=None, max_iter=300, tol=1e-6):
    """Analytical Block-Coordinate Descent with fixed precedence."""
    W1, H1, W2, H2 = svd_init(X_np, rank, mask_np)
    m, n, p = X_np.shape
    prev_err = np.inf

    for _ in range(max_iter):
        for i in range(p):
            Xi = X_np[:, :, i]
            w1, h1 = W1[:,:,i], H1[:,:,i]
            w2, h2 = W2[:,:,i], H2[:,:,i]

            # Precompute products to avoid precedence errors
            M1 = w1 @ h1
            M2 = w2 @ h2

            # Update W1
            B = (M2 ** 2) @ (h1.T @ h1) # Parentheses here
            L = np.linalg.norm(B) + 1e-8
            # Use (M1 * M2 - Xi) logic correctly
            grad = ((M1 * M2 - Xi) * M2) @ h1.T
            W1[:,:,i] -= grad / L

            # Update H1
            w1 = W1[:,:,i]
            L = np.linalg.norm(w1.T @ w1) * np.linalg.norm(M2**2) + 1e-8
            grad = w1.T @ ((M1 * M2 - Xi) * M2)
            H1[:,:,i] -= grad / L

            # Recompute M1 after updates
            M1 = W1[:,:,i] @ H1[:,:,i]

            # Update W2
            L = np.linalg.norm((M1**2) @ (h2.T @ h2)) + 1e-8
            grad = ((M1 * M2 - Xi) * M1) @ h2.T
            W2[:,:,i] -= grad / L

            # Update H2
            w2 = W2[:,:,i]
            L = np.linalg.norm(w2.T @ w2) * np.linalg.norm(M1**2) + 1e-8
            grad = w2.T @ ((M1 * M2 - Xi) * M1)
            H2[:,:,i] -= grad / L

        # Convergence check
        X_pred = np.zeros_like(X_np)
        for i in range(p):
            X_pred[:,:,i] = (W1[:,:,i] @ H1[:,:,i]) * (W2[:,:,i] @ H2[:,:,i])

        err = np.linalg.norm(X_np - X_pred)
        if abs(prev_err - err) < tol: break
        prev_err = err

    return X_pred


# ─────────────────────────────────────────────
# 4. HADNET / MASKHADNET (Neural Models)
# ─────────────────────────────────────────────

class HadNet(nn.Module):
    def __init__(self, X_np, rank, mask_np=None):
        super().__init__()
        W1_0, H1_0, W2_0, H2_0 = svd_init(X_np, rank, mask_np)
        self.W1 = nn.Parameter(torch.tensor(W1_0, dtype=torch.float32))
        self.H1 = nn.Parameter(torch.tensor(H1_0, dtype=torch.float32))
        self.W2 = nn.Parameter(torch.tensor(W2_0, dtype=torch.float32))
        self.H2 = nn.Parameter(torch.tensor(H2_0, dtype=torch.float32))

    def forward(self):
        M1 = hadamard_product(self.W1, self.H1)
        M2 = hadamard_product(self.W2, self.H2)
        return M1 * M2


def train_hadnet(X_np, rank, max_epochs=1000, mask=None):
    mask_np = mask.numpy() if mask is not None else None
    X_t = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
    model = HadNet(X_np, rank, mask_np).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    if mask is not None:
        mask_t = mask.to(DEVICE)

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        X_pred = model()

        # Loss calculation: Frobenius norm via torch.norm
        if mask is None:
            loss = torch.norm(X_t - X_pred)
        else:
            loss = torch.norm(mask_t * (X_t - X_pred))

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return model().cpu().numpy()


# ─────────────────────────────────────────────
# 5. UTILITIES (Masks, Metrics, Data)
# ─────────────────────────────────────────────

def random_mask(shape, rate):
    return (torch.rand(shape) > rate).float()

def block_mask(shape, rate):
    m = torch.ones(shape)
    h, w, c = shape
    target = int(rate * h * w * c)
    masked = 0
    while masked < target:
        bh, bw = np.random.randint(h//8, h//3), np.random.randint(w//8, w//3)
        r, c_ = np.random.randint(0, h-bh), np.random.randint(0, w-bw)
        m[r:r+bh, c_:c_+bw, :] = 0
        masked += bh * bw * c
    return m.clamp(0, 1)

def slice_mask(shape, rate):
    m = torch.ones(shape)
    channels = np.random.choice(shape[2], max(1, int(rate * shape[2])), replace=False)
    for ch in channels: m[:, :, ch] = 0
    return m

def compute_metrics(X_true, X_pred):
    X_true, X_pred = np.clip(X_true, 0, 1), np.clip(X_pred, 0, 1)
    # FIX: Flattened norm for 3D tensors
    frob = np.linalg.norm(X_true - X_pred)
    rel  = frob / (np.linalg.norm(X_true) + 1e-8)
    psnr = psnr_metric(X_true, X_pred, data_range=1.0)
    ssim = ssim_metric(X_true, X_pred, data_range=1.0, channel_axis=2)
    return {"frob": frob, "rel": rel, "psnr": psnr, "ssim": ssim}

def get_test_image(size=64):
    try:
        r = requests.get("https://picsum.photos/seed/hadnet/128/128", timeout=5)
        img = Image.open(io.BytesIO(r.content)).convert("RGB").resize((size, size))
        return np.array(img).astype(np.float32) / 255.0
    except:
        x = np.linspace(0, 1, size)
        xx, yy = np.meshgrid(x, x)
        img = np.stack([np.sin(np.pi*xx), np.cos(np.pi*yy), xx*yy], axis=2)
        return (img - img.min()) / (img.max() - img.min())


# ─────────────────────────────────────────────
# 6. TESTING PSNR VALUES FOR MASKHADNET
# ─────────────────────────────────────────────

def main():
    OUT_DIR = "./outputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    RANK, MISSING_RATES = 10, [0.2, 0.4, 0.6]
    MASK_TYPES = ["random", "block", "slice"]
    METHODS = ["TSVD", "TBCD", "HadNet", "MaskHadNet"]

    X_np = get_test_image(64)
    results = {}

    for m_type in MASK_TYPES:
        for rate in MISSING_RATES:
            # Generate mask once for all methods in this slice
            shape = X_np.shape
            if m_type == "random": mask = random_mask(shape, rate)
            elif m_type == "block": mask = block_mask(shape, rate)
            else: mask = slice_mask(shape, rate)

            mask_np = mask.numpy()
            X_corrupt = X_np * mask_np

            for method in METHODS:
                print(f"Running: {m_type} {int(rate*100)}% {method} ... ", end="", flush=True)
                t0 = time.time()

                if method == "TSVD": X_pred = tsvd(X_corrupt, RANK)
                elif method == "TBCD": X_pred = tbcd(X_corrupt, RANK, mask_np)
                elif method == "HadNet": X_pred = train_hadnet(X_corrupt, RANK, 1000, None)
                elif method == "MaskHadNet": X_pred = train_hadnet(X_corrupt, RANK, 1000, mask)

                elapsed = time.time() - t0
                m = compute_metrics(X_np, X_pred)
                results[(m_type, rate, method)] = {**m, "time": elapsed}
                print(f"PSNR: {m['psnr']:.2f}")

    # Output Summary
    print("\n" + "="*70)
    print(f"{'Mask':8s} {'Rate':4s} {'Method':12s} {'PSNR':>7s} {'SSIM':>7s} {'Time':>6s}")
    print("="*70)
    for m_type in MASK_TYPES:
        for rate in MISSING_RATES:
            for method in METHODS:
                res = results[(m_type, rate, method)]
                print(f"{m_type:8s} {int(rate*100):2d}%  {method:12s} {res['psnr']:7.2f} {res['ssim']:7.3f} {res['time']:6.1f}s")

    # Save a simple plot
    plt.figure(figsize=(10,6))
    for method in METHODS:
        psnrs = [results[("random", r, method)]["psnr"] for r in MISSING_RATES]
        plt.plot([20, 40, 60], psnrs, label=method, marker='o')
    plt.title("Random Dropout Recovery: PSNR vs Rate")
    plt.xlabel("Missing Rate %")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.savefig(f"{OUT_DIR}/recovery_plot.png")
    print(f"\nAll results and plots saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
import os, warnings, time
from tqdm import tqdm # Import this at the top

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# 1. CORE MATH & INITIALIZATION
# ─────────────────────────────────────────────

def hadamard_product(A, B):
    A_slices = A.permute(2, 0, 1)
    B_slices = B.permute(2, 0, 1)
    return torch.bmm(A_slices, B_slices).permute(1, 2, 0)

def svd_init(X_np, rank, mask_np=None):
    m, n, p = X_np.shape
    X_init = X_np.copy()
    if mask_np is not None:
        obs_mean = X_np[mask_np > 0].mean() if np.any(mask_np > 0) else 0.5
        X_init[mask_np == 0] = obs_mean

    W1, H1, W2, H2 = [np.zeros((dim, rank, p)) if i % 2 == 0 else np.zeros((rank, dim, p))
                      for i, dim in enumerate([m, n, m, n])]

    for i in range(p):
        Xi = X_init[:, :, i]
        for target_W, target_H, mat in [(W1, H1, np.sqrt(np.abs(Xi))),
                                        (W2, H2, np.sqrt(np.abs(Xi)) * np.sign(Xi))]:
            U, s, Vt = np.linalg.svd(mat, full_matrices=False)
            sq = np.sqrt(s[:rank])
            target_W[:, :len(sq), i] = U[:, :rank] * sq
            target_H[:len(sq), :, i] = sq[:, None] * Vt[:rank]
    return W1, H1, W2, H2

# ─────────────────────────────────────────────
# 2. MODELS & TRAINING
# ─────────────────────────────────────────────

class MaskHadNet(nn.Module):
    def __init__(self, X_np, rank, mask_np=None):
        super().__init__()
        w1, h1, w2, h2 = svd_init(X_np, rank, mask_np)
        self.W1 = nn.Parameter(torch.tensor(w1, dtype=torch.float32))
        self.H1 = nn.Parameter(torch.tensor(h1, dtype=torch.float32))
        self.W2 = nn.Parameter(torch.tensor(w2, dtype=torch.float32))
        self.H2 = nn.Parameter(torch.tensor(h2, dtype=torch.float32))
    def forward(self):
        return hadamard_product(self.W1, self.H1) * hadamard_product(self.W2, self.H2)

def train_nn(X_np, rank, mask=None):
    mask_np = mask.numpy() if mask is not None else None
    model = MaskHadNet(X_np, rank, mask_np).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    X_t = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
    m_t = mask.to(DEVICE) if mask is not None else torch.ones_like(X_t)
    for _ in range(800):
        optimizer.zero_grad()
        loss = torch.norm(m_t * (X_t - model()))
        loss.backward(); optimizer.step()
    return model().detach().cpu().numpy()

def tbcd(X_np, rank, mask_np=None):
    W1, H1, W2, H2 = svd_init(X_np, rank, mask_np)
    for _ in range(150):
        for i in range(X_np.shape[2]):
            Xi = X_np[:, :, i]; M1, M2 = W1[:,:,i] @ H1[:,:,i], W2[:,:,i] @ H2[:,:,i]
            L1 = np.linalg.norm((M2**2) @ (H1[:,:,i].T @ H1[:,:,i])) + 1e-8
            W1[:,:,i] -= (((M1 * M2 - Xi) * M2) @ H1[:,:,i].T) / L1
            M1 = W1[:,:,i] @ H1[:,:,i]
            L2 = np.linalg.norm(W1[:,:,i].T @ W1[:,:,i]) * np.linalg.norm(M2**2) + 1e-8
            H1[:,:,i] -= (W1[:,:,i].T @ ((M1 * M2 - Xi) * M2)) / L2
            M1 = W1[:,:,i] @ H1[:,:,i]
            L3 = np.linalg.norm((M1**2) @ (H2[:,:,i].T @ H2[:,:,i])) + 1e-8
            W2[:,:,i] -= (((M1 * M2 - Xi) * M1) @ H2[:,:,i].T) / L3
            M2 = W2[:,:,i] @ H2[:,:,i]
            L4 = np.linalg.norm(W2[:,:,i].T @ W2[:,:,i]) * np.linalg.norm(M1**2) + 1e-8
            H2[:,:,i] -= (W2[:,:,i].T @ ((M1 * M2 - Xi) * M1)) / L4
    X_out = np.zeros_like(X_np)
    for i in range(X_np.shape[2]): X_out[:,:,i] = (W1[:,:,i] @ H1[:,:,i]) * (W2[:,:,i] @ H2[:,:,i])
    return X_out

# ─────────────────────────────────────────────
# 3. UTILITIES (Masks)
# ─────────────────────────────────────────────

def random_mask(shape, rate):
    return (torch.rand(shape) > rate).float()

def block_mask(shape, rate):
    m = torch.ones(shape)
    h, w, c = shape
    target = int(rate * h * w * c)
    masked = 0
    while masked < target:
        bh, bw = np.random.randint(h//8, h//3), np.random.randint(w//8, w//3)
        r, c_ = np.random.randint(0, h-bh), np.random.randint(0, w-bw)
        m[r:r+bh, c_:c_+bw, :] = 0
        masked += bh * bw * c
    return m.clamp(0, 1)

def slice_mask(shape, rate):
    m = torch.ones(shape)
    # Ensure at least one channel is masked if rate > 0
    num_channels_to_mask = max(1, int(rate * shape[2])) if rate > 0 else 0
    if num_channels_to_mask > shape[2]: num_channels_to_mask = shape[2] # Cap at total channels

    if num_channels_to_mask > 0:
        channels_to_mask = np.random.choice(shape[2], num_channels_to_mask, replace=False)
        for ch in channels_to_mask: m[:, :, ch] = 0
    return m

# ─────────────────────────────────────────────
# 4. EXPERIMENT MAIN
# ─────────────────────────────────────────────

def main():
    FOLDER = "./plots_and_outputs"
    os.makedirs(FOLDER, exist_ok=True)

    RANK, RATES = 10, [0.2, 0.4, 0.6]
    M_TYPES = ["random", "block", "slice"]
    METHODS = ["TSVD", "TBCD", "HadNet", "MaskHadNet"]

    # Image Generation with more channels for better slice mask granularity
    x = np.linspace(0, 1, 64); xx, yy = np.meshgrid(x, x)
    num_channels = 10 # Increased number of channels
    channels_data = []
    for i in range(num_channels):
        channel_data = np.sin(np.pi * (xx + i * 0.05)) + np.cos(np.pi * (yy - i * 0.03))
        channels_data.append(channel_data)
    X_np = np.stack(channels_data, axis=2)
    X_np = (X_np - X_np.min()) / (X_np.max() - X_np.min()) # Normalize to [0, 1]

    results = {}
    header = f"{'Method':12s} | {'Mask':8s} | {'Rate':4s} | {'PSNR':>7s} | {'SSIM':>7s}"
    print(header + "\n" + "-" * len(header))

    for mt in M_TYPES:
        for r in RATES:
            if mt == "random": mask = random_mask(X_np.shape, r)
            elif mt == "block": mask = block_mask(X_np.shape, r)
            else: mask = slice_mask(X_np.shape, r)

            mask_np = mask.numpy(); X_corr = X_np * mask_np

            for meth in METHODS:
                if meth == "TSVD":
                    xp = np.zeros_like(X_corr)
                    for c in range(num_channels): # Loop through all channels
                        U, s_val, Vt = np.linalg.svd(X_corr[:,:,c], False)
                        # Ensure rank is not greater than the smallest dimension of the slice or singular values available
                        effective_rank = min(RANK, s_val.shape[0])
                        xp[:,:,c] = (U[:,:effective_rank] * s_val[:effective_rank]) @ Vt[:effective_rank]
                elif meth == "TBCD": xp = tbcd(X_corr, RANK, mask_np)
                elif meth == "HadNet": xp = train_nn(X_corr, RANK, None)
                else: xp = train_nn(X_corr, RANK, mask)

                xp_c = np.clip(xp, 0, 1)
                psnr_v = psnr_metric(X_np, xp_c, data_range=1.0)
                ssim_v = ssim_metric(X_np, xp_c, data_range=1.0, channel_axis=2)
                results[(mt, r, meth)] = {"psnr": psnr_v, "ssim": ssim_v}
                print(f"{meth:12s} | {mt:8s} | {int(r*100):2d}%  | {psnr_v:7.2f} | {ssim_v:7.3f}")
            print("-" * len(header))
        print("=" * len(header))

    # ─────────────────────────────────────────────
    # 5. ROBUST PLOTTING (Ensures lines appear)
    # ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = {"TSVD": "#1f77b4", "TBCD": "#ff7f0e", "HadNet": "#2ca02c", "MaskHadNet": "#d62728"}

    for i, mt in enumerate(M_TYPES):
        for meth in METHODS:
            # Extract data specifically for this mask and method
            x_axis = [int(r*100) for r in RATES]
            y_axis = [results[(mt, r, meth)]["psnr"] for r in RATES]

            # Explicitly plot lines and markers
            axes[i].plot(x_axis, y_axis, label=meth, color=colors[meth],
                         marker='o', markersize=8, linewidth=2.5, linestyle='-')

        axes[i].set_title(f"{mt.upper()} DROPOUT", fontsize=14, fontweight='bold')
        axes[i].set_xlabel("Missing Rate (%)", fontsize=12)
        axes[i].set_ylabel("PSNR (dB)", fontsize=12)
        axes[i].set_xticks(x_axis)
        axes[i].grid(True, which='both', linestyle='--', alpha=0.5)
        axes[i].legend(loc='best')

    plt.tight_layout()
    plt.savefig(f"{FOLDER}/fig1_complete_lines.png", dpi=200)
    print(f"\nSaved line plots to: {os.path.abspath(FOLDER)}")

if __name__ == "__main__":
    main()

# RANK + MASK
def generate_robustness_heatmap():
    # 1. Setup paths and parameters
    FOLDER = "./plots_and_outputs/heatmap_analysis"
    os.makedirs(FOLDER, exist_ok=True)

    # Define the grid
    RANKS = [2, 5, 10, 15, 20, 30, 40]
    RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 10% to 80% missing

    # Generate Synthetic Data
    x = np.linspace(0, 1, 64); xx, yy = np.meshgrid(x, x)
    X_np = np.stack([np.sin(np.pi*xx), np.cos(np.pi*yy), xx*yy], axis=2)
    X_np = (X_np - X_np.min()) / (X_np.max() - X_np.min())

    # Matrix to store heatmap data
    psnr_grid = np.zeros((len(RATES), len(RANKS)))

    print("Mapping the Robustness Landscape (this may take a few minutes)...")

    for i, r in enumerate(RATES):
        # Create a single consistent mask for this rate to keep it fair
        mask = (torch.rand(X_np.shape) > r).float()
        X_corr = X_np * mask.numpy()

        for j, k in enumerate(RANKS):
            # Train MaskHadNet
            xp = train_nn(X_corr, k, mask) # Uses the train function we built
            xp_c = np.clip(xp, 0, 1)

            psnr_v = psnr_metric(X_np, xp_c, data_range=1.0)
            psnr_grid[i, j] = psnr_v

        print(f"Completed Missing Rate: {int(r*100)}%")

    # 2. Visualisation: The Heatmap
    plt.figure(figsize=(10, 8))
    # Using 'magma' or 'viridis' for high-contrast scientific plotting
    im = plt.imshow(psnr_grid, origin='lower', aspect='auto', cmap='magma')

    # Labeling the axes
    plt.xticks(range(len(RANKS)), RANKS)
    plt.yticks(range(len(RATES)), [f"{int(r*100)}%" for r in RATES])

    plt.xlabel("Tensor Rank (Model Complexity)", fontsize=12, fontweight='bold')
    plt.ylabel("Missing Data Rate (%)", fontsize=12, fontweight='bold')
    plt.title("MaskHadNet: Phase Diagram of Recovery Quality (PSNR)", fontsize=14, pad=20)

    # Add a colorbar with label
    cbar = plt.colorbar(im)
    cbar.set_label('Reconstruction PSNR (dB)', rotation=270, labelpad=15)

    # Add text annotations for each cell (optional but very clear)
    for i in range(len(RATES)):
        for j in range(len(RANKS)):
            plt.text(j, i, f"{psnr_grid[i,j]:.1f}",
                     ha="center", va="center", color="w", fontsize=8)

    plt.savefig(f"{FOLDER}/robustness_heatmap.png", dpi=300)
    print(f"\nPhase Diagram saved to: {FOLDER}")

if __name__ == "__main__":
    generate_robustness_heatmap()

from tqdm import tqdm # Import this at the top
import matplotlib.pyplot as plt

def generate_comparison_heatmaps():
    FOLDER = "./plots_and_outputs/comparison_heatmaps"
    os.makedirs(FOLDER, exist_ok=True)

    RANKS = [2, 5, 10, 20, 40]
    RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    METHODS = ["TSVD", "TBCD", "HadNet", "MaskHadNet"]

    # Generate Synthetic Image
    x = np.linspace(0, 1, 64); xx, yy = np.meshgrid(x, x)
    X_np = np.stack([np.sin(np.pi*xx), np.cos(np.pi*yy), xx*yy], axis=2)
    X_np = (X_np - X_np.min()) / (X_np.max() - X_np.min())

    master_grids = {m: np.zeros((len(RATES), len(RANKS))) for m in METHODS}

    # Total iterations for the progress bar
    total_steps = len(RATES) * len(RANKS) * len(METHODS)

    # Visual Progress Bar
    pbar = tqdm(total=total_steps, desc="Overall Progress")

    for i, r in enumerate(RATES):
        mask = (torch.rand(X_np.shape) > r).float()
        mask_np = mask.numpy()
        X_corr = X_np * mask_np

        for j, k in enumerate(RANKS):
            for meth in METHODS:
                # Update bar description to show what we are working on
                pbar.set_description(f"Processing {meth} | Rate {int(r*100)}% | Rank {k}")

                if meth == "TSVD":
                    xp = np.zeros_like(X_corr)
                    for c in range(3):
                        U, s, Vt = np.linalg.svd(X_corr[:,:,c], full_matrices=False)
                        xp[:,:,c] = (U[:,:k] * s[:k]) @ Vt[:k]
                elif meth == "TBCD":
                    xp = tbcd(X_corr, k, mask_np)
                elif meth == "HadNet":
                    xp = train_nn(X_corr, k, None)
                else:
                    xp = train_nn(X_corr, k, mask)

                psnr_v = psnr_metric(X_np, np.clip(xp, 0, 1), data_range=1.0)
                master_grids[meth][i, j] = psnr_v
                pbar.update(1) # Move the bar forward

    pbar.close() # Finish the bar

    # --- PLOTTING LOGIC (Add this back) ---
    for meth, psnr_grid in master_grids.items():
        plt.figure(figsize=(10, 8))
        im = plt.imshow(psnr_grid, origin='lower', aspect='auto', cmap='magma')

        plt.xticks(range(len(RANKS)), RANKS)
        plt.yticks(range(len(RATES)), [f"{int(r*100)}%" for r in RATES])

        plt.xlabel("Tensor Rank (Model Complexity)", fontsize=12, fontweight='bold')
        plt.ylabel("Missing Data Rate (%)", fontsize=12, fontweight='bold')
        plt.title(f"{meth} Performance: PSNR vs Rank and Missing Rate", fontsize=14, pad=20)

        cbar = plt.colorbar(im)
        cbar.set_label('Reconstruction PSNR (dB)', rotation=270, labelpad=15)

        for i in range(len(RATES)):
            for j in range(len(RANKS)):
                plt.text(j, i, f"{psnr_grid[i,j]:.1f}",
                         ha="center", va="center", color="w", fontsize=8)

        plt.savefig(f"{FOLDER}/{meth.lower()}_heatmap.png", dpi=300)
        plt.close() # Close plot to free memory

    print(f"\nQuad-Comparison complete. Files saved to: {FOLDER}")

if __name__ == "__main__":
    generate_comparison_heatmaps()

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from tqdm import tqdm

def generate_block_mask_heatmap():
    # 1. Setup paths and parameters
    FOLDER = "./plots_and_outputs/block_heatmap_analysis"
    os.makedirs(FOLDER, exist_ok=True)

    # Define the grid
    RANKS = [2, 5, 10, 15, 20, 30, 40]
    RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 10% to 80% missing
    METHODS = ["TSVD", "TBCD", "HadNet", "MaskHadNet"]

    # Generate Synthetic Data
    x = np.linspace(0, 1, 64); xx, yy = np.meshgrid(x, x)
    X_np = np.stack([np.sin(np.pi*xx), np.cos(np.pi*yy), xx*yy], axis=2)
    X_np = (X_np - X_np.min()) / (X_np.max() - X_np.min())

    master_grids = {m: np.zeros((len(RATES), len(RANKS))) for m in METHODS}

    # Total iterations for the progress bar
    total_steps = len(RATES) * len(RANKS) * len(METHODS)
    pbar = tqdm(total=total_steps, desc="Overall Progress (Block Mask)")

    print("\nMapping the Block Mask Robustness Landscape (this may take a few minutes)...")

    for i, r in enumerate(RATES):
        # Create BLOCK mask for this rate
        mask = block_mask(X_np.shape, r)
        mask_np = mask.numpy()
        X_corr = X_np * mask_np

        for j, k in enumerate(RANKS):
            for meth in METHODS:
                pbar.set_description(f"Processing Block Mask | {meth} | Rate {int(r*100)}% | Rank {k}")

                if meth == "TSVD":
                    xp = np.zeros_like(X_corr)
                    for c in range(3):
                        U, s, Vt = np.linalg.svd(X_corr[:,:,c], full_matrices=False)
                        xp[:,:,c] = (U[:,:k] * s[:k]) @ Vt[:k]
                elif meth == "TBCD":
                    xp = tbcd(X_corr, k, mask_np)
                elif meth == "HadNet":
                    xp = train_nn(X_corr, k, None)
                else:
                    xp = train_nn(X_corr, k, mask)

                psnr_v = psnr_metric(X_np, np.clip(xp, 0, 1), data_range=1.0)
                master_grids[meth][i, j] = psnr_v
                pbar.update(1)
    pbar.close()

    # 2. Visualisation: The Heatmaps for all methods
    for meth, psnr_grid in master_grids.items():
        plt.figure(figsize=(10, 8))
        im = plt.imshow(psnr_grid, origin='lower', aspect='auto', cmap='magma')

        # Labeling the axes
        plt.xticks(range(len(RANKS)), RANKS)
        plt.yticks(range(len(RATES)), [f"{int(r*100)}%" for r in RATES])

        plt.xlabel("Tensor Rank (Model Complexity)", fontsize=12, fontweight='bold')
        plt.ylabel("Missing Data Rate (%)", fontsize=12, fontweight='bold')
        plt.title(f"{meth}: Block Mask Recovery Quality (PSNR)", fontsize=14, pad=20)

        # Add a colorbar with label
        cbar = plt.colorbar(im)
        cbar.set_label('Reconstruction PSNR (dB)', rotation=270, labelpad=15)

        # Add text annotations for each cell
        for i in range(len(RATES)):
            for j in range(len(RANKS)):
                plt.text(j, i, f"{psnr_grid[i,j]:.1f}",
                         ha="center", va="center", color="w", fontsize=8)

        plt.savefig(f"{FOLDER}/{meth.lower()}_block_robustness_heatmap.png", dpi=300)
        plt.close() # Close plot to free memory

    print(f"\nBlock Mask Phase Diagrams saved to: {FOLDER}")

if __name__ == "__main__":
    generate_block_mask_heatmap()

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from tqdm import tqdm

def generate_slice_mask_heatmap():
    # 1. Setup paths and parameters
    FOLDER = "./plots_and_outputs/slice_heatmap_analysis_30_slices"
    os.makedirs(FOLDER, exist_ok=True)

    # Define the grid
    RANKS = [2, 5, 10, 15, 20, 30, 40]
    RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 10% to 80% missing
    METHODS = ["TSVD", "TBCD", "HadNet", "MaskHadNet"]

    # Generate Synthetic Data with 30 slices
    x = np.linspace(0, 1, 64)
    xx, yy = np.meshgrid(x, x)
    num_slices = 30 # Changed to 30 slices
    channels = []
    for i in range(num_slices):
        # Create a pattern that varies smoothly across slices
        channel_data = np.sin(np.pi * (xx + i * 0.05)) + np.cos(np.pi * (yy - i * 0.03))
        channels.append(channel_data)
    X_np = np.stack(channels, axis=2)
    X_np = (X_np - X_np.min()) / (X_np.max() - X_np.min()) # Normalize to [0, 1]

    master_grids = {m: np.zeros((len(RATES), len(RANKS))) for m in METHODS}

    # Total iterations for the progress bar
    total_steps = len(RATES) * len(RANKS) * len(METHODS)
    pbar = tqdm(total=total_steps, desc="Overall Progress (Slice Mask, 30 Slices)")

    print("\nMapping the Slice Mask Robustness Landscape (30 Slices, this may take a few minutes)...")

    for i, r in enumerate(RATES):
        # Create SLICE mask for this rate
        mask = slice_mask(X_np.shape, r)
        mask_np = mask.numpy()
        X_corr = X_np * mask_np

        for j, k in enumerate(RANKS):
            for meth in METHODS:
                pbar.set_description(f"Processing Slice Mask (30S) | {meth} | Rate {int(r*100)}% | Rank {k}")

                if meth == "TSVD":
                    xp = np.zeros_like(X_corr)
                    # TSVD is applied per slice, so it works on the individual channels
                    for c in range(num_slices): # Loop through 30 slices
                        U, s, Vt = np.linalg.svd(X_corr[:,:,c], full_matrices=False)
                        xp[:,:,c] = (U[:,:k] * s[:k]) @ Vt[:k]
                elif meth == "TBCD":
                    xp = tbcd(X_corr, k, mask_np)
                elif meth == "HadNet":
                    xp = train_nn(X_corr, k, None)
                else:
                    xp = train_nn(X_corr, k, mask)

                psnr_v = psnr_metric(X_np, np.clip(xp, 0, 1), data_range=1.0)
                master_grids[meth][i, j] = psnr_v
                pbar.update(1)
    pbar.close()

    # 2. Visualisation: The Heatmaps for all methods
    for meth, psnr_grid in master_grids.items():
        plt.figure(figsize=(10, 8))
        im = plt.imshow(psnr_grid, origin='lower', aspect='auto', cmap='magma')

        # Labeling the axes
        plt.xticks(range(len(RANKS)), RANKS)
        plt.yticks(range(len(RATES)), [f"{int(r*100)}%" for r in RATES])

        plt.xlabel("Tensor Rank (Model Complexity)", fontsize=12, fontweight='bold')
        plt.ylabel("Missing Data Rate (%)", fontsize=12, fontweight='bold')
        plt.title(f"{meth}: Slice Mask Recovery Quality (PSNR) - 30 Slices", fontsize=14, pad=20)

        # Add a colorbar with label
        cbar = plt.colorbar(im)
        cbar.set_label('Reconstruction PSNR (dB)', rotation=270, labelpad=15)

        # Add text annotations for each cell
        for i in range(len(RATES)):
            for j in range(len(RANKS)):
                plt.text(j, i, f"{psnr_grid[i,j]:.1f}",
                         ha="center", va="center", color="w", fontsize=8)

        plt.savefig(f"{FOLDER}/{meth.lower()}_slice_robustness_heatmap_30_slices.png", dpi=300)
        plt.close() # Close plot to free memory

    print(f"\nSlice Mask Phase Diagrams (30 Slices) saved to: {FOLDER}")

if __name__ == "__main__":
    # The main() function from the previous cell should already be run. No need to call again.
    # main() # Commenting out to avoid redundant execution
    generate_slice_mask_heatmap()
