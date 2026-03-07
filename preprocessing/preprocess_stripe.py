import numpy as np
import cv2
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


# ---------------------------
# Utilities: 1D vertical box filter (fast) for column windows
# ---------------------------
def boxfilter_cols(x: np.ndarray, r: int) -> np.ndarray:
    """
    1D box filter along rows (vertical) for each column.
    Window size = 2r+1.
    """
    h, w = x.shape
    c = np.cumsum(x, axis=0)

    out = np.empty_like(x, dtype=np.float32)

    # top
    out[:r+1, :] = c[r:2*r+1, :]
    # middle
    out[r+1:h-r, :] = c[2*r+1:, :] - c[:-2*r-1, :]
    # bottom
    out[h-r:, :] = c[h-1:h, :] - c[h-2*r-1:h-r-1, :]

    return out


def mean_cols(x: np.ndarray, r: int) -> np.ndarray:
    h, w = x.shape
    N = boxfilter_cols(np.ones((h, w), dtype=np.float32), r)
    return boxfilter_cols(x.astype(np.float32), r) / (N + 1e-8), N


# ---------------------------
# Step 1: BLF-LS Smoothing (Eq. 1-2)
# ---------------------------
def blf_ls_smooth(g_u8: np.ndarray, lam: float = 1024.0, sigma_s: float = 2.0, sigma_r: float = 0.04) -> np.ndarray:
    """
    Implements Eq. (1)-(2):
      min_u Σ_p (u_p - g_p)^2 + λ Σ_{q in {x,y}} (∇u_{p,q} - f_BLF(∇g_q)_p)^2

    Where f_BLF is bilateral filtering on the gradient images.
    Solves the resulting linear system with a 5-point Laplacian.
    """
    g = g_u8.astype(np.float32) / 255.0
    h, w = g.shape
    n = h * w

    # Forward gradients of g (∇g_x, ∇g_y)
    gx = np.zeros_like(g, dtype=np.float32)
    gy = np.zeros_like(g, dtype=np.float32)
    gx[:, :-1] = g[:, 1:] - g[:, :-1]
    gy[:-1, :] = g[1:, :] - g[:-1, :]

    # Bilateral filter the gradients (Eq. 2)
    # OpenCV bilateralFilter expects sigmaColor in same units as pixels; gx/gy are small -> sigma_r=0.04 works well.
    gx_b = cv2.bilateralFilter(gx, d=-1, sigmaColor=sigma_r, sigmaSpace=sigma_s)
    gy_b = cv2.bilateralFilter(gy, d=-1, sigmaColor=sigma_r, sigmaSpace=sigma_s)

    # Build sparse matrix A = I + λ (Dx^T Dx + Dy^T Dy)
    # 5-point Laplacian with Neumann-like handling at boundaries via missing neighbors.
    main = np.ones(n, dtype=np.float64)
    east = np.zeros(n, dtype=np.float64)
    west = np.zeros(n, dtype=np.float64)
    south = np.zeros(n, dtype=np.float64)
    north = np.zeros(n, dtype=np.float64)

    # Helper to map (i,j) -> idx
    idx = np.arange(n).reshape(h, w)

    # Horizontal neighbors
    has_e = np.ones((h, w), dtype=bool)
    has_e[:, -1] = False
    has_w = np.ones((h, w), dtype=bool)
    has_w[:, 0] = False

    # Vertical neighbors
    has_s = np.ones((h, w), dtype=bool)
    has_s[-1, :] = False
    has_n = np.ones((h, w), dtype=bool)
    has_n[0, :] = False

    # For each valid neighbor, add -λ off-diagonal and +λ to diagonal
    # East
    east_idx = idx[has_e]
    east[east_idx] = -lam
    main[east_idx] += lam
    # West (offset -1 handled by placing west on idx where neighbor exists)
    west_idx = idx[has_w]
    west[west_idx] = -lam
    main[west_idx] += lam

    # South (offset +w)
    south_idx = idx[has_s]
    south[south_idx] = -lam
    main[south_idx] += lam
    # North (offset -w)
    north_idx = idx[has_n]
    north[north_idx] = -lam
    main[north_idx] += lam

    # Right-hand side: b = g + λ (Dx^T v_x + Dy^T v_y)
    # where v_x = gx_b, v_y = gy_b
    # Dx^T v_x at (i,j) = -v_x(i,j) + v_x(i,j-1)
    # Dy^T v_y at (i,j) = -v_y(i,j) + v_y(i-1,j)
    div = np.zeros_like(g, dtype=np.float32)
    # x part
    div[:, :-1] -= gx_b[:, :-1]
    div[:, 1:]  += gx_b[:, :-1]
    # y part
    div[:-1, :] -= gy_b[:-1, :]
    div[1:,  :] += gy_b[:-1, :]

    b = (g + lam * div).reshape(-1).astype(np.float64)

    A = diags(
        diagonals=[main, east[:-1], west[1:], south[:-w], north[w:]],
        offsets=[0, 1, -1, w, -w],
        shape=(n, n),
        format="csr"
    )

    u = spsolve(A, b).reshape(h, w).astype(np.float32)
    u = np.clip(u, 0.0, 1.0)
    return u


# ---------------------------
# Step 2: 1D Column GDGIF (Eq. 6-11)
# ---------------------------
def gdgif_1d_column_stripe(
    u: np.ndarray,          # smooth guide image in [0,1]
    X: np.ndarray,          # high-frequency input in [0,1] (can be negative, but we'll work in float)
    mu: float = 0.022,
    h_win: int = 16,
    eps: float = 1e-8,
    small_win: int = 3
) -> np.ndarray:
    """
    Implements the 1D Column GDGIF stripe extraction per Eq. (6)-(11).
    Uses column windows of size h_win (paper sets h=16).
    """
    assert h_win % 2 == 0, "Paper uses window size 16; we'll treat it as size h, radius = h//2"
    r = h_win // 2

    # Means/vars over local column window ω_ck (size h_win)
    mean_u, N = mean_cols(u, r)
    mean_X, _ = mean_cols(X, r)
    mean_uX, _ = mean_cols(u * X, r)
    var_u = (mean_cols(u * u, r)[0] - mean_u * mean_u)

    cov_uX = mean_uX - mean_u * mean_X

    # Edge indicator γ_ck (Eq. 9) needs χ(ck)
    # Paper defines χ(ck)=σ_u,1(ck) σ_u,16(ck). We'll approximate σ_u,1 with a tiny window std (size small_win)
    rs = max(1, small_win // 2)
    mean_u_s, _ = mean_cols(u, rs)
    var_u_s = (mean_cols(u * u, rs)[0] - mean_u_s * mean_u_s)

    sigma16 = np.sqrt(np.maximum(var_u, 0.0))
    sigmaS  = np.sqrt(np.maximum(var_u_s, 0.0))
    chi = sigma16 * sigmaS  # χ(ck)

    chi_mean = np.mean(chi)
    chi_min = np.min(chi)

    # η = 4 / (ξχ,∞ - min(χ(i)))
    denom = max(chi_mean - chi_min, 1e-6)
    eta = 4.0 / denom

    # γ_ck = 1 - 1/(1 + exp(η(χ - ξχ,∞)))
    gamma = 1.0 - 1.0 / (1.0 + np.exp(eta * (chi - chi_mean)))

    # Edge perception weight Γ_u (Eq. 10):
    # Γ_u(ck) = (χ(ck)+ε) * mean( 1/(χ(i)+ε) )
    inv_mean = np.mean(1.0 / (chi + eps))
    Gamma = (chi + eps) * inv_mean

    # Eq. (11):
    # a_ck = ( cov(u,X) + μ Γ_u γ_ck ) / ( var(u) + μ Γ_u )
    # b_ck = mean(X) - a_ck mean(u)
    a = (cov_uX + mu * Gamma * gamma) / (var_u + mu * Gamma + eps)
    b = mean_X - a * mean_u

    # Average overlapping windows (guided filter style)
    mean_a, _ = mean_cols(a, r)
    mean_b, _ = mean_cols(b, r)

    # Stripe estimate S = ā * u + b̄  (Eq. 6 with window-averaged coefficients)
    S = mean_a * u + mean_b
    return S

def project_to_column_bias(S):
    # one value per column
    col = np.median(S, axis=0)              # robust
    col = col - np.mean(col)                # keep global brightness
    return np.tile(col[None, :], (S.shape[0], 1)).astype(np.float32)


# ---------------------------
# Full pipeline
# ---------------------------
def destripe_shao2021(
    img_u8: np.ndarray,
    lam: float = 1024.0,
    sigma_s: float = 2.0,
    sigma_r: float = 0.04,
    mu: float = 0.022,
    h_win: int = 16
):
    """
    Full method:
      u = BLF-LS smooth(g)
      X = g - u
      S = 1D Column GDGIF(u, X)
      clean = g - S
    """
    u = blf_ls_smooth(img_u8, lam=lam, sigma_s=sigma_s, sigma_r=sigma_r)  # [0,1]
    g = img_u8.astype(np.float32) / 255.0
    X = g - u  # high-frequency (stripe + edges + texture)

    S = gdgif_1d_column_stripe(u=u, X=X, mu=mu, h_win=h_win)  # stripe estimate (mostly column artifacts)
    S = cv2.GaussianBlur(S, (1, 31), 0)
    S = project_to_column_bias(S)

    clean = np.clip(g - S, 0, 1)

    return (
        (clean * 255).astype(np.uint8),
        (S * 255).astype(np.int16),         # stripe can be signed-ish; keep int16 for inspection
        (u * 255).astype(np.uint8),
        (X * 255).astype(np.int16)          # high-freq can be signed-ish
    )


# ---------------------------
# Demo on your image
# ---------------------------
if __name__ == "__main__":
    path = r"C:\Users\Johnny\Desktop\deer_detector\image.png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    clean, stripe, smooth_u, high = destripe_shao2021(
        img,
        lam=1024.0,
        sigma_s=2.0,
        sigma_r=0.04,
        mu=0.001,     # try 0.012, 0.022, 0.032 as paper suggests
        h_win=32
    )

    # Visualize
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax = ax.ravel()

    ax[0].imshow(img, cmap="gray");       ax[0].set_title("Input g"); ax[0].axis("off")
    ax[1].imshow(smooth_u, cmap="gray");  ax[1].set_title("BLF-LS smooth u"); ax[1].axis("off")

    # For signed images: show with centered colormap scaling
    high_disp = np.clip(high + 128, 0, 255).astype(np.uint8)
    stripe_disp = np.clip(stripe + 128, 0, 255).astype(np.uint8)

    ax[2].imshow(high_disp, cmap="gray");   ax[2].set_title("High-freq X (shifted)"); ax[2].axis("off")
    ax[3].imshow(stripe_disp, cmap="gray"); ax[3].set_title("Estimated stripe S (shifted)"); ax[3].axis("off")
    ax[4].imshow(clean, cmap="gray");       ax[4].set_title("Denoised = g - S"); ax[4].axis("off")

    diff = cv2.absdiff(img, clean)
    ax[5].imshow(diff, cmap="gray");        ax[5].set_title("|Input - Clean|"); ax[5].axis("off")

    plt.tight_layout()
    plt.show()
