import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_coherence_map(F_model, I0, I1, I2, I3):
    """
    ECG coherence map M_c:
      - interpolate between (I0,I2) and (I1,I3)
      - diff = |I_hat1 - I_hat2|, average channels
      - normalize to [0,1] per-sample
      - return 1 - diff_norm  (1=incoherent, 0=coherent)
    """
    I_hat1 = F_model(I0, I2)
    I_hat2 = F_model(I1, I3)

    diff = (I_hat1 - I_hat2).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]

    diff_min = diff.amin(dim=(2, 3), keepdim=True)
    diff_max = diff.amax(dim=(2, 3), keepdim=True)
    norm = (diff - diff_min) / (diff_max - diff_min + 1e-6)
    M_c = 1.0 - norm
    return M_c


def compute_error_mask(I_c, I_gt, t: float = 0.3):
    """
    Pixel-wise error mask M_e:
      M_e(x) = 1 if |I_c - I_gt|_1 > t else 0
    threshold t can be changed via args.
    """
    diff = (I_c - I_gt).abs().mean(dim=1, keepdim=True)
    M_e = (diff >= t).float()
    return M_e


def blend_with_dmap(I1, I_c, D):
    """
    I_out = D * I1 + (1 - D) * I_c
    D: [B,1,H,W] in [0,1]
    """
    return D * I1 + (1.0 - D) * I_c


def pixelwise_discontinuity_loss(I, I_gt, M_e):
    """
    Weighted L1 loss using error mask M_e.
    """
    diff = (I - I_gt).abs()
    loss = (M_e * diff.mean(dim=1, keepdim=True)).mean()
    return loss
