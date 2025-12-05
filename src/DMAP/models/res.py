import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from typing import Optional, Tuple
import math
from torchvision.transforms import Grayscale

def _make_gaussian_kernel(window_size: int, sigma: float, device=None, dtype=None):
    # Create 2D gaussian kernel (window_size x window_size) separable
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    g2d = g1d[:, None] @ g1d[None, :]
    g2d = g2d / g2d.sum()
    return g2d

def structural_similarity_torch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    *,
    win_size: Optional[int] = None,
    data_range: Optional[float] = None,
    gaussian_weights: bool = False,
    sigma: float = 1.5,
    use_sample_covariance: bool = True,
    K1: float = 0.01,
    K2: float = 0.03,
    full: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute SSIM per scikit-image implementation (2D images).

    Parameters
    ----------
    img1, img2: torch.Tensor
        Input image pairs. Allowed shape: (B,C,H,W).
        Channel-first convention is used.
    data_range: float
        Difference between maximum and minimum possible values. Required for
        floating point tensors to match scikit-image behavior.
    gaussian_weights: bool
        If True, use Gaussian kernel of `sigma` (truncate factor = 3.5 like skimage).
    full: bool
        If True, return (mssim, S_map). S_map has shape (B, H, W) (averaged over channels).

    Returns
    -------
    mssim: torch.Tensor
        Mean SSIM for each image in the batch (shape (B,)). If batch size is 1,
        a scalar tensor is returned (0-dim).
    S_map: torch.Tensor or None
        If full=True, returns the SSIM map averaged across channels with shape (B, H, W).
    """
    
    if img1.dim() != 4 or img2.dim() != 4:
        raise ValueError("img1 and img2 must be both 4D")
    
    grayscale_transform = Grayscale(num_output_channels=1)
    img1 = grayscale_transform(img1)
    img2 = grayscale_transform(img2)

    device = img1.device
    dtype = img1.dtype

    if img1.dtype != img2.dtype:
        # follow skimage: prefer im1 dtype
        img2 = img2.to(dtype)

    # Data range checks (scikit forces explicit for floating dtypes)
    # if data_range is None:
    #     if torch.is_floating_point(img1):
    #         raise ValueError(
    #             "Since image dtype is floating point, you must specify the data_range parameter."
    #         )
    #     # integer types: infer range
    #     # match numpy dtype_range behaviour for common integer types
    #     # We'll support uint8, uint16, int16, etc. Use torch.iinfo
    #     info = torch.iinfo(img1.dtype)
    #     data_range = float(info.max - info.min)

    # determine window size
    ndim = 2
    if win_size is None:
        win_size = 3 if not gaussian_weights else int(math.floor(2 * (int(3.5 * sigma + 0.5)) + 1))
    if win_size % 2 == 0:
        raise ValueError("Window size must be odd.")

    B, C, H, W = img1.shape
    if min(H, W) < win_size:
        raise ValueError("win_size exceeds image extent. Increase image size or reduce win_size.")

    # Prepare filter kernel
    if gaussian_weights:
        truncate = 3.5
        # recompute radius & win_size matching skimage logic
        r = int(truncate * sigma + 0.5)
        win_size = 2 * r + 1
        kernel_2d = _make_gaussian_kernel(win_size, sigma, device=device, dtype=dtype)
    else:
        # uniform filter kernel
        kernel_2d = torch.ones((win_size, win_size), device=device, dtype=dtype)
        kernel_2d = kernel_2d / (win_size ** 2)

    # build conv weight of shape (C,1,win,win) and use groups=C to convolve each channel individually
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0)  # 1 x 1 x win x win
    kernel = kernel.expand(C, 1, win_size, win_size).contiguous()

    pad = (win_size - 1) // 2

    # helper conv (preserve dtype)
    def conv_per_channel(x):
        # x: (B, C, H, W) -> out (B, C, H, W)
        return F.conv2d(x, kernel, bias=None, stride=1, padding=pad, groups=C)

    # convert to float_type (use img1 dtype; scikit uses supported_float_type but here assume float dtype is supplied or integers)
    # ensure float
    if not torch.is_floating_point(img1):
        img1 = img1.float()
        img2 = img2.float()
        dtype = img1.dtype

    # compute means
    ux = conv_per_channel(img1)
    uy = conv_per_channel(img2)

    # compute squared means and products
    uxx = conv_per_channel(img1 * img1)
    uyy = conv_per_channel(img2 * img2)
    uxy = conv_per_channel(img1 * img2)

    NP = win_size ** ndim
    cov_norm = NP / (NP - 1.0) if use_sample_covariance else 1.0

    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = float(data_range)
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1 = 2 * ux * uy + C1
    A2 = 2 * vxy + C2
    B1 = ux * ux + uy * uy + C1
    B2 = vx + vy + C2

    D = B1 * B2
    S = (A1 * A2) / D

    # mean SSIM averaged over spatial dims and channels like scikit (they compute per-channel then mean)
    # scikit computes S per-channel and then channel mean -> we average channels on S first
    # S has shape (B, C, H, W). They then average channels -> (B, H, W)
    S_map = S.mean(dim=1)  # (B, H, W)

    return S_map

def smooth_structural_diff(img1, img2):
    """
    Same functionality as the original function,
    but supports torch tensors with shape (B,3,H,W).
    Uses cv2 + skimage's SSIM.
    """
    B, C, H, W = img1.shape

    soft_masks = []
    smoothed_masks = []

    for b in range(B):

        # ---- Convert to numpy (H,W,3) uint8 ----
        i1 = img1[b].detach().cpu().permute(1,2,0).numpy()
        i2 = img2[b].detach().cpu().permute(1,2,0).numpy()

        # ---- Convert to grayscale ----
        gray1 = cv2.cvtColor(i1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(i2, cv2.COLOR_RGB2GRAY)

        # ---- SSIM (full diff map) ----
        score, diff = ssim(gray1, gray2, full=True, data_range=255)
        soft_mask = 1.0 - diff  # difference probability

        # uint8 mask
        soft_mask_uint8 = (soft_mask * 255).astype("uint8")

        # ---- Binary mask with OTSU ----
        _, binary_mask = cv2.threshold(
            soft_mask_uint8, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        # ---- Smooth edges ----
        smoothed_edges_mask = cv2.GaussianBlur(binary_mask, (15, 15), 0)

        # Collect
        soft_masks.append(torch.from_numpy(soft_mask_uint8))
        smoothed_masks.append(torch.from_numpy(smoothed_edges_mask))

    # Stack back into (B,1,H,W)
    soft_masks = torch.stack(soft_masks).unsqueeze(1)
    smoothed_masks = torch.stack(smoothed_masks).unsqueeze(1)

    return soft_masks, smoothed_masks

class DoubleConv(nn.Module):
    """
    (Conv2d -> BN -> ReLU) * 2
    We use standard Convolution here (not depthwise) for better feature mixing.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class AttentionBlock(nn.Module):
    """
    Standard Attention Gate.
    Filters the skip connection (x) using the gating signal (g) from the coarser layer.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DiscontinuityRefinementNet(nn.Module):
    def __init__(self):
        super().__init__()
        in_ch = 10 # 3(I1) + 3(I2) + 3(Ic) + 1(Mask)
        
        # --- Encoder ---
        self.inc = DoubleConv(in_ch, 16)
        
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        
        # --- Bottleneck ---
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # --- Decoder with Attention ---
        
        # Up 1 (256 -> 128)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att1 = AttentionBlock(F_g=256, F_l=128, F_int=64)
        self.conv1 = DoubleConv(256 + 128, 128)
        
        # Up 2 (128 -> 64)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att2 = AttentionBlock(F_g=128, F_l=64, F_int=32)
        self.conv2 = DoubleConv(128 + 64, 64)
        
        # Up 3 (64 -> 32)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att3 = AttentionBlock(F_g=64, F_l=32, F_int=16)
        self.conv3 = DoubleConv(64 + 32, 32)
        
        # Up 4 (32 -> 16)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att4 = AttentionBlock(F_g=32, F_l=16, F_int=8)
        self.conv4 = DoubleConv(32 + 16, 16)

        # --- Output Head ---
        # 16 -> 1 channel
        self.outc = nn.Conv2d(16, 1, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, I1, I2, Ic, coarse_mask):
        # 1. Concat Inputs
        x_input = torch.cat([I1, I2, Ic, coarse_mask], dim=1)
        
        # 2. Encoder
        x1 = self.inc(x_input)      # 16
        x2 = self.down1(x1)         # 32
        x3 = self.down2(x2)         # 64
        x4 = self.down3(x3)         # 128
        x5 = self.down4(x4)         # 256 (Bottleneck)
        
        # 3. Decoder
        
        # Block 1
        u1 = self.up1(x5)
        x4_weighted = self.att1(g=u1, x=x4) # Attention: Filter skip connection
        u1 = torch.cat([u1, x4_weighted], dim=1)
        u1 = self.conv1(u1)
        
        # Block 2
        u2 = self.up2(u1)
        x3_weighted = self.att2(g=u2, x=x3)
        u2 = torch.cat([u2, x3_weighted], dim=1)
        u2 = self.conv2(u2)
        
        # Block 3
        u3 = self.up3(u2)
        x2_weighted = self.att3(g=u3, x=x2)
        u3 = torch.cat([u3, x2_weighted], dim=1)
        u3 = self.conv3(u3)
        
        # Block 4
        u4 = self.up4(u3)
        x1_weighted = self.att4(g=u4, x=x1)
        u4 = torch.cat([u4, x1_weighted], dim=1)
        u4 = self.conv4(u4)
        
        # 4. Residual & Final Calculation
        residual = self.tanh(self.outc(u4))
        
        refined_mask = coarse_mask + residual
        refined_mask = torch.clamp(refined_mask, 0.0, 1.0)
        
        return refined_mask