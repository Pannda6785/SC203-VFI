import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAMBlock(nn.Module):
    """
    CBAM: channel + spatial attention.
    Kept simple but faithful enough for our use.
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # channel attention MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        # spatial attention conv over [avg,max] maps
        self.conv_spatial = nn.Conv2d(
            2, 1, kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # ----- Channel attention -----
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        ca = self.mlp(avg_pool) + self.mlp(max_pool)
        ca = torch.sigmoid(ca).view(b, c, 1, 1)
        x = x * ca

        # ----- Spatial attention -----
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = torch.sigmoid(self.conv_spatial(sa))
        x = x * sa

        return x


class ResBlock(nn.Module):
    """
    Simple residual block: Conv-ReLU-Conv + skip.
    Used to approximate the ResNet50 chunk in Fig. 4.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + identity
        out = self.relu(out)
        return out


class FrameEncoder(nn.Module):
    """
    Shared Conv2d + MaxPool used on each frame
    (I0, I1, I2, I3, Ic) as in the extraction module.
    """
    def __init__(self, in_ch=3, out_ch=32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x)
        return x


class CrossAttentionBlock(nn.Module):
    """
    Multi-head attention between a frame feature and Ic feature.

    Given x = feature of I1 (or I2) and c = feature of Ic:
      - use x as query, c as key/value
      - attention result is turned into a gate via sigmoid
      - gate is multiplied element-wise with x (Fig. 4: ⊗ after MHA)
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)

    def forward(self, x, c):
        """
        x, c: [B,C,H,W]
        returns: attended/gated x, same shape
        """
        b, c_ch, h, w = x.shape

        # [B, C, H, W] -> [B, HW, C]
        q = x.flatten(2).transpose(1, 2)      # [B, HW, C]
        k = c.flatten(2).transpose(1, 2)      # [B, HW, C]
        v = k

        q = self.norm_q(q)
        k = self.norm_kv(k)
        v = self.norm_kv(v)

        attn_out, _ = self.attn(q, k, v)      # [B, HW, C]
        attn_out = attn_out.transpose(1, 2).reshape(b, c_ch, h, w)

        gate = torch.sigmoid(attn_out)
        return x * gate                       # element-wise multiplication (⊗)


class DMapEstimator(nn.Module):
    """
    Coherence & attention guided D-map Estimator (Fig. 4).

    Inputs:
      I0, I1, I2, I3, I_c : [B,3,H,W]
      M_c                 : [B,1,H,W]  (ECG: 1 = incoherent/discontinuous, 0 = coherent)

    Output:
      logits for D-map    : [B,1,H,W] (BCEWithLogitsLoss outside)
    """
    def __init__(
        self,
        base_channels: int = 32,
        num_res_blocks: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()

        C = base_channels

        # ----- Extraction module -----
        self.encoder = FrameEncoder(in_ch=3, out_ch=C)

        # attention between adjacent frames (I1, I2) and Ic
        self.attn_1 = CrossAttentionBlock(C, num_heads=num_heads)
        self.attn_2 = CrossAttentionBlock(C, num_heads=num_heads)

        # concat [I0, att(I1), att(I2), I3] -> Conv2d
        self.conv_concat_neighbors = nn.Conv2d(4 * C, 2 * C, kernel_size=3, padding=1, bias=True)

        # concat with Ic feature
        self.conv_after_ic = nn.Conv2d(3 * C, 4 * C, kernel_size=3, padding=1, bias=True)

        # ----- Fusion module (Conv2d + CBAM + ResBlocks + global residual) -----
        self.fuse_conv1 = nn.Conv2d(4 * C, 4 * C, kernel_size=3, padding=1, bias=True)
        self.cbam1 = CBAMBlock(4 * C)
        self.cbam2 = CBAMBlock(4 * C)
        self.fuse_conv2 = nn.Conv2d(4 * C, 4 * C, kernel_size=3, padding=1, bias=True)
        self.fuse_conv3 = nn.Conv2d(4 * C, 4 * C, kernel_size=3, padding=1, bias=True)

        self.res_blocks = nn.Sequential(
            *[ResBlock(4 * C) for _ in range(num_res_blocks)]
        )

        # final 1x1-ish conv to get 1-channel logits
        self.conv_out_lowres = nn.Conv2d(4 * C, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, I0, I1, I2, I3, I_c, M_c):
        """
        I0..I3, I_c : [B,3,H,W]
        M_c         : [B,1,H,W] (ECG map, 1 = discontinuous)

        Returns:
          logits_D   : [B,1,H,W]
        """
        # ----- Shared Conv2d + MaxPool on each frame -----
        f0 = self.encoder(I0)   # [B,C,H/2,W/2]
        f1 = self.encoder(I1)
        f2 = self.encoder(I2)
        f3 = self.encoder(I3)
        fc = self.encoder(I_c)

        # ----- Multi-head attention with Ic (adjacent frames only) -----
        f1_att = self.attn_1(f1, fc)  # I1 <-> Ic
        f2_att = self.attn_2(f2, fc)  # I2 <-> Ic

        # concat attended neighbors with I0, I3
        x = torch.cat([f0, f1_att, f2_att, f3], dim=1)   # [B,4C,H/2,W/2]
        x = F.relu(self.conv_concat_neighbors(x), inplace=True)  # [B,2C,H/2,W/2]

        # concat with Ic feature and refine
        x = torch.cat([x, fc], dim=1)                   # [B,3C,H/2,W/2]
        x = F.relu(self.conv_after_ic(x), inplace=True) # [B,4C,H/2,W/2]

        # ----- Apply coherence mask Mc as element-wise multiplication -----
        # Mc is defined on full-res; downsample to feature size.
        if M_c is not None:
            M_c_ds = F.interpolate(
                M_c,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            x = x * M_c_ds  # filter out continuous regions

        # Save for global residual connection (Fig. 4, long skip)
        residual = x

        # ----- Fusion module -----
        x = F.relu(self.fuse_conv1(x), inplace=True)
        x = self.cbam1(x)
        x = self.cbam2(x)
        x = F.relu(self.fuse_conv2(x), inplace=True)
        x = F.relu(self.fuse_conv3(x), inplace=True)
        x = self.res_blocks(x)

        # global residual
        x = x + residual

        # low-resolution logits
        logits_low = self.conv_out_lowres(x)            # [B,1,H/2,W/2]

        # upsample back to image resolution for blending & loss
        H, W = I0.shape[-2:]
        logits = F.interpolate(
            logits_low,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        return logits
