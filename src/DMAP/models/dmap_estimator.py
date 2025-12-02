import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAMBlock(nn.Module):
    """
    Simple CBAM: channel + spatial attention.
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.conv_spatial = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel attention
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        ca = self.mlp(avg_pool) + self.mlp(max_pool)
        ca = torch.sigmoid(ca).view(b, c, 1, 1)
        x = x * ca

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = torch.sigmoid(self.conv_spatial(sa))
        x = x * sa

        return x


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + identity
        out = self.relu(out)
        return out


class DMapEstimator(nn.Module):
    """
    Input: I0,I1,I2,I3,I_c,M_c  (concatenated along channel)
      - I*: [B,3,H,W]
      - M_c: [B,1,H,W]

    Output: logits for D-map [B,1,H,W]
    """
    def __init__(self, base_channels=32, num_blocks=4):
        super().__init__()
        in_ch = 5 * 3 + 1  # 5 RGB frames + 1 coherence mask = 16

        self.conv_in = nn.Conv2d(in_ch, base_channels, 3, padding=1)
        self.conv_down1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.conv_down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)

        self.cbam1 = CBAMBlock(base_channels * 4)
        self.blocks = nn.Sequential(
            *[ResBlock(base_channels * 4) for _ in range(num_blocks)]
        )

        self.conv_up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.conv_up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.conv_out = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, I0, I1, I2, I3, I_c, M_c):
        """
        I0..I3,I_c: [B,3,H,W]
        M_c: [B,1,H,W]
        """
        x = torch.cat([I0, I1, I2, I3, I_c, M_c], dim=1)  # [B,16,H,W]

        x = self.conv_in(x)
        x1 = F.relu(x, inplace=True)
        x2 = F.relu(self.conv_down1(x1), inplace=True)
        x3 = F.relu(self.conv_down2(x2), inplace=True)

        x3 = self.cbam1(x3)
        x3 = self.blocks(x3)

        x = F.relu(self.conv_up1(x3), inplace=True)
        x = F.relu(self.conv_up2(x), inplace=True)

        # skip from early features
        x = x + x1

        logits = self.conv_out(x)  # [B,1,H,W]
        return logits
