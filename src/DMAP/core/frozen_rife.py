import os
import sys
from typing import Optional

import torch
import torch.nn as nn

# RIFE-VFI root (adjust if your layout differs)
RIFE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RIFE-VFI"))
if RIFE_ROOT not in sys.path:
    sys.path.append(RIFE_ROOT)


def _get_rife_model_class():
    """
    Mirror your original try/except cascade:

    try:
        from model.RIFE_HDv2 import Model
    except:
        from train_log.RIFE_HDv3 import Model
    except:
        from model.RIFE_HD import Model
    except:
        from model.RIFE import Model
    """
    try:
        from model.RIFE_HDv2 import Model as RIFEModel  # type: ignore
        version = "v2.x HD"
        return RIFEModel, version
    except Exception:
        pass

    try:
        from train_log.RIFE_HDv3 import Model as RIFEModel  # type: ignore
        version = "v3.x HD"
        return RIFEModel, version
    except Exception:
        pass

    try:
        from model.RIFE_HD import Model as RIFEModel  # type: ignore
        version = "v1.x HD"
        return RIFEModel, version
    except Exception:
        pass

    from model.RIFE import Model as RIFEModel  # type: ignore
    version = "ArXiv-RIFE"
    return RIFEModel, version


class FrozenRIFEWrapper(nn.Module):
    """
    Wrapper around whatever RIFE variant your repo can load.

    Args:
        model_dir: directory passed to `load_model(model_dir, -1)`
    """

    def __init__(self, model_dir: Optional[str] = None):
        super().__init__()
        RIFEModel, version = _get_rife_model_class()
        print(f"[FrozenRIFEWrapper] Using RIFE variant: {version}")

        # This is NOT an nn.Module; it owns self.flownet which is.
        self.model = RIFEModel()

        if model_dir is not None:
            self.model.load_model(model_dir, -1)
        else:
            print("[FrozenRIFEWrapper] WARNING: model_dir is None, weights may not be loaded.")

        self.model.eval()

        # Infer the device from underlying flownet
        try:
            self._device = next(self.model.flownet.parameters()).device
        except Exception:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[FrozenRIFEWrapper] Underlying flownet device: {self._device}")

    @property
    def device(self):
        return self._device

    @torch.no_grad()
    def forward(self, I_a: torch.Tensor, I_b: torch.Tensor) -> torch.Tensor:
        """
        I_a, I_b: [B,3,H,W] float in [0,1]
        Returns: merged[2] from RIFE, [B,3,H,W]
        """
        I_a = I_a.to(self._device)
        I_b = I_b.to(self._device)
        out = self.model.inference(I_a, I_b)
        # RIFE already returns tensor on its own device.
        return out
