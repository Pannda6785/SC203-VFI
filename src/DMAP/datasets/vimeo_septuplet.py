import os
import random
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


from aug.ftm_plus import apply_ftm_plus


def _load_image(path: str):
    img = Image.open(path).convert("RGB")
    return img


def _pil_to_tensor(img: Image.Image):
    # [H,W,3] uint8 -> [3,H,W] float32 in [0,1]
    arr = np.array(img, dtype=np.uint8)        # H, W, 3
    t = torch.from_numpy(arr).permute(2, 0, 1) # 3, H, W
    return t.float() / 255.0

class VimeoSeptupletDataset(Dataset):
    """
    Vimeo-90K Septuplet dataset.

    Uses 7 frames per clip (im1..im7.png) and picks:
      I0 = im2, I1 = im3, I_gt = im4, I2 = im5, I3 = im6

    Supports:
      - arbitrary crop_size (int or (h,w))
      - optional FTM+ augmentation
      - limiting to first max_samples for subset training
    """

    def __init__(
        self,
        vimeo_root: str,
        list_path: str,
        split: str = "train",
        crop_size: int | Tuple[int, int] = 256,
        use_ftm_plus: bool = True,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.vimeo_root = vimeo_root
        self.sequence_root = os.path.join(vimeo_root, "sequence")
        self.split = split
        self.use_ftm_plus = use_ftm_plus and (split == "train")

        if isinstance(crop_size, int):
            self.crop_h = self.crop_w = crop_size
        else:
            self.crop_h, self.crop_w = crop_size

        with open(list_path, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        if max_samples is not None:
            lines = lines[:max_samples]

        self.samples = lines

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, rel: str):
        """
        rel is like '00081/0001'
        """
        seq_dir = os.path.join(self.sequence_root, rel)
        imgs = [
            _load_image(os.path.join(seq_dir, f"im{i}.png"))
            for i in range(1, 8)
        ]
        # I0=im2, I1=im3, I_gt=im4, I2=im5, I3=im6
        I0, I1, I_gt, I2, I3 = imgs[1], imgs[2], imgs[3], imgs[4], imgs[5]
        return I0, I1, I_gt, I2, I3

    def _random_crop_params(self, w: int, h: int):
        if w == self.crop_w and h == self.crop_h:
            return 0, 0
        if w < self.crop_w or h < self.crop_h:
            # if crop is larger than image, just center-crop as much as possible
            x = max((w - self.crop_w) // 2, 0)
            y = max((h - self.crop_h) // 2, 0)
            return x, y
        x = random.randint(0, w - self.crop_w)
        y = random.randint(0, h - self.crop_h)
        return x, y

    def _apply_shared_crop(self, imgs):
        """
        imgs: list of PIL Images with same size
        """
        w, h = imgs[0].size
        x, y = self._random_crop_params(w, h)
        cropped = [im.crop((x, y, x + self.crop_w, y + self.crop_h)) for im in imgs]
        return cropped

    def _apply_random_flips(self, imgs):
        # imgs: list of PIL Images
        # horizontal flip
        if random.random() < 0.5:
            imgs = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in imgs]
        # vertical flip
        if random.random() < 0.5:
            imgs = [im.transpose(Image.FLIP_TOP_BOTTOM) for im in imgs]
        # temporal reverse (only for training)
        if self.split == "train" and random.random() < 0.5:
            I0, I1, I_gt, I2, I3 = imgs
            imgs = [I3, I2, I_gt, I1, I0]
        return imgs

    def __getitem__(self, idx: int):
        rel = self.samples[idx]  # e.g. '00081/0202'
        I0, I1, I_gt, I2, I3 = self._load_frames(rel)

        # shared random crop
        I0, I1, I_gt, I2, I3 = self._apply_shared_crop([I0, I1, I_gt, I2, I3])

        # shared random flips (train only)
        if self.split == "train":
            I0, I1, I_gt, I2, I3 = self._apply_random_flips([I0, I1, I_gt, I2, I3])

        # FTM+ augmentation (train)
        D_gt = None
        if self.use_ftm_plus:
            I0, I1, I_gt, I2, I3, D_gt = apply_ftm_plus(I0, I1, I_gt, I2, I3)

        # to tensors
        I0 = _pil_to_tensor(I0)
        I1 = _pil_to_tensor(I1)
        I_gt = _pil_to_tensor(I_gt)
        I2 = _pil_to_tensor(I2)
        I3 = _pil_to_tensor(I3)

        if D_gt is None:
            D_gt = torch.zeros(1, I0.shape[1], I0.shape[2])
        else:
            # PIL mask in [0,255] -> [1,H,W] float32 in [0,1]
            mask = np.array(D_gt, dtype=np.uint8)   # H, W
            D_gt = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        return {
            "I0": I0,
            "I1": I1,
            "I2": I2,
            "I3": I3,
            "I_gt": I_gt,
            "D_gt": D_gt,
            "relpath": rel,
        }
