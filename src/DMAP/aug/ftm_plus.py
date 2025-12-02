import random
from typing import Tuple

from PIL import Image, ImageDraw
from PIL import ImageChops

def _draw_random_shape(draw, w: int, h: int):
    """Draw one random shape, return its mask as a PIL Image (L)."""
    mask = Image.new("L", (w, h), 0)
    mdraw = ImageDraw.Draw(mask)

    shape_type = random.choice(["rect", "ellipse", "polygon", "line"])
    thickness = random.randint(1, 6)
    alpha = random.randint(100, 255)  # 0-255, but we use for overlay

    x1 = random.randint(0, w - 1)
    y1 = random.randint(0, h - 1)
    x2 = random.randint(x1, min(w, x1 + random.randint(10, w // 2)))
    y2 = random.randint(y1, min(h, y1 + random.randint(10, h // 2)))

    fill = alpha  # mask uses intensity as "probability" of being discontinuous

    if shape_type == "rect":
        if random.random() < 0.5:
            mdraw.rectangle([x1, y1, x2, y2], outline=fill, width=thickness)
        else:
            mdraw.rectangle([x1, y1, x2, y2], fill=fill)
    elif shape_type == "ellipse":
        if random.random() < 0.5:
            mdraw.ellipse([x1, y1, x2, y2], outline=fill, width=thickness)
        else:
            mdraw.ellipse([x1, y1, x2, y2], fill=fill)
    elif shape_type == "polygon":
        pts = []
        for _ in range(random.randint(3, 6)):
            px = random.randint(0, w - 1)
            py = random.randint(0, h - 1)
            pts.append((px, py))
        if random.random() < 0.5:
            mdraw.polygon(pts, fill=fill)
        else:
            mdraw.line(pts + [pts[0]], fill=fill, width=thickness)
    else:  # line / arc style
        x3 = random.randint(0, w - 1)
        y3 = random.randint(0, h - 1)
        mdraw.line([x1, y1, x3, y3], fill=fill, width=thickness)

    return mask


def _apply_overlay(base: Image.Image, mask: Image.Image):
    """Blend a random color into base wherever mask>0, using mask as alpha."""
    w, h = base.size
    color = tuple(random.randint(0, 255) for _ in range(3))

    overlay = Image.new("RGB", (w, h), color)
    # Treat mask as alpha
    base = base.convert("RGBA")
    overlay = overlay.convert("RGBA")
    mask_rgba = Image.new("L", (w, h))
    mask_rgba.paste(mask)

    out = Image.composite(overlay, base, mask_rgba)
    return out.convert("RGB")


def apply_ftm_plus(I0, I1, I_gt, I2, I3):
    """
    Apply FTM+-style augmentation to a random subset of frames.

    Returns augmented (I0,I1,I_gt,I2,I3,D_gt_mask).
    D_gt_mask is a PIL 'L' image where >0 means discontinuous/overlaid.
    """

    # probability to augment this sample
    if random.random() < 0.5:
        # no augmentation, D_gt=all zeros
        w, h = I_gt.size
        D_gt = Image.new("L", (w, h), 0)
        return I0, I1, I_gt, I2, I3, D_gt

    frames = [I0, I1, I_gt, I2, I3]
    w, h = I_gt.size

    # Decide number of shapes and which frames they appear in
    num_shapes = random.randint(1, 5)
    D_gt = Image.new("L", (w, h), 0)
    Ddraw = ImageDraw.Draw(D_gt)

    # For each shape, pick a time pattern
    for _ in range(num_shapes):
        mask = _draw_random_shape(ImageDraw.Draw(Image.new("L", (w, h), 0)), w, h)

    # Actually we want multiple shapes in one mask:
    D_local = Image.new("L", (w, h), 0)
    D_local_draw = ImageDraw.Draw(D_local)
    for _ in range(num_shapes):
        shape_mask = _draw_random_shape(D_local_draw, w, h)
        D_local = ImageChops.lighter(D_local, shape_mask)

    # decide which frames get this overlay: appear, disappear, etc.
    pattern = random.choice([
        [0, 1, 1, 1, 0],  # appear and disappear
        [0, 0, 1, 0, 0],  # only GT frame
        [1, 1, 1, 1, 1],  # static overlay
        [0, 1, 1, 0, 0],  # only around GT
    ])

    aug_frames = []
    for i, f in enumerate(frames):
        if pattern[i]:
            aug_frames.append(_apply_overlay(f, D_local))
        else:
            aug_frames.append(f)

    # Accumulate D_gt: any pixel that was overlaid is discontinuous
    D_gt = D_local  # can later be combined if multiple groups

    return (*aug_frames, D_gt)
