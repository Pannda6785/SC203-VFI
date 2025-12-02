import random
from typing import Tuple

from PIL import Image, ImageDraw, ImageChops
import numpy as np

# -------------------------
# Shape / text mask helpers
# -------------------------

def _draw_random_shape_mask(w: int, h: int) -> Image.Image:
    """
    Draw one random "figure" shape (rect, ellipse, polygon, line, arc)
    into an 'L' mask in [0,255].
    """
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    shape_type = random.choice(["rect", "ellipse", "polygon", "line", "arc"])

    # random bounding box
    x1 = random.randint(0, w - 1)
    y1 = random.randint(0, h - 1)
    x2 = random.randint(x1 + 1, min(w, x1 + random.randint(10, max(10, w // 2))))
    y2 = random.randint(y1 + 1, min(h, y1 + random.randint(10, max(10, h // 2))))

    alpha = random.randint(100, 255)
    thickness = random.randint(1, 6)

    if shape_type == "rect":
        if random.random() < 0.5:
            draw.rectangle([x1, y1, x2, y2], fill=alpha)
        else:
            draw.rectangle([x1, y1, x2, y2], outline=alpha, width=thickness)

    elif shape_type == "ellipse":
        if random.random() < 0.5:
            draw.ellipse([x1, y1, x2, y2], fill=alpha)
        else:
            draw.ellipse([x1, y1, x2, y2], outline=alpha, width=thickness)

    elif shape_type == "polygon":
        pts = [
            (random.randint(0, w - 1), random.randint(0, h - 1))
            for _ in range(random.randint(3, 7))
        ]
        if random.random() < 0.5:
            draw.polygon(pts, fill=alpha)
        else:
            draw.line(pts + [pts[0]], fill=alpha, width=thickness)

    elif shape_type == "line":
        x3 = random.randint(0, w - 1)
        y3 = random.randint(0, h - 1)
        draw.line([x1, y1, x3, y3], fill=alpha, width=thickness)

    else:  # "arc"-like
        bbox = [x1, y1, x2, y2]
        start = random.randint(0, 360)
        end = start + random.randint(30, 270)
        draw.arc(bbox, start=start, end=end, fill=alpha, width=thickness)

    return mask


def _draw_random_text_mask(w: int, h: int) -> Image.Image:
    """
    Draw a random "text-like" blob in an 'L' mask.
    This approximates watermark / chat overlay regions.
    """
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = "".join(random.choice(alphabet) for _ in range(random.randint(3, 10)))

    # Rough text box; we don't care about exact font metrics, only region.
    text_w = random.randint(w // 8, w // 2)
    text_h = random.randint(h // 12, h // 6)
    x = random.randint(0, max(0, w - text_w))
    y = random.randint(0, max(0, h - text_h))

    alpha = random.randint(120, 255)
    draw.rectangle([x, y, x + text_w, y + text_h], fill=0)
    draw.text((x, y), text, fill=alpha)

    return mask


def _build_ftm_mask(w: int, h: int, mode: str = "figure") -> Image.Image:
    """
    Build a combined mask by layering several random shapes or text blocks.
    This corresponds to FTM+ (polygons, ellipses, irregular lines, etc.).
    """
    num_layers = random.randint(1, 4)
    total = Image.new("L", (w, h), 0)
    for _ in range(num_layers):
        if mode == "text":
            m = _draw_random_text_mask(w, h)
        else:
            m = _draw_random_shape_mask(w, h)
        total = ImageChops.lighter(total, m)
    return total


def _apply_mask_to_frame(
    frame: Image.Image,
    mask: Image.Image,
    alpha_factor: float | None = None,
    color: Tuple[int, int, int] | None = None,
) -> Image.Image:
    """
    Blend a random color into `frame` using `mask` as alpha.

    alpha_factor in (0,1] -> partial transparency
    mask: 'L' image in [0,255]
    """
    if alpha_factor is None:
        alpha_factor = random.uniform(0.3, 1.0)  # encourage intermediate D_map
    if color is None:
        color = tuple(random.randint(0, 255) for _ in range(3))

    w, h = frame.size
    frame_rgba = frame.convert("RGBA")
    overlay = Image.new("RGBA", (w, h), color + (255,))

    # scale mask intensities by alpha_factor
    mask_scaled = mask.point(lambda v: int(v * alpha_factor))
    out = Image.composite(overlay, frame_rgba, mask_scaled)
    return out.convert("RGB")


# -------------------------
# Main FTM+ augmentation
# -------------------------

def apply_ftm_plus(I0, I1, I_gt, I2, I3, p_augment: float = 0.75):
    """
    Apply FTM+ style augmentation.

    - Randomly chooses between:
        * Figure Mixing (FM)       -> static figures on all frames
        * Text Mixing (TM)         -> static / appear / disappear / moving text
        * Scene-change-like (FTM+) -> full-frame overlays for discontinuities
    - Uses polygons, ellipses, lines, arcs, etc.
    - Uses alpha in (0,1] so the D_map has intermediate values.

    Returns:
        I0', I1', I_gt', I2', I3', D_gt

    where:
        - frames are PIL RGB images (possibly augmented)
        - D_gt is PIL 'L' image in [0,255], proportional to
          the per-pixel difference between ORIGINAL I_gt and AUGMENTED I_gt.
    """

    # With probability (1 - p_augment): no FTM/FTM+, only flips/crops.
    if random.random() > p_augment:
        w, h = I_gt.size
        D_gt = Image.new("L", (w, h), 0)
        return I0, I1, I_gt, I2, I3, D_gt

    frames = [I0, I1, I_gt, I2, I3]
    w, h = I_gt.size

    # 1) Choose augmentation family: figure, text, or scene-change-like
    aug_family = random.choice(["figure", "text", "scene"])

    # 2) Choose temporal pattern (TM patterns follow paper's four cases)
    if aug_family == "figure":
        # Figure Mixing: static figure on all frames
        pattern = [1, 1, 1, 1, 1]

    elif aug_family == "scene":
        # Crude scene-change-like patterns: entire later/earlier frames overlaid
        pattern = random.choice([
            [0, 0, 1, 1, 1],  # scene changes around/after GT
            [1, 1, 1, 0, 0],  # scene changes before GT
        ])

    else:  # "text" -> Text Mixing cases 1–4 (+ one extra moving version)
        pattern = random.choice([
            [1, 1, 1, 1, 1],  # 1) static text
            [0, 0, 1, 1, 1],  # 2) appears (not in prev, appears in future)
            [1, 1, 1, 0, 0],  # 3) disappears
            [0, 1, 1, 0, 0],  # 4) only around GT
            [0, 1, 1, 0, 1],  # extra "moving" pattern
        ])

    # 3) Build a base mask for this augmentation type
    if aug_family == "scene":
        # full-frame mask to simulate hard scene changes
        base_mask = Image.new("L", (w, h), random.randint(180, 255))
    else:
        mode = "text" if aug_family == "text" else "figure"
        base_mask = _build_ftm_mask(w, h, mode=mode)

    # 4) Apply to each frame according to the temporal pattern
    aug_frames = []
    for i, f in enumerate(frames):
        if pattern[i]:
            mask = base_mask

            # For figure/text we sometimes translate the mask a bit
            # to mimic moving overlays (FTM+ motion variety).
            if aug_family in ["figure", "text"] and random.random() < 0.3:
                dx = random.randint(-w // 16, w // 16)
                dy = random.randint(-h // 16, h // 16)
                translated = Image.new("L", (w, h), 0)
                translated.paste(base_mask, (dx, dy))
                mask = translated

            aug_frames.append(_apply_mask_to_frame(f, mask))
        else:
            aug_frames.append(f)

    I0_aug, I1_aug, I_gt_aug, I2_aug, I3_aug = aug_frames

    # 5) Build D_gt as the DIFFERENCE MAP between original and augmented GT
    #    (this is what the paper uses for supervision).
    gt_orig = np.array(I_gt, dtype=np.float32)     # H,W,3
    gt_aug = np.array(I_gt_aug, dtype=np.float32)  # H,W,3

    diff = np.abs(gt_aug - gt_orig).mean(axis=2)   # H,W, average over channels

    if diff.max() > 0:
        diff_norm = diff / diff.max()              # [0,1]
    else:
        diff_norm = diff                           # all zeros

    D_gt = Image.fromarray(
        np.clip(diff_norm * 255.0, 0, 255).astype(np.uint8),
        mode="L",
    )

    return I0_aug, I1_aug, I_gt_aug, I2_aug, I3_aug, D_gt
