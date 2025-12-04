import os
import argparse
import time

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from datasets.vimeo_septuplet import VimeoSeptupletDataset
from models.dmap_estimator import DMapEstimator
from core.frozen_rife import FrozenRIFEWrapper
from core.dmap_utils import (
    compute_coherence_map,
    compute_error_mask,
    blend_with_dmap,
    pixelwise_discontinuity_loss,
)


def parse_args():
    parser = argparse.ArgumentParser("Train D-map estimator on Vimeo-90K")

    parser.add_argument(
        "--vimeo-root",
        type=str,
        required=True,
        help="Path to vimeo-90k root (contains sequence/ and sep_trainlist.txt)",
    )
    parser.add_argument(
        "--rife_model_dir",
        type=str,
        required=True,
        help=(
            "Path to RIFE model directory (modelDir used in original RIFE scripts, "
            "must contain flownet.pkl for the chosen variant)"
        ),
    )

    parser.add_argument(
        "--crop-size",
        type=int,
        default=256,
        help="Square crop size (e.g., 128/160/192/256)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument(
        "--error-threshold",
        type=float,
        default=0.3,
        help="Threshold t for error mask M_e (see Eq. (4) in paper)",
    )
    parser.add_argument("--lambda-e", type=float, default=1.0)
    parser.add_argument("--lambda-d", type=float, default=5.0)

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit number of training samples for subset training",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=500,
        help="Limit number of validation samples",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of DataLoader workers (0 for single-process)",
    )
    parser.add_argument("--save-dir", type=str, default="./checkpoints")

    parser.add_argument(
        "--attention-mode",
        type=str,
        default="down",
        choices=["none", "down", "full"],
        help="Cross-attention mode: 'none' = disabled, 'down' = downsampled MHA, 'full' = CrossMHA"
    )

    return parser.parse_args()


def make_dataloaders(args):
    train_list = os.path.join(args.vimeo_root, "sep_trainlist.txt")
    test_list = os.path.join(args.vimeo_root, "sep_testlist.txt")

    # Train: with FTM+ (as in the paper)
    train_ds = VimeoSeptupletDataset(
        vimeo_root=args.vimeo_root,
        list_path=train_list,
        split="train",
        crop_size=args.crop_size,
        use_ftm_plus=True,
        max_samples=args.max_train_samples,
    )

    # Val: no FTM+ (just real frames)
    val_ds = VimeoSeptupletDataset(
        vimeo_root=args.vimeo_root,
        list_path=test_list,
        split="val",
        crop_size=args.crop_size,
        use_ftm_plus=False,
        max_samples=args.max_val_samples,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def charbonnier_loss(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    Charbonnier loss Φ(x) = sqrt(x^2 + eps^2), averaged over all elements.
    Used as a smooth L1 for LVFI (and optionally Le if you want).
    """
    return torch.mean(torch.sqrt(x * x + eps * eps))


def validate(model_E, model_F, val_loader, error_t: float):
    """
    Validation: just L1 between blended output and GT, to track quality.
    """
    device = model_F.device
    model_E.eval()

    total_l1 = 0.0
    n = 0

    with torch.no_grad():
        for batch in val_loader:
            I0 = batch["I0"].to(device)
            I1 = batch["I1"].to(device)
            I2 = batch["I2"].to(device)
            I3 = batch["I3"].to(device)
            I_gt = batch["I_gt"].to(device)

            # Baseline interpolation and ECG
            I_c = model_F(I1, I2)
            M_c = compute_coherence_map(model_F, I0, I1, I2, I3)

            # D-map prediction
            logits_D = model_E(I0, I1, I2, I3, I_c, M_c)
            D = torch.sigmoid(logits_D)

            # Blend discontinuous (I1) and continuous (Ic) pixels
            I_out = blend_with_dmap(I1, I_c, D)

            # plain L1 for reporting
            l1 = (I_out - I_gt).abs().mean()
            b = I_gt.shape[0]

            total_l1 += l1.item() * b
            n += b

    model_E.train()
    return total_l1 / max(n, 1)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Data
    train_loader, val_loader = make_dataloaders(args)

    # RIFE backbone (frozen), chooses its own device internally
    model_F = FrozenRIFEWrapper(args.rife_model_dir)
    device = model_F.device
    print(f"[train_dmap] Using device: {device}")

    # D-map estimator on same device
    model_E = DMapEstimator(
        base_channels=32,
        num_res_blocks=4,
        attention_mode=args.attention_mode
    ).to(device)

    # Optimizer & scheduler (Adamax + StepLR, as in paper)
    optimizer = optim.Adamax(model_E.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # D-map supervision loss: BCE(D, D_gt)
    bce = torch.nn.BCEWithLogitsLoss()

    best_val = float("inf")
    total_epochs = args.epochs
    global_start = time.time()

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()

        model_E.train()
        running_loss = 0.0
        n_samples = 0

        for step, batch in enumerate(train_loader):
            if step % 50 == 0:
                print(f"[Epoch {epoch}] step {step}/{len(train_loader)}")

            I0 = batch["I0"].to(device)
            I1 = batch["I1"].to(device)
            I2 = batch["I2"].to(device)
            I3 = batch["I3"].to(device)
            I_gt = batch["I_gt"].to(device)
            D_gt = batch["D_gt"].to(device)

            # 1) frozen RIFE interpolation + ECG (no gradients)
            with torch.no_grad():
                I_c = model_F(I1, I2)
                M_c = compute_coherence_map(model_F, I0, I1, I2, I3)

            # 2) D-map prediction
            logits_D = model_E(I0, I1, I2, I3, I_c, M_c)
            D = torch.sigmoid(logits_D)

            # 3) Blending (Eq. (1) in paper)
            I_out = blend_with_dmap(I1, I_c, D)

            # 4) Losses
            # LVFI: Charbonnier between blended output and GT
            diff = I_out - I_gt
            L_vfi = charbonnier_loss(diff)

            # Pixel-wise discontinuity loss Le:
            # uses error mask Me = |Ic - IGT| >= t (see Eq. (4),(5))
            M_e = compute_error_mask(I_c, I_gt, t=args.error_threshold)
            L_e = pixelwise_discontinuity_loss(I_out, I_gt, M_e)

            # D-map supervision loss (FTM+ mask)
            L_D = bce(logits_D, D_gt)

            # total loss (Eq. (6))
            loss = L_vfi + args.lambda_e * L_e + args.lambda_d * L_D

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bsz = I_gt.shape[0]
            running_loss += loss.item() * bsz
            n_samples += bsz

        scheduler.step()

        # Metrics
        train_loss = running_loss / max(n_samples, 1)
        val_l1 = validate(model_E, model_F, val_loader, args.error_threshold)

        # Timing / ETA
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - global_start
        remaining_epochs = total_epochs - epoch
        est_remaining = epoch_time * remaining_epochs

        print(
            f"Epoch [{epoch}/{total_epochs}] "
            f"TrainLoss={train_loss:.4f} "
            f"ValL1={val_l1:.4f} "
            f"LR={scheduler.get_last_lr()[0]:.6f} "
            f"| epoch={epoch_time/60:.2f} min, "
            f"elapsed={elapsed/60:.2f} min, "
            f"ETA~{est_remaining/60:.2f} min"
        )

        # Save best model by val L1 (lower is better)
        if val_l1 < best_val:
            best_val = val_l1
            ckpt_path = os.path.join(args.save_dir, "dmap_estimator_best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model_E.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )

    print("Training finished.")


if __name__ == "__main__":
    main()
