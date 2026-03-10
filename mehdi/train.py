"""
train.py — Entraînement de l'Attention U-Net avec polygon-level weak supervision.

Optimisations RTX 5070 Ti (Blackwell, 16 GB VRAM) :
  • torch.bfloat16 (AMP) : plus stable que float16, natif sur Blackwell
  • torch.compile(mode="max-autotune") : +30–40% de débit
  • pin_memory + persistent_workers pour maximiser le débit CPU→GPU
  • batch_size=8, pas d'accumulation de gradient (VRAM suffisant)
  • AdamW + CosineAnnealingLR avec warmup linéaire

Lancement :
    cd mehdi
    python train.py                          # paramètres par défaut
    python train.py --epochs 30 --lr 1e-3   # override CLI
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:256")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

# Imports locaux (mêmes dossier)
sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from dataset import AI4ArcticDataset, split_scenes
from losses import total_loss as compute_loss
from model import AttentionUNet


#  Monitoring ressources 

def _fmt(n: int | None) -> str:
    if n is None:
        return "n/a"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TiB"


def log_resources(tag: str, device: torch.device) -> None:
    msg = tag
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated()
        reserv = torch.cuda.memory_reserved()
        free, total = torch.cuda.mem_get_info()
        msg += (f"  VRAM alloc={_fmt(alloc)} / res={_fmt(reserv)} "
                f"/ free={_fmt(free)}/{_fmt(total)}")
    print(f"[GPU] {msg}")


#  Visualisation debug 

def save_debug_fig(
    inputs: torch.Tensor,
    logits: torch.Tensor,
    ct_map: torch.Tensor,
    valid: torch.Tensor,
    epoch: int,
    out_dir: Path,
) -> None:
    """
    Sauvegarde une figure 4 panneaux pour le premier item du batch :
    [SAR HH | CT cible | Prédiction brute | Prédiction (sigmoid)]
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    b = 0
    sar_hh  = inputs[b, 0].cpu().float().numpy()
    sar_hv  = inputs[b, 1].cpu().float().numpy()
    ct      = ct_map[b].cpu().float().numpy()
    pred    = torch.sigmoid(logits[b, 0]).detach().cpu().float().numpy()
    msk     = valid[b].cpu().numpy()

    # Masquer la terre
    ct_disp   = np.where(msk, ct, np.nan)
    pred_disp = np.where(msk, pred, np.nan)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Époque {epoch+1} — debug batch", fontsize=13)

    axes[0].imshow(sar_hh, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("SAR HH (normalisé)")
    axes[0].axis("off")

    axes[1].imshow(sar_hv, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("SAR HV (normalisé)")
    axes[1].axis("off")

    im2 = axes[2].imshow(ct_disp, cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title("Label CT (polygones)")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(pred_disp, cmap="Blues", vmin=0, vmax=1)
    mean_pred = float(np.nanmean(pred_disp)) if msk.any() else 0.0
    axes[3].set_title(f"Prédiction σ(logits)  μ={mean_pred:.2f}")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    plt.tight_layout()
    path = out_dir / f"epoch_{epoch+1:03d}_debug.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


#  Entraînement 

def train(cfg: Config) -> None:
    #  Device 
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU : {gpu_name}")
    else:
        print("Attention : entraînement sur CPU (lent)")

    #  Données 
    data_dir = Path(cfg.data_dir)
    train_files, val_files = split_scenes(data_dir, cfg.val_scene_keyword)

    if not train_files:
        raise FileNotFoundError(f"Aucun fichier .nc trouvé dans {data_dir.resolve()}")

    train_sets = [
        AI4ArcticDataset(f, patch_size=cfg.patch_size, augment=cfg.augment)
        for f in train_files
    ]
    val_sets = [
        AI4ArcticDataset(f, patch_size=cfg.patch_size, augment=False)
        for f in val_files
    ]

    train_loader = DataLoader(
        ConcatDataset(train_sets),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.num_workers > 0 and cfg.persistent_workers),
    )
    val_loader = DataLoader(
        ConcatDataset(val_sets) if val_sets else ConcatDataset(train_sets[:1]),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.num_workers > 0 and cfg.persistent_workers),
    ) if val_files else None

    n_train = sum(len(d) for d in train_sets)
    n_val   = sum(len(d) for d in val_sets) if val_sets else 0
    print(f"\nPatches train : {n_train}  |  val : {n_val}")

    #  Modèle 
    model = AttentionUNet(
        n_channels=cfg.n_channels,
        base_features=cfg.base_features,
        dropout=cfg.dropout,
    ).to(device)
    print(f"Modèle : AttentionUNet  |  paramètres : {model.num_parameters:,}")

    if cfg.compile_model and hasattr(torch, "compile"):
        print("Compilation du modèle (torch.compile)…")
        model = torch.compile(model, mode="max-autotune")

    #  Optimiseur & scheduler 
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Cosine annealing avec warmup linéaire
    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = cfg.warmup_epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    #  AMP 
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    use_amp = cfg.use_amp and (device.type == "cuda")

    if use_amp:
        print(f"AMP activé ({cfg.amp_dtype})")
        # GradScaler utile surtout pour FP16 ; avec BF16 pas strictement nécessaire
        # mais on le garde pour compatibilité
        scaler = torch.amp.GradScaler(device.type, enabled=(amp_dtype == torch.float16))
    else:
        scaler = torch.amp.GradScaler(device.type, enabled=False)

    #  Sorties 
    save_dir = Path(cfg.save_dir)
    debug_dir = save_dir / "debug"
    save_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    log_resources("startup", device)

    #  Boucle d'entraînement 
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "lr": []
    }
    best_val_loss = float("inf")
    best_ckpt_path = save_dir / "best_model.pth"

    for epoch in range(cfg.epochs):
        t0 = time.perf_counter()
        model.train()
        running: dict[str, float] = {"loss_total": 0.0, "loss_mse": 0.0, "loss_entropy": 0.0}
        n_batches = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{cfg.epochs}", leave=False, ncols=110)
        debug_saved = False

        for inputs, ct_map, poly_map, valid in loop:
            inputs   = inputs.to(device, non_blocking=True)
            ct_map   = ct_map.to(device, non_blocking=True)
            poly_map = poly_map.to(device, non_blocking=True)
            valid    = valid.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(inputs)
                loss, metrics = compute_loss(
                    logits, poly_map, ct_map, valid,
                    lambda_entropy=cfg.lambda_entropy,
                )

            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v
            n_batches += 1

            loop.set_postfix(
                loss=f"{metrics['loss_total']:.4f}",
                mse=f"{metrics['loss_mse']:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            # Debug figure pour le premier batch de l'époque
            if not debug_saved and (epoch % cfg.debug_every == 0):
                save_debug_fig(inputs, logits, ct_map, valid, epoch, debug_dir)
                debug_saved = True

        avg_train = {k: v / n_batches for k, v in running.items()}

        #  Validation 
        avg_val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            val_loss_acc = 0.0
            val_batches = 0
            with torch.no_grad():
                for inputs, ct_map, poly_map, valid in val_loader:
                    inputs   = inputs.to(device, non_blocking=True)
                    ct_map   = ct_map.to(device, non_blocking=True)
                    poly_map = poly_map.to(device, non_blocking=True)
                    valid    = valid.to(device, non_blocking=True)
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        logits = model(inputs)
                        loss, _ = compute_loss(logits, poly_map, ct_map, valid, cfg.lambda_entropy)
                    val_loss_acc += float(loss.item())
                    val_batches += 1
            avg_val_loss = val_loss_acc / max(val_batches, 1)

            # Sauvegarde du meilleur checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    {"epoch": epoch + 1, "model_state": model.state_dict(),
                     "val_loss": best_val_loss, "cfg": cfg},
                    best_ckpt_path,
                )

        #  Checkpoint de l'époque 
        if not cfg.save_best_only:
            torch.save(
                {"epoch": epoch + 1, "model_state": model.state_dict(), "cfg": cfg},
                save_dir / f"ckpt_epoch_{epoch+1:03d}.pth",
            )

        history["train_loss"].append(avg_train["loss_total"])
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(scheduler.get_last_lr()[0])

        dt = time.perf_counter() - t0
        val_str = f"val={avg_val_loss:.4f}" if not np.isnan(avg_val_loss) else ""
        print(
            f"  Ép.{epoch+1:3d}/{cfg.epochs}  "
            f"train={avg_train['loss_total']:.4f} "
            f"(mse={avg_train['loss_mse']:.4f} ent={avg_train['loss_entropy']:.4f})  "
            f"{val_str}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  {dt:.1f}s"
        )

        if device.type == "cuda":
            log_resources(f"epoch {epoch+1}", device)

    #  Courbes de loss 
    _save_loss_curves(history, save_dir)
    print(f"\nEntraînement terminé. Sorties dans : {save_dir.resolve()}")
    if val_loader is not None:
        print(f"Meilleur checkpoint (val={best_val_loss:.4f}) : {best_ckpt_path}")


def _save_loss_curves(history: dict[str, list], save_dir: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Courbes d'entraînement — Attention U-Net", fontsize=13)

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train loss", markersize=4)
    val_clean = [v for v in history["val_loss"] if not np.isnan(v)]
    if val_clean:
        axes[0].plot(
            [e for e, v in zip(epochs, history["val_loss"]) if not np.isnan(v)],
            val_clean, "r-o", label="Val loss", markersize=4,
        )
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Loss totale")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Loss (MSE polygone + entropie)")

    axes[1].semilogy(epochs, history["lr"], "g-", linewidth=2)
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Learning rate")
    axes[1].set_title("Scheduler (warmup + cosine annealing)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = save_dir / "loss_curves.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Courbes sauvegardées : {path}")


#  CLI 

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Attention U-Net — AI4Arctic weak supervision")
    p.add_argument("--data-dir",       default=None)
    p.add_argument("--save-dir",       default=None)
    p.add_argument("--epochs",         type=int,   default=None)
    p.add_argument("--lr",             type=float, default=None)
    p.add_argument("--batch-size",     type=int,   default=None)
    p.add_argument("--patch-size",     type=int,   default=None)
    p.add_argument("--base-features",  type=int,   default=None)
    p.add_argument("--num-workers",    type=int,   default=None)
    p.add_argument("--device",         default=None, choices=["auto", "cpu", "cuda"])
    p.add_argument("--no-amp",         action="store_true")
    p.add_argument("--no-compile",     action="store_true")
    p.add_argument("--lambda-entropy", type=float, default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _make_parser().parse_args(argv)
    cfg = Config()
    if args.data_dir:       cfg.data_dir        = args.data_dir
    if args.save_dir:       cfg.save_dir        = args.save_dir
    if args.epochs:         cfg.epochs          = args.epochs
    if args.lr:             cfg.lr              = args.lr
    if args.batch_size:     cfg.batch_size      = args.batch_size
    if args.patch_size:     cfg.patch_size      = args.patch_size
    if args.base_features:  cfg.base_features   = args.base_features
    if args.num_workers:    cfg.num_workers     = args.num_workers
    if args.device:         cfg.device          = args.device
    if args.no_amp:         cfg.use_amp         = False
    if args.no_compile:     cfg.compile_model   = False
    if args.lambda_entropy: cfg.lambda_entropy  = args.lambda_entropy
    train(cfg)


if __name__ == "__main__":
    main()
