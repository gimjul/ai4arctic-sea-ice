"""
inference.py — Inférence complète sur une scène SAR + Analytical Logit Scaling.

Pipeline :
    1. Sliding window (256×256, stride=64) → logit map [H, W]
       Chaque pixel est inféré (256/64)² = 16 fois sous des contextes différents.
    2. Gaussian blur sur la logit map (filtre le speckle SAR).
    3. Analytical Logit Scaling (ALS) :
           b = (z_98% + z_2%) / 2
           T = (z_98% - z_2%) / 10
           p = σ( (z - b) / T )
       → Force une binarisation physique sans supervision pixel-wise.
    4. Génération de figures de présentation professionnelles.

Lancement :
    python inference.py --nc ../sujet/Northwest_2019/20190105T*.nc --ckpt outputs/best_model.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import xarray as xr
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from dataset import SIGRID_TO_CT
from model import AttentionUNet


#  Chargement de la scène complète 

def load_scene(nc_path: str | Path, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge une scène complète et retourne :
    - image    : [C, H, W] float32 normalisé
    - ct_map   : [H, W] float32 concentration CT
    - land_mask: [H, W] bool (True = terre)
    """
    ds = xr.open_dataset(str(nc_path), engine="h5netcdf")
    H = int(ds.sizes["sar_lines"])
    W = int(ds.sizes["sar_samples"])

    hh_key = "nersc_sar_primary" if "nersc_sar_primary" in ds else "sar_primary"
    hv_key = "nersc_sar_secondary" if "nersc_sar_secondary" in ds else "sar_secondary"

    print(f"  Chargement scène {Path(nc_path).name[:35]}  ({H}×{W} px)…")

    #  SAR 
    hh = np.abs(ds[hh_key].values).astype(np.float32)
    hv = np.abs(ds[hv_key].values).astype(np.float32)
    land_mask = np.isnan(hh)
    hh = np.nan_to_num(hh, nan=0.0)
    hv = np.nan_to_num(hv, nan=0.0)

    hh_db = 10.0 * np.log10(hh + 1e-6)
    hv_db = 10.0 * np.log10(hv + 1e-6)
    hh_n = (np.clip(hh_db, -30.0, 20.0) + 30.0) / 50.0
    hv_n = (np.clip(hv_db, -30.0, 20.0) + 30.0) / 50.0

    #  Angle d'incidence 
    angle = np.full((H, W), 0.5, dtype=np.float32)
    try:
        if "sar_incidenceangles" in ds:
            raw = ds["sar_incidenceangles"].values.astype(np.float32)
            raw = np.nan_to_num(raw, nan=30.0)
            if raw.ndim == 1:
                angle = np.tile(np.clip(raw / 60.0, 0, 1)[np.newaxis, :], (H, 1))
            elif raw.ndim == 2 and raw.shape[0] == 1:
                angle = np.tile(np.clip(raw[0] / 60.0, 0, 1)[np.newaxis, :], (H, 1))
        elif "sar_grid_incidenceangle" in ds:
            raw = ds["sar_grid_incidenceangle"].values.astype(np.float32)
            raw = np.nan_to_num(raw, nan=30.0)
            side = int(np.sqrt(raw.size))
            if side * side == raw.size:
                grid = raw.reshape(side, side)
                angle = cv2.resize(grid, (W, H), interpolation=cv2.INTER_LINEAR)
                angle = np.clip(angle / 60.0, 0, 1)
    except Exception:
        pass

    #  AMSR-2 
    amsr = np.zeros((H, W), dtype=np.float32)
    if "btemp_36.5v" in ds:
        da = ds["btemp_36.5v"]
        raw = da.values.astype(np.float32)
        raw = np.nan_to_num(raw, nan=225.0)
        raw = np.clip(raw, 150.0, 300.0)
        normed = (raw - 150.0) / 150.0
        if "sar_lines" in da.dims and "sar_samples" in da.dims:
            amsr = normed
        else:
            amsr = cv2.resize(normed, (W, H), interpolation=cv2.INTER_CUBIC)

    image = np.stack([hh_n, hv_n, angle, amsr], axis=0)  # [4, H, W]

    #  Labels (CT) 
    ct_lookup: dict[int, float] = {}
    if "polygon_codes" in ds:
        for raw_line in ds["polygon_codes"].values:
            line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, bytes) else str(raw_line)
            parts = line.split(";")
            if len(parts) >= 2:
                try:
                    ct_lookup[int(parts[0])] = SIGRID_TO_CT.get(int(parts[1]), 0.0)
                except ValueError:
                    pass

    ct_map = np.zeros((H, W), dtype=np.float32)
    if "polygon_icechart" in ds and ct_lookup:
        poly = np.nan_to_num(ds["polygon_icechart"].values, nan=-1).astype(np.int32)
        max_id = max(ct_lookup.keys())
        table = np.zeros(max_id + 1, dtype=np.float32)
        for pid, ct in ct_lookup.items():
            if 0 <= pid <= max_id:
                table[pid] = ct
        valid_mask = (poly >= 0) & (poly <= max_id)
        ct_map[valid_mask] = table[poly[valid_mask]]

    ds.close()
    return image, ct_map, land_mask


#  Sliding window 

@torch.no_grad()
def sliding_window_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    patch_size: int = 256,
    stride: int = 64,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_amp: bool = True,
) -> np.ndarray:
    """
    Inférence par fenêtre glissante sur la scène complète.

    Retourne un logit map [H, W] float32.
    Chaque pixel est inféré (patch_size/stride)² fois → moyenne des logits.
    """
    model.eval()
    C, H, W = image.shape
    logit_sum = np.zeros((H, W), dtype=np.float64)
    count     = np.zeros((H, W), dtype=np.float64)

    # Coordonnées de départ des fenêtres
    row_starts = list(range(0, H - patch_size + 1, stride))
    col_starts = list(range(0, W - patch_size + 1, stride))
    # S'assurer d'inclure les bords
    if row_starts[-1] + patch_size < H:
        row_starts.append(H - patch_size)
    if col_starts[-1] + patch_size < W:
        col_starts.append(W - patch_size)

    total_patches = len(row_starts) * len(col_starts)
    print(f"  Sliding window : {total_patches} patches ({len(row_starts)}×{len(col_starts)})…")

    for r in tqdm(row_starts, desc="Inférence (lignes)", ncols=80):
        for c in col_starts:
            patch = image[:, r:r+patch_size, c:c+patch_size]
            x = torch.from_numpy(patch).unsqueeze(0).to(device)  # [1, C, H, W]

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logit = model(x)                # [1, 1, H, W]

            logit_np = logit[0, 0].float().cpu().numpy()
            logit_sum[r:r+patch_size, c:c+patch_size] += logit_np
            count    [r:r+patch_size, c:c+patch_size] += 1.0

    count = np.maximum(count, 1.0)
    return (logit_sum / count).astype(np.float32)


#  Analytical Logit Scaling (ALS) 

def analytical_logit_scaling(
    logits: np.ndarray,
    land_mask: np.ndarray,
    sigma: float = 1.5,
    pct_low: float = 2.0,
    pct_high: float = 98.0,
    clip_val: float = 15.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Applique l'ALS de Reda et al. 2025.

    Étapes :
    1. Gaussian blur (filtre speckle, sigma ≈ 1–2 px)
    1.5. Clip robuste des logits extrêmes (|z| > clip_val).
         Sans ce clip, un modèle entraîné sur des données d'été peut générer des
         logits jusqu'à ±60, ce qui tire z_98% à 60, b à 26, et mappe toute la
         glace hivernale (logits modérés ≈ 5–10) à des concentrations quasi-nulles.
    2. Calcul analytique de T et b depuis les percentiles
    3. σ( (z - b) / T )

    Retourne
    --------
    prob_scaled : [H, W] probabilités binarisées
    prob_raw    : [H, W] probabilités sans scaling (pour comparaison)
    info        : dict avec T, b, z_2%, z_98%
    """
    # 1. Filtre Gaussian sur les logits (pixels hors terre uniquement)
    z = logits.copy()
    z_blurred = gaussian_filter(z, sigma=sigma)
    z_blurred[land_mask] = np.nan

    # 1.5. Clip robuste des logits extrêmes (hors terre seulement)
    if clip_val is not None and clip_val > 0:
        ocean_mask = ~land_mask
        z_blurred[ocean_mask] = np.clip(z_blurred[ocean_mask], -clip_val, clip_val)

    # 2. Calcul des percentiles (hors terre)
    valid_logits = z_blurred[~land_mask]
    z_lo = float(np.percentile(valid_logits, pct_low))
    z_hi = float(np.percentile(valid_logits, pct_high))

    b = (z_hi + z_lo) / 2.0
    T = (z_hi - z_lo) / 10.0  # stretch vers [-5, +5] → saturation sigmoid
    T = max(T, 1e-6)

    # 3. Application
    z_scaled = (z_blurred - b) / T
    prob_scaled = 1.0 / (1.0 + np.exp(-z_scaled))
    prob_raw    = torch.sigmoid(torch.tensor(z)).numpy()

    prob_scaled[land_mask] = np.nan
    prob_raw   [land_mask] = np.nan

    info = {"T": T, "b": b, "z_2pct": z_lo, "z_98pct": z_hi}
    return prob_scaled.astype(np.float32), prob_raw.astype(np.float32), info


#  Figures de présentation 

def save_result_figures(
    nc_path: Path,
    image: np.ndarray,
    ct_map: np.ndarray,
    land_mask: np.ndarray,
    prob_raw: np.ndarray,
    prob_scaled: np.ndarray,
    als_info: dict,
    out_dir: Path,
) -> None:
    """
    Génère deux figures :
    1. Figure principale 5 panneaux (publication-quality)
    2. Figure de détail zoom sur une région d'intérêt
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_name = nc_path.name[:16]

    #  Couleurs 
    # Colormap spéciale glace : blanc pour glace, bleu profond pour eau
    ice_colors = plt.cm.Blues_r  # blanc→bleu (glace=1→blanc, eau=0→bleu)

    #  Figure 1 : vue complète 5 panneaux 
    fig = plt.figure(figsize=(25, 8))
    fig.suptitle(
        f"Segmentation glace de mer — Attention U-Net  |  {scene_name}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # Sous-échantillonnage pour la figure (max 1000 px de large pour lisibilité)
    step = max(1, image.shape[2] // 1000)
    hh_disp = image[0, ::step, ::step]
    hv_disp = image[1, ::step, ::step]
    ct_disp = np.where(~land_mask[::step, ::step], ct_map[::step, ::step], np.nan)
    raw_disp = prob_raw[::step, ::step]
    scl_disp = prob_scaled[::step, ::step]

    axes = fig.subplots(1, 5, gridspec_kw={"wspace": 0.04})

    im0 = axes[0].imshow(hh_disp, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("SAR HH", fontsize=11, pad=6)
    axes[0].axis("off")

    im1 = axes[1].imshow(hv_disp, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("SAR HV", fontsize=11, pad=6)
    axes[1].axis("off")

    im2 = axes[2].imshow(ct_disp, cmap=ice_colors, vmin=0, vmax=1)
    axes[2].set_title("Label faible CT\n(polygones WMO)", fontsize=11, pad=6)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.04, label="Concentration")

    im3 = axes[3].imshow(raw_disp, cmap=ice_colors, vmin=0, vmax=1)
    axes[3].set_title("Prédiction brute\n(U-Net)", fontsize=11, pad=6)
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.04, label="P(glace)")

    im4 = axes[4].imshow(scl_disp, cmap=ice_colors, vmin=0, vmax=1)
    axes[4].set_title(
        f"Après ALS\n(T={als_info['T']:.3f}, b={als_info['b']:.3f})",
        fontsize=11, pad=6,
    )
    axes[4].axis("off")
    plt.colorbar(im4, ax=axes[4], fraction=0.04, label="P(glace)")

    path1 = out_dir / f"{scene_name}_result_full.png"
    fig.savefig(path1, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figure principale : {path1.name}")

    #  Figure 2 : zoom sur une sous-région 
    H, W = image.shape[1], image.shape[2]
    # Choisir la région centrale (souvent là où il y a le plus d'intérêt)
    r0, r1 = H // 4, H // 4 + H // 4
    c0, c1 = W // 4, W // 4 + W // 4

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle(
        f"Zoom région d'intérêt — {scene_name}  "
        f"[lignes {r0}–{r1}, cols {c0}–{c1}]",
        fontsize=13, fontweight="bold",
    )

    axes2[0].imshow(image[0, r0:r1, c0:c1], cmap="gray", vmin=0, vmax=1)
    axes2[0].set_title("SAR HH", fontsize=12)
    axes2[0].axis("off")

    im_b = axes2[1].imshow(prob_raw[r0:r1, c0:c1], cmap=ice_colors, vmin=0, vmax=1)
    axes2[1].set_title("Prédiction brute (sous-confiant)", fontsize=12)
    axes2[1].axis("off")
    plt.colorbar(im_b, ax=axes2[1], label="P(glace)")

    im_c = axes2[2].imshow(prob_scaled[r0:r1, c0:c1], cmap=ice_colors, vmin=0, vmax=1)
    axes2[2].set_title("Après ALS (fractures visibles)", fontsize=12)
    axes2[2].axis("off")
    plt.colorbar(im_c, ax=axes2[2], label="P(glace)")

    path2 = out_dir / f"{scene_name}_result_zoom.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"  Figure zoom       : {path2.name}")

    #  Figure 3 : distribution des logits (pour expliquer l'ALS) 
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
    fig3.suptitle("Analytical Logit Scaling — distribution des logits", fontsize=12)

    z = prob_raw.copy()
    valid_z = z[~np.isnan(z)].ravel()

    axes3[0].hist(valid_z, bins=200, color="steelblue", alpha=0.7, density=True)
    axes3[0].axvline(0.5, color="red", lw=2, label="0.5 (décision)")
    axes3[0].set_xlabel("P(glace) brute")
    axes3[0].set_ylabel("Densité")
    axes3[0].set_title("Distribution sans ALS\n(pileup autour de 0.5 = sous-confiance)")
    axes3[0].legend()

    valid_s = prob_scaled[~np.isnan(prob_scaled)].ravel()
    axes3[1].hist(valid_s, bins=200, color="coral", alpha=0.7, density=True)
    axes3[1].axvline(0.5, color="red", lw=2, label="0.5 (décision)")
    axes3[1].set_xlabel("P(glace) après ALS")
    axes3[1].set_ylabel("Densité")
    axes3[1].set_title("Distribution après ALS\n(binarisation physique eau/glace)")
    axes3[1].legend()

    path3 = out_dir / f"{scene_name}_als_distribution.png"
    fig3.savefig(path3, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig3)
    print(f"  Distribution ALS  : {path3.name}")


#  Évaluation quantitative 

def evaluate_scene(
    prob_scaled: np.ndarray,
    ct_map: np.ndarray,
    land_mask: np.ndarray,
    nc_path: Path,
    out_dir: Path,
) -> dict:
    """
    Évalue la conservation de la concentration macroscopique par polygone
    (Table I dans le papier Reda et al.).
    """
    ds = xr.open_dataset(str(nc_path), engine="h5netcdf")
    if "polygon_icechart" not in ds:
        ds.close()
        return {}

    poly = np.nan_to_num(ds["polygon_icechart"].values, nan=-1).astype(np.int32)
    ct_lookup: dict[int, float] = {}
    if "polygon_codes" in ds:
        for raw in ds["polygon_codes"].values:
            line = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
            parts = line.split(";")
            if len(parts) >= 2:
                try:
                    ct_lookup[int(parts[0])] = SIGRID_TO_CT.get(int(parts[1]), 0.0)
                except ValueError:
                    pass
    ds.close()

    results = []
    pred_binary = (prob_scaled >= 0.5).astype(np.float32)

    for pid, ct_target in ct_lookup.items():
        if pid < 0:
            continue
        mask = (poly == pid) & (~land_mask) & (~np.isnan(prob_scaled))
        if mask.sum() < 100:
            continue
        pred_mean = float(pred_binary[mask].mean())
        results.append({
            "polygon_id": pid,
            "ct_target":  ct_target,
            "ct_pred":    pred_mean,
            "abs_error":  abs(pred_mean - ct_target),
            "n_pixels":   int(mask.sum()),
        })

    if not results:
        return {}

    mae = np.mean([r["abs_error"] for r in results])
    acc = np.mean([(r["abs_error"] < 0.1) for r in results])  # à 10% près

    # Affichage tableau
    print(f"\n{'Polygone':>10} {'CT cible':>10} {'CT prédit':>10} {'Erreur':>8} {'N px':>8}")
    print("" * 55)
    for r in sorted(results, key=lambda x: x["polygon_id"]):
        print(f"{r['polygon_id']:>10}  {r['ct_target']:>9.3f}  "
              f"{r['ct_pred']:>9.3f}  {r['abs_error']:>7.3f}  {r['n_pixels']:>7}")
    print("" * 55)
    print(f"  MAE concentration : {mae:.4f}  |  Accuracy (±0.10) : {acc:.1%}")

    # Figure tableau
    fig, ax = plt.subplots(figsize=(9, max(3, len(results) * 0.5 + 1.5)))
    ax.axis("off")
    cell_text = [[r["polygon_id"], f"{r['ct_target']:.3f}",
                  f"{r['ct_pred']:.3f}", f"{r['abs_error']:.3f}"]
                 for r in sorted(results, key=lambda x: x["polygon_id"])]
    tbl = ax.table(
        cellText=cell_text,
        colLabels=["Polygone", "CT cible", "CT prédit", "|Erreur|"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.4)
    ax.set_title(
        f"Évaluation concentration macroscopique — {nc_path.name[:16]}\n"
        f"MAE = {mae:.4f}  |  Accuracy ±0.10 = {acc:.1%}",
        fontsize=11, pad=12,
    )
    path_tbl = out_dir / f"{nc_path.name[:16]}_eval_table.png"
    fig.savefig(path_tbl, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Tableau eval : {path_tbl.name}")

    return {"mae": mae, "accuracy": acc, "per_polygon": results}


#  Point d'entrée 

def run_inference(
    nc_path: str | Path,
    ckpt_path: str | Path,
    cfg: Config,
    out_dir: Optional[Path] = None,
) -> dict:
    """Lance le pipeline complet pour une scène."""
    nc_path = Path(nc_path)
    ckpt_path = Path(ckpt_path)
    if out_dir is None:
        out_dir = Path(cfg.save_dir) / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16

    #  Modèle 
    print(f"\n[1/4] Chargement du checkpoint : {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = AttentionUNet(
        n_channels=cfg.n_channels,
        base_features=cfg.base_features,
        dropout=0.0,
    ).to(device)
    # Gestion du préfixe "_orig_mod." ajouté par torch.compile
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    #  Scène 
    print(f"[2/4] Chargement de la scène…")
    image, ct_map, land_mask = load_scene(nc_path, cfg)

    #  Sliding window 
    print(f"[3/4] Inférence sliding window (stride={cfg.infer_stride})…")
    logits = sliding_window_inference(
        model, image, device,
        patch_size=cfg.patch_size,
        stride=cfg.infer_stride,
        amp_dtype=amp_dtype,
        use_amp=cfg.use_amp,
    )

    #  ALS 
    print(f"[4/4] Analytical Logit Scaling (σ={cfg.gaussian_sigma})…")
    prob_scaled, prob_raw, als_info = analytical_logit_scaling(
        logits, land_mask,
        sigma=cfg.gaussian_sigma,
        pct_low=cfg.als_pct_low,
        pct_high=cfg.als_pct_high,
        clip_val=getattr(cfg, "als_clip", 15.0),
    )
    print(f"  ALS : T={als_info['T']:.4f}  b={als_info['b']:.4f}  "
          f"z_2%={als_info['z_2pct']:.4f}  z_98%={als_info['z_98pct']:.4f}")

    #  Figures 
    save_result_figures(nc_path, image, ct_map, land_mask, prob_raw, prob_scaled, als_info, out_dir)

    #  Évaluation 
    eval_results = evaluate_scene(prob_scaled, ct_map, land_mask, nc_path, out_dir)

    return {"als_info": als_info, "eval": eval_results}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Inférence Attention U-Net — AI4Arctic")
    p.add_argument("--nc",    required=True, help="Fichier .nc de la scène")
    p.add_argument("--ckpt",  required=True, help="Checkpoint .pth (best_model.pth)")
    p.add_argument("--out",   default=None,  help="Dossier de sortie")
    p.add_argument("--stride", type=int, default=None)
    args = p.parse_args(argv)

    cfg = Config()
    if args.stride:
        cfg.infer_stride = args.stride

    run_inference(
        nc_path=args.nc,
        ckpt_path=args.ckpt,
        cfg=cfg,
        out_dir=Path(args.out) if args.out else None,
    )


if __name__ == "__main__":
    main()
