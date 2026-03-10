"""
dataset.py — Dataset PyTorch pour AI4Arctic (Northwest 2019).

Différences clés par rapport à step1_data_loader.py :
1. Retourne `ct_map` [H, W] et `poly_map` [H, W] au lieu d'un scalaire.
   → Permet une loss au niveau polygone (vraie weak supervision).
2. Augmentations appliquées cohéremment à l'image ET aux cartes de labels.
3. Canal 5 optionnel : ratio HH/HV (discriminant pour le type de glace).
4. valid_mask [H, W] : True là où un polygone valide est défini (hors terre).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


# ─── Table de conversion SIGRID-3 CT_code → concentration [0, 1] ─────────────
SIGRID_TO_CT: dict[int, float] = {
    92: 1.00, 91: 0.95, 90: 0.90,
    81: 0.90, 80: 0.80, 70: 0.70,
    60: 0.60, 55: 0.55, 50: 0.50,
    40: 0.40, 30: 0.30, 20: 0.20,
    10: 0.10,  1: 0.00,  0: 0.00,
}


class AI4ArcticDataset(Dataset):
    """
    Dataset patch-wise pour une scène AI4Arctic.

    Paramètres
    ----------
    nc_path : chemin vers un fichier .nc
    patch_size : taille des patches carrés (ex: 256)
    augment : active les augmentations spatiales (flip + rotation 90°)
    add_ratio : ajoute le canal HH/HV (ratio dB) comme 5e canal
    """

    def __init__(
        self,
        nc_path: str | Path,
        patch_size: int = 256,
        augment: bool = False,
        add_ratio: bool = False,
    ):
        self.patch_size = int(patch_size)
        self.augment = bool(augment)
        self.add_ratio = bool(add_ratio)
        self.nc_path = Path(nc_path)

        self.ds = xr.open_dataset(str(nc_path), engine="h5netcdf")
        self.H = int(self.ds.sizes["sar_lines"])
        self.W = int(self.ds.sizes["sar_samples"])
        self.n_ph = self.H // self.patch_size
        self.n_pw = self.W // self.patch_size

        print(f"  [{self.nc_path.name[:35]}]  scene {self.H}x{self.W}  "
              f"->  {self.n_ph * self.n_pw} patches")

        self._build_amsr()
        self._build_angle()
        self._build_ct_table()

    # ── Construction des features globales ────────────────────────────────────

    def _build_amsr(self) -> None:
        """Charge et normalise AMSR-2 btemp_36.5v."""
        self.amsr_mode = "none"
        self.amsr_lowres: Optional[np.ndarray] = None

        if "btemp_36.5v" not in self.ds:
            return

        da = self.ds["btemp_36.5v"]
        raw = da.values.astype(np.float32)
        raw = np.nan_to_num(raw, nan=225.0)
        raw = np.clip(raw, 150.0, 300.0)
        normed = (raw - 150.0) / 150.0  # → [0, 1]

        if "sar_lines" in da.dims and "sar_samples" in da.dims:
            # Déjà sur la grille SAR haute résolution
            self.amsr_mode = "sar_grid"
            self.amsr_hires = normed
        else:
            # Basse résolution : on stocke en RAM et on redimentionne à la demande
            self.amsr_mode = "lowres"
            self.amsr_lowres = normed

    def _build_angle(self) -> None:
        """Construit le vecteur d'angle d'incidence normalisé [0, 1]."""
        self.angle_vector = np.full((self.W,), 0.5, dtype=np.float32)
        try:
            if "sar_incidenceangles" in self.ds:
                raw = self.ds["sar_incidenceangles"].values.astype(np.float32)
                raw = np.nan_to_num(raw, nan=30.0)
                if raw.ndim == 2:
                    raw = raw.mean(axis=0) if raw.shape[0] > 1 else raw[0]
                self.angle_vector = np.clip(raw / 60.0, 0.0, 1.0)
            elif "sar_grid_incidenceangle" in self.ds:
                raw = self.ds["sar_grid_incidenceangle"].values.astype(np.float32)
                raw = np.nan_to_num(raw, nan=30.0)
                side = int(np.sqrt(raw.size))
                if side * side == raw.size:
                    grid = raw.reshape(side, side)
                    vec = cv2.resize(grid, (self.W, 1), interpolation=cv2.INTER_LINEAR).flatten()
                    self.angle_vector = np.clip(vec / 60.0, 0.0, 1.0)
        except Exception:
            pass

    def _build_ct_table(self) -> None:
        """Construit la LUT polygon_id → CT [0, 1]."""
        self.ct_lookup: dict[int, float] = {}

        if "polygon_codes" not in self.ds:
            self._ct_table = np.zeros(1, dtype=np.float32)
            self._ct_max_id = 0
            return

        for raw_line in self.ds["polygon_codes"].values:
            line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, bytes) else str(raw_line)
            parts = line.split(";")
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
                ct_code = int(parts[1])
                self.ct_lookup[pid] = SIGRID_TO_CT.get(ct_code, 0.0)
            except ValueError:
                continue

        if self.ct_lookup:
            valid_ids = [pid for pid in self.ct_lookup if pid >= 0]
            self._ct_max_id = int(max(valid_ids))
            self._ct_table = np.zeros(self._ct_max_id + 1, dtype=np.float32)
            for pid, ct in self.ct_lookup.items():
                if 0 <= pid <= self._ct_max_id:
                    self._ct_table[pid] = ct
        else:
            self._ct_max_id = 0
            self._ct_table = np.zeros(1, dtype=np.float32)

    # ── Interface Dataset ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.n_ph * self.n_pw

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retourne
        --------
        image    : [C, H, W]  float32  (C = 4 ou 5 si add_ratio)
        ct_map   : [H, W]     float32  CT ∈ [0, 1] par pixel selon son polygone
        poly_map : [H, W]     int32    ID du polygone (-1 = terre/non défini)
        valid    : [H, W]     bool     masque des pixels avec label valide
        """
        row, col = divmod(idx, self.n_pw)
        y0, y1 = row * self.patch_size, (row + 1) * self.patch_size
        x0, x1 = col * self.patch_size, (col + 1) * self.patch_size

        # ── SAR HH / HV ──────────────────────────────────────────────────────
        hh_key = "nersc_sar_primary" if "nersc_sar_primary" in self.ds else "sar_primary"
        hv_key = "nersc_sar_secondary" if "nersc_sar_secondary" in self.ds else "sar_secondary"

        hh = self.ds[hh_key].isel(sar_lines=slice(y0, y1), sar_samples=slice(x0, x1)).values
        hv = self.ds[hv_key].isel(sar_lines=slice(y0, y1), sar_samples=slice(x0, x1)).values

        hh = np.nan_to_num(np.abs(hh), nan=0.0).astype(np.float32)
        hv = np.nan_to_num(np.abs(hv), nan=0.0).astype(np.float32)

        # Amplitude → dB → normalisation [0, 1]
        hh_db = 10.0 * np.log10(hh + 1e-6)
        hv_db = 10.0 * np.log10(hv + 1e-6)
        hh_n = (np.clip(hh_db, -30.0, 20.0) + 30.0) / 50.0
        hv_n = (np.clip(hv_db, -30.0, 20.0) + 30.0) / 50.0

        # ── Angle d'incidence ─────────────────────────────────────────────────
        angle_vec = self.angle_vector[x0:x1]
        if len(angle_vec) < self.patch_size:
            angle_vec = np.pad(angle_vec, (0, self.patch_size - len(angle_vec)), mode="edge")
        angle = np.repeat(angle_vec[np.newaxis, :], self.patch_size, axis=0)

        # ── AMSR-2 ────────────────────────────────────────────────────────────
        if self.amsr_mode == "sar_grid":
            amsr = self.amsr_hires[y0:y1, x0:x1].astype(np.float32)
        elif self.amsr_mode == "lowres":
            ha, wa = self.amsr_lowres.shape
            yy0 = int(np.clip(np.floor(y0 * ha / self.H), 0, ha - 1))
            yy1 = int(np.clip(np.ceil(y1 * ha / self.H), yy0 + 1, ha))
            xx0 = int(np.clip(np.floor(x0 * wa / self.W), 0, wa - 1))
            xx1 = int(np.clip(np.ceil(x1 * wa / self.W), xx0 + 1, wa))
            crop = self.amsr_lowres[yy0:yy1, xx0:xx1]
            amsr = cv2.resize(crop, (self.patch_size, self.patch_size),
                              interpolation=cv2.INTER_CUBIC).astype(np.float32)
        else:
            amsr = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)

        # ── Assemblage canaux ─────────────────────────────────────────────────
        channels = [hh_n, hv_n, angle, amsr]
        if self.add_ratio:
            # Ratio HH/HV en dB → discriminant pour type de glace
            ratio = np.clip((hh_db - hv_db + 30.0) / 60.0, 0.0, 1.0)
            channels.append(ratio)
        image = np.stack(channels, axis=0).astype(np.float32)  # [C, H, W]

        # ── Labels : poly_map et ct_map ───────────────────────────────────────
        poly_raw = self.ds["polygon_icechart"].isel(
            sar_lines=slice(y0, y1), sar_samples=slice(x0, x1)
        ).values
        poly_raw = np.nan_to_num(poly_raw, nan=-1).astype(np.int32)

        # Pixels valides (polygone défini, hors terre)
        valid_mask = poly_raw >= 0

        # Carte CT via lookup table vectorisé
        poly_clip = np.clip(poly_raw, 0, self._ct_max_id)
        ct_map = self._ct_table[poly_clip].astype(np.float32)
        ct_map[~valid_mask] = 0.0
        # Polygones hors table
        oob = poly_raw > self._ct_max_id
        if np.any(oob):
            ct_map[oob] = 0.0
            valid_mask[oob] = False

        poly_map = poly_raw.copy()
        poly_map[~valid_mask] = -1

        # ── Augmentations ─────────────────────────────────────────────────────
        if self.augment:
            image, ct_map, poly_map, valid_mask = _augment(image, ct_map, poly_map, valid_mask)

        return (
            torch.from_numpy(image),
            torch.from_numpy(ct_map),
            torch.from_numpy(poly_map),
            torch.from_numpy(valid_mask),
        )


# ─── Augmentations ────────────────────────────────────────────────────────────

_AUG_OPS = [
    lambda x: x,                                           # identité
    lambda x: np.flip(x, axis=-1).copy(),                  # flip horizontal
    lambda x: np.flip(x, axis=-2).copy(),                  # flip vertical
    lambda x: np.rot90(x, k=1, axes=(-2, -1)).copy(),      # rot 90°
    lambda x: np.rot90(x, k=2, axes=(-2, -1)).copy(),      # rot 180°
    lambda x: np.rot90(x, k=3, axes=(-2, -1)).copy(),      # rot 270°
    lambda x: np.flip(np.rot90(x, k=1, axes=(-2, -1)), axis=-1).copy(),  # flip + rot
    lambda x: np.flip(np.rot90(x, k=3, axes=(-2, -1)), axis=-1).copy(),
]


def _augment(image, ct_map, poly_map, valid_mask):
    op_idx = np.random.randint(len(_AUG_OPS))
    op = _AUG_OPS[op_idx]
    image = op(image)
    ct_map = op(ct_map[np.newaxis])[0]
    poly_map = op(poly_map[np.newaxis])[0]
    valid_mask = op(valid_mask[np.newaxis])[0]
    return image, ct_map, poly_map, valid_mask


# ─── Utilitaire : split train/val ─────────────────────────────────────────────

def split_scenes(
    data_dir: str | Path,
    val_keyword: str = "20190522",
) -> tuple[list[Path], list[Path]]:
    """Sépare les fichiers .nc en train/val selon un mot-clé de date."""
    all_files = sorted(Path(data_dir).glob("*.nc"))
    train = [f for f in all_files if val_keyword not in f.name]
    val = [f for f in all_files if val_keyword in f.name]
    print(f"Scènes d'entraînement ({len(train)}) : {[f.name[:8] for f in train]}")
    print(f"Scènes de validation  ({len(val)})  : {[f.name[:8] for f in val]}")
    return train, val
