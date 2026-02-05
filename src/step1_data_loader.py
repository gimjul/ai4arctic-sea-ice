"""
step1_data_loader.py

Dataset PyTorch pour les fichiers netCDF AI4Arctic (zone Northwest).

On travaille en patches (ex: 256x256) pour entrainer un modele de segmentation.

X (features) par patch:
- SAR HH (amplitude) -> conversion en dB -> normalisation [0, 1]
- SAR HV (amplitude) -> conversion en dB -> normalisation [0, 1]
- Angle d'incidence SAR -> normalisation [0, 1]
- 1 canal AMSR (temperature de brillance) -> normalisation [0, 1]

Y (label faible) par patch:
- un seul scalaire: concentration moyenne de glace (CT) sur le patch, derivee des
  polygones. On convertit polygon_id -> CT% via la table polygon_codes contenue
  dans le fichier, puis on moyenne sur les pixels du patch.

Important:
- on n'a pas de masque pixel-wise "vrai".
- on entraine ensuite un reseau qui predit une carte pixel-wise, dont la moyenne
  doit respecter cette concentration CT.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class AI4ArcticWeaklyLabeledDataset(Dataset):
    """Un Dataset = un fichier .nc, qui fournit une grille reguliere de patches."""

    def __init__(self, nc_file_path: str, patch_size: int = 256, augment: bool = False):
        self.patch_size = int(patch_size)
        self.augment = bool(augment)

        # Sur Windows, l'engine "netcdf4" peut echouer quand le chemin contient des
        # caracteres non-ASCII (ex: "reels"). "h5netcdf" est plus robuste ici.
        self.ds = xr.open_dataset(nc_file_path, engine="h5netcdf")

        self.height = int(self.ds.sizes["sar_lines"])
        self.width = int(self.ds.sizes["sar_samples"])

        # On decoupe en patches sur une grille reguliere. Les bords incomplets sont ignores.
        self.n_patches_h = self.height // self.patch_size
        self.n_patches_w = self.width // self.patch_size

        filename = Path(nc_file_path).name
        print(f"Pre-traitement : {filename}...")

        # AMSR (temperature de brillance).
        # Pour eviter de stocker une scene 10k x 10k (trop lourd), on privilegie:
        # - soit une grille basse resolution stockee en RAM (si AMSR est low-res)
        # - soit une lecture patch-wise directe (si AMSR est deja sur la grille SAR)
        self.amsr_var = None
        self.amsr_mode = "none"  # "none" | "lowres" | "sar_grid"
        self.amsr_lowres = None

        if "btemp_36.5v" in self.ds:
            self.amsr_var = "btemp_36.5v"
            da = self.ds[self.amsr_var]
            if ("sar_lines" in da.dims) and ("sar_samples" in da.dims):
                self.amsr_mode = "sar_grid"
            else:
                raw = da.values.astype(np.float32)
                raw = np.nan_to_num(raw, nan=0.0)
                min_k, max_k = 150.0, 300.0
                raw = np.clip(raw, min_k, max_k)
                self.amsr_lowres = ((raw - min_k) / (max_k - min_k)).astype(np.float32)
                self.amsr_mode = "lowres"
        else:
            self.amsr_mode = "none"

        # Angle d'incidence.
        # Souvent fourni comme un vecteur 1D (largeur W). On le stocke comme tel, puis
        # on reconstruit un patch 2D par repetition verticale.
        self.angle_vector = np.full((self.width,), 0.5, dtype=np.float32)
        try:
            if "sar_incidenceangles" in self.ds:
                raw_angle = self.ds["sar_incidenceangles"].values
                raw_angle = np.nan_to_num(raw_angle, nan=30.0).astype(np.float32)
                if raw_angle.ndim == 2:
                    if raw_angle.shape[0] == 1:
                        raw_angle = raw_angle[0]
                    else:
                        raw_angle = raw_angle.mean(axis=0)
                self.angle_vector = np.clip(raw_angle / 60.0, 0.0, 1.0).astype(np.float32)
            elif "sar_grid_incidenceangle" in self.ds:
                raw_angle = self.ds["sar_grid_incidenceangle"].values.astype(np.float32)
                raw_angle = np.nan_to_num(raw_angle, nan=30.0)
                side = int(np.sqrt(raw_angle.size))
                if side * side == raw_angle.size:
                    grid = raw_angle.reshape(side, side)
                    vec = cv2.resize(grid, (self.width, 1), interpolation=cv2.INTER_LINEAR).reshape(-1)
                    self.angle_vector = np.clip(vec / 60.0, 0.0, 1.0).astype(np.float32)
        except Exception:
            self.angle_vector = np.full((self.width,), 0.5, dtype=np.float32)

        # Construction du mapping polygon_id -> CT%.
        # Le fichier contient une table "polygon_codes" (texte) qui associe a chaque
        # polygone des codes SIGRID-3, dont on derive une concentration.
        self.ct_lookup: dict[int, float] = {}

        sigrid_to_ct = {
            92: 100.0,
            91: 95.0,
            90: 90.0,
            81: 90.0,
            80: 80.0,
            70: 70.0,
            60: 60.0,
            50: 50.0,
            40: 40.0,
            30: 30.0,
            20: 20.0,
            10: 10.0,
            1: 0.0,
            0: 0.0,
        }

        try:
            if "polygon_codes" in self.ds:
                raw_codes = self.ds["polygon_codes"].values
                for raw_line in raw_codes:
                    if isinstance(raw_line, bytes):
                        line = raw_line.decode("utf-8", errors="ignore")
                    else:
                        line = str(raw_line)

                    parts = line.split(";")
                    if len(parts) < 2:
                        continue

                    try:
                        poly_id = int(parts[0])
                        ct_code = int(parts[1])
                    except ValueError:
                        continue

                    self.ct_lookup[poly_id] = float(sigrid_to_ct.get(ct_code, 0.0))
        except Exception:
            self.ct_lookup = {}

        # Acceleration: on transforme le dictionnaire en table numpy (indexable).
        # But: au lieu d'un lookup Python pixel-par-pixel, on fait une indexation numpy.
        if self.ct_lookup:
            valid_ids = [pid for pid in self.ct_lookup.keys() if pid >= 0]
            self._ct_table_max_id = int(max(valid_ids)) if valid_ids else 0
            self._ct_table = np.zeros((self._ct_table_max_id + 1,), dtype=np.float32)
            for pid, ct in self.ct_lookup.items():
                if 0 <= pid <= self._ct_table_max_id:
                    self._ct_table[pid] = np.float32(ct)
        else:
            self._ct_table_max_id = 0
            self._ct_table = np.zeros((1,), dtype=np.float32)

    def __len__(self) -> int:
        return self.n_patches_h * self.n_patches_w

    def __getitem__(self, idx: int):
        row = idx // self.n_patches_w
        col = idx % self.n_patches_w

        y0 = row * self.patch_size
        y1 = (row + 1) * self.patch_size
        x0 = col * self.patch_size
        x1 = (col + 1) * self.patch_size

        # SAR HH/HV (lecture patch-wise pour eviter de charger l'image complete).
        try:
            hh = self.ds["nersc_sar_primary"].isel(sar_lines=slice(y0, y1), sar_samples=slice(x0, x1)).values
            hv = self.ds["nersc_sar_secondary"].isel(sar_lines=slice(y0, y1), sar_samples=slice(x0, x1)).values
        except KeyError:
            hh = self.ds["sar_primary"].isel(sar_lines=slice(y0, y1), sar_samples=slice(x0, x1)).values
            hv = self.ds["sar_secondary"].isel(sar_lines=slice(y0, y1), sar_samples=slice(x0, x1)).values

        hh = np.nan_to_num(np.abs(hh), nan=0.0).astype(np.float32)
        hv = np.nan_to_num(np.abs(hv), nan=0.0).astype(np.float32)

        # Normalisation: amplitude -> dB -> clip -> [0, 1].
        hh_db = 10.0 * np.log10(hh + 1e-6)
        hv_db = 10.0 * np.log10(hv + 1e-6)
        hh_norm = (np.clip(hh_db, -30.0, 20.0) + 30.0) / 50.0
        hv_norm = (np.clip(hv_db, -30.0, 20.0) + 30.0) / 50.0

        # Angle: on prend un morceau du vecteur, puis on le repete sur la hauteur.
        angle_vec = self.angle_vector[x0:x1]
        if angle_vec.shape[0] != self.patch_size:
            angle_vec = np.pad(angle_vec, (0, max(0, self.patch_size - angle_vec.shape[0])), mode="edge")[: self.patch_size]
        angle_patch = np.repeat(angle_vec[np.newaxis, :], self.patch_size, axis=0).astype(np.float32)

        # AMSR: 3 cas.
        if self.amsr_mode == "none" or self.amsr_var is None:
            amsr_patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        elif self.amsr_mode == "sar_grid":
            raw = self.ds[self.amsr_var].isel(sar_lines=slice(y0, y1), sar_samples=slice(x0, x1)).values.astype(np.float32)
            raw = np.nan_to_num(raw, nan=0.0)
            min_k, max_k = 150.0, 300.0
            raw = np.clip(raw, min_k, max_k)
            amsr_patch = ((raw - min_k) / (max_k - min_k)).astype(np.float32)
        else:
            assert self.amsr_lowres is not None
            h_a, w_a = self.amsr_lowres.shape
            yy0 = int(np.floor(y0 * h_a / self.height))
            yy1 = int(np.ceil(y1 * h_a / self.height))
            xx0 = int(np.floor(x0 * w_a / self.width))
            xx1 = int(np.ceil(x1 * w_a / self.width))

            yy0 = int(np.clip(yy0, 0, h_a - 1))
            xx0 = int(np.clip(xx0, 0, w_a - 1))
            yy1 = int(np.clip(yy1, yy0 + 1, h_a))
            xx1 = int(np.clip(xx1, xx0 + 1, w_a))

            crop = self.amsr_lowres[yy0:yy1, xx0:xx1]
            amsr_patch = cv2.resize(crop, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)

        # Tenseur final: [C, H, W]
        image = np.stack([hh_norm, hv_norm, angle_patch, amsr_patch], axis=0).astype(np.float32)

        # Label faible: on map polygon_id -> CT% -> moyenne.
        poly = self.ds["polygon_icechart"].isel(sar_lines=slice(y0, y1), sar_samples=slice(x0, x1)).values
        poly = np.nan_to_num(poly, nan=0).astype(np.int32)

        max_id = int(self._ct_table_max_id)
        if max_id <= 0:
            ct_map = np.zeros_like(poly, dtype=np.float32)
        else:
            poly_clip = np.clip(poly, 0, max_id)
            ct_map = self._ct_table[poly_clip]
            # Les ids hors table doivent valoir 0 (et pas "max_id").
            ct_map = ct_map.astype(np.float32, copy=False)
            oob = poly > max_id
            if np.any(oob):
                ct_map = ct_map.copy()
                ct_map[oob] = 0.0

        weak_label = float(ct_map.mean() / 100.0)

        if self.augment:
            # Flips simples (rapides). Permet de multiplier les configurations sans
            # changer la statistique globale du patch.
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()

        x = torch.from_numpy(image.copy())
        y = torch.tensor([weak_label], dtype=torch.float32)
        return x, y
