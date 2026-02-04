import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2 

class AI4ArcticWeaklyLabeledDataset(Dataset):
    def __init__(self, nc_file_path, patch_size=256, augment=False):
        self.patch_size = patch_size
        self.augment = augment
        
        # 1. Ouverture du fichier
        # On force engine="netcdf4" pour bien lire les attributs complexes
        self.ds = xr.open_dataset(nc_file_path, engine="netcdf4")
        
        # Dimensions cibles
        self.height = self.ds.sizes['sar_lines']
        self.width = self.ds.sizes['sar_samples']
        
        self.n_patches_h = self.height // patch_size
        self.n_patches_w = self.width // patch_size
        
        filename = nc_file_path.split('/')[-1] if isinstance(nc_file_path, str) else str(nc_file_path)
        print(f"Pré-traitement : {filename}...")
        
        # --- A. AMSR-2 (Température) ---
        try:
            raw_amsr = self.ds["btemp_36.5v"].values
            raw_amsr = np.nan_to_num(raw_amsr, nan=0.0)
            
            # Upscaling Cubic pour éviter les blocs
            self.amsr_full = cv2.resize(
                raw_amsr.astype(np.float32), 
                (self.width, self.height), 
                interpolation=cv2.INTER_CUBIC 
            )
            
            # Normalisation PHYSIQUE (150K - 300K) -> [0, 1]
            min_k, max_k = 150.0, 300.0
            self.amsr_full = (np.clip(self.amsr_full, min_k, max_k) - min_k) / (max_k - min_k)
            
        except KeyError:
            print("⚠️ AMSR-2 manquant. Canal vide.")
            self.amsr_full = np.zeros((self.height, self.width), dtype=np.float32)

        # --- B. Angle d'Incidence (CORRECTION CRITIQUE) ---
        try:
            # 1. Gestion du vecteur 1D (Le fix "Rayures")
            if 'sar_incidenceangles' in self.ds:
                raw_angle = self.ds['sar_incidenceangles'].values
                if raw_angle.ndim == 1:
                    raw_angle = raw_angle[np.newaxis, :] # Force (1, W)
                
                self.angle_full = cv2.resize(
                    raw_angle.astype(np.float32), 
                    (self.width, self.height), 
                    interpolation=cv2.INTER_LINEAR
                )
            
            # 2. Gestion de la Grille (Fallback)
            elif 'sar_grid_incidenceangle' in self.ds:
                raw_angle = self.ds['sar_grid_incidenceangle'].values
                side = int(np.sqrt(raw_angle.size))
                if side * side == raw_angle.size:
                    raw_angle = raw_angle.reshape(side, side)
                    self.angle_full = cv2.resize(
                        raw_angle.astype(np.float32), 
                        (self.width, self.height), 
                        interpolation=cv2.INTER_LINEAR
                    )
                else:
                    self.angle_full = np.full((self.height, self.width), 30.0, dtype=np.float32)

            # Normalisation PHYSIQUE (0 - 60 degrés) -> [0, 1]
            self.angle_full = np.clip(self.angle_full / 60.0, 0, 1)
            
        except Exception as e:
            print(f"⚠️ Erreur Angle ({e}). Moyenne 0.5 utilisée.")
            self.angle_full = np.full((self.height, self.width), 0.5, dtype=np.float32)

        # --- C. CONSTRUCTION DYNAMIQUE DU DICTIONNAIRE (AUTO-PARSING) ---
        self.ct_lookup = {}
        
        # Table de conversion standard SIGRID-3 (Code Météo -> Concentration %)
        SIGRID_CONVERSION = {
            92: 100.0, 91: 95.0, 90: 90.0,
            81: 90.0, 80: 80.0,
            70: 70.0, 60: 60.0, 50: 50.0,
            40: 40.0, 30: 30.0, 20: 20.0, 10: 10.0,
            1: 0.0, 0: 0.0
        }

        try:
            # Lecture de la variable contenant la table interne du fichier
            if "polygon_codes" in self.ds:
                raw_codes = self.ds["polygon_codes"].values
                # print(f"   📖 Table interne trouvée : {len(raw_codes)} entrées.")
                
                for raw_line in raw_codes:
                    # Gestion encodage bytes/string
                    if isinstance(raw_line, bytes):
                        line = raw_line.decode('utf-8')
                    else:
                        line = str(raw_line)
                    
                    # Parsing format "id;CT;CA;..."
                    parts = line.split(';')
                    
                    if len(parts) >= 2:
                        try:
                            poly_id = int(parts[0])
                            ct_code = int(parts[1])
                            # Mapping Code Météo -> Pourcentage
                            self.ct_lookup[poly_id] = SIGRID_CONVERSION.get(ct_code, 0.0)
                        except ValueError:
                            continue
            else:
                print("⚠️ Variable 'polygon_codes' absente. Dictionnaire vide.")

        except Exception as e:
            print(f"⚠️ Erreur parsing table: {e}")
            self.ct_lookup = {}

    def __len__(self):
        return self.n_patches_h * self.n_patches_w

    def __getitem__(self, idx):
        row = idx // self.n_patches_w
        col = idx % self.n_patches_w
        y_start, y_end = row * self.patch_size, (row + 1) * self.patch_size
        x_start, x_end = col * self.patch_size, (col + 1) * self.patch_size
        
        # --- D. SAR ---
        try:
            hh = self.ds["nersc_sar_primary"].isel(sar_lines=slice(y_start, y_end), sar_samples=slice(x_start, x_end)).values
            hv = self.ds["nersc_sar_secondary"].isel(sar_lines=slice(y_start, y_end), sar_samples=slice(x_start, x_end)).values
        except KeyError:
            hh = self.ds["sar_primary"].isel(sar_lines=slice(y_start, y_end), sar_samples=slice(x_start, x_end)).values
            hv = self.ds["sar_secondary"].isel(sar_lines=slice(y_start, y_end), sar_samples=slice(x_start, x_end)).values
        
        hh = np.nan_to_num(np.abs(hh), nan=0.0)
        hv = np.nan_to_num(np.abs(hv), nan=0.0)
        
        # Normalisation dB [-30, 20]
        hh_norm = (np.clip(10 * np.log10(hh + 1e-6), -30, 20) + 30) / 50
        hv_norm = (np.clip(10 * np.log10(hv + 1e-6), -30, 20) + 30) / 50
        
        # Auxiliaires
        angle_patch = self.angle_full[y_start:y_end, x_start:x_end]
        amsr_patch = self.amsr_full[y_start:y_end, x_start:x_end]

        image = np.stack([hh_norm, hv_norm, angle_patch, amsr_patch], axis=0).astype(np.float32)

        # --- E. LABEL (OPTIMISÉ) ---
        polygon_ids = self.ds["polygon_icechart"].isel(sar_lines=slice(y_start, y_end), sar_samples=slice(x_start, x_end)).values
        polygon_ids = np.nan_to_num(polygon_ids, nan=0).astype(int)
        
        # OPTIMISATION : Remplacement de la double boucle par une vectorisation
        # Beaucoup plus rapide pour l'entraînement
        # On crée une fonction qui applique le dictionnaire
        vectorized_lookup = np.vectorize(lambda x: self.ct_lookup.get(x, 0.0))
        ct_map = vectorized_lookup(polygon_ids).astype(np.float32)

        # Moyenne 0-100 -> 0-1
        weak_label = np.mean(ct_map) / 100.0

        if self.augment:
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()
        
        return torch.from_numpy(image.copy()), torch.tensor([weak_label], dtype=torch.float32)