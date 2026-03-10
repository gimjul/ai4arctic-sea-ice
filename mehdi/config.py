"""
config.py — Configuration centralisée du projet sea ice segmentation.

Choix architectural : Attention U-Net + polygon-level weak supervision loss.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Données ──────────────────────────────────────────────────────────────
    data_dir: str = "../sujet/Northwest"   # dataset complet (59 scenes, 37 GB)
    patch_size: int = 256
    augment: bool = True
    n_channels: int = 4        # HH, HV, angle incidence, AMSR-2
    num_workers: int = 4      

    # Split train/val temporel : 2018 (53 scenes) -> train | 2019 (6 scenes) -> val
    val_scene_keyword: str = "2019"  # toutes les scenes 2019 = validation

    # ── Modèle ───────────────────────────────────────────────────────────────
    base_features: int = 64    # doublement à chaque niveau : 64→128→256→512→1024
    dropout: float = 0.1       # dropout dans les blocs encodeur

    # ── Entraînement ─────────────────────────────────────────────────────────
    batch_size: int = 16       
    epochs: int = 50          
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3   
    lambda_entropy: float = 0.01  
    grad_clip: float = 1.0         # gradient clipping

    # ── Hardware ─────────────────────────────────────────────────────────────
    device: str = "auto"           
    use_amp: bool = True
    amp_dtype: str = "bfloat16"    
    compile_model: bool = False   
    pin_memory: bool = True
    persistent_workers: bool = True

    # ── Inférence (scène complète) ────────────────────────────────────────────
    infer_stride: int = 64         # stride sliding window (overlap = 256/64 = 4x par dimension)
    gaussian_sigma: float = 1.5    # blur sur logits avant ALS
    als_pct_low: float = 2.0       # percentile bas pour l'Analytical Logit Scaling
    als_pct_high: float = 98.0     # percentile haut
    als_clip: float = 15.0         # clip robuste des logits avant ALS (evite z_98%>>15 apres training ete)

    # ── Sorties ───────────────────────────────────────────────────────────────
    save_dir: str = "outputs_northwest"
    debug_every: int = 1           # sauvegarder viz debug tous les N epochs
    save_best_only: bool = True    # ne conserver que le meilleur checkpoint val


# Instance globale importable directement
cfg = Config()
