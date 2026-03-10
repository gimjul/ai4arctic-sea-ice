"""
losses.py — Fonctions de perte pour la supervision faible au niveau polygone.

Pourquoi la polygon-level loss est supérieure à la patch-level loss ?

L'approche naïve (step3_train.py) calcule la loss comme :
    MSE( mean(sigmoid(logits_patch)), CT_patch_moyen )

Problème : un patch 256×256 contient souvent plusieurs polygones différents
(ex : 70% et 90% de concentration). Moyenner donne CT=80%, ce qui n'est
pas un label réel. Le réseau apprend alors à prédire une valeur intermédiaire
partout → cartes floues.

Notre approche :
    Pour chaque polygone p dans le batch :
        loss += MSE( mean(sigmoid(logits[masque_p])), CT_p )

Le réseau doit respecter la concentration exacte de chaque polygone séparément.
C'est la vraie formulation "region-based loss" telle que définie dans le papier.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def polygon_mse_loss(
    logits: torch.Tensor,
    poly_map: torch.Tensor,
    ct_map: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """
    Loss MSE au niveau polygone (vraie weak supervision).

    Paramètres
    ----------
    logits   : [B, 1, H, W]  logits bruts (avant sigmoid)
    poly_map : [B, H, W]     int  — ID polygone par pixel (-1 = non défini)
    ct_map   : [B, H, W]     float — concentration CT ∈ [0, 1] par pixel
    valid    : [B, H, W]     bool  — pixels avec label valide

    Retourne
    --------
    loss scalaire (moyenne sur tous les polygones du batch)
    """
    # Float32 pour la loss (BF16 insuffisant pour MSE précise sur concentrations)
    pred = torch.sigmoid(logits.squeeze(1).float())   # [B, H, W]
    B = pred.shape[0]
    device = logits.device

    total = torch.zeros(1, device=device, dtype=torch.float32)
    n_polys = 0

    for b in range(B):
        # Masque des pixels valides pour ce batch item
        v = valid[b]                        # [H, W] bool
        pm = poly_map[b]                    # [H, W] int
        cm = ct_map[b]                      # [H, W] float
        pb = pred[b]                        # [H, W] float

        unique_ids = torch.unique(pm[v])    # IDs polygones valides

        for pid in unique_ids:
            if pid < 0:
                continue
            mask = v & (pm == pid)          # [H, W] bool
            n_pix = mask.sum()
            if n_pix == 0:
                continue

            pred_mean = pb[mask].mean()             # scalaire différentiable
            ct_target = cm[mask].mean().detach()    # cible fixe (pas de grad)

            total = total + (pred_mean - ct_target) ** 2
            n_polys += 1

    if n_polys == 0:
        # Aucun polygone valide dans le batch -> loss nulle connectee au graphe
        return (logits * 0.0).sum()
    return (total / n_polys).to(logits.dtype)


def entropy_regularization(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    Régularisation d'entropie : pénalise les prédictions intermédiaires (p ≈ 0.5).

    L'entropie binaire H(p) = -p*log(p) - (1-p)*log(1-p) est maximale en p=0.5.
    Minimiser cette pénalité pousse le réseau vers des prédictions nettes (0 ou 1),
    ce qui est physiquement correct (un pixel est soit glace, soit eau).

    Note : Cette régularisation remplace la "régularisation anti-binaire" de
    step3_train.py qui faisait l'inverse (pénalisait les prédictions binaires).
    Ici on encourage la binarisation, comme le fait l'ALS en post-traitement.
    """
    if valid.sum() == 0:
        # Aucun pixel valide -> loss nulle connectee au graphe
        return (logits * 0.0).sum().to(logits.dtype)

    # Calcul en float32 pour stabilité numérique (log() en BF16 -> NaN si proche 0/1)
    p = torch.sigmoid(logits.squeeze(1).float())[valid]   # pixels valides, float32
    p = p.clamp(1e-6, 1 - 1e-6)
    entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
    return entropy.mean().to(logits.dtype)                 # max à 0.693 (log 2)


def total_loss(
    logits: torch.Tensor,
    poly_map: torch.Tensor,
    ct_map: torch.Tensor,
    valid: torch.Tensor,
    lambda_entropy: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Loss combinée : MSE polygone + régularisation entropie.

    Retourne la loss totale + un dict de métriques pour le logging.
    """
    l_mse = polygon_mse_loss(logits, poly_map, ct_map, valid)
    l_ent = entropy_regularization(logits, valid)
    loss = l_mse + lambda_entropy * l_ent

    metrics = {
        "loss_mse":     float(l_mse.item()),
        "loss_entropy": float(l_ent.item()),
        "loss_total":   float(loss.item()),
    }
    return loss, metrics
