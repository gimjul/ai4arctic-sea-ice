"""
model.py — Attention U-Net pour la segmentation de glace de mer (weak supervision).

Référence : Oktay et al., 2018 — "Attention U-Net: Learning Where to Look for
the Pancreas" — https://arxiv.org/abs/1804.03999

Pourquoi Attention U-Net vs U-Net classique ?

1. Les attention gates filtrent les skip connections : le décodeur ne "voit" que
   les régions spatiales pertinentes de l'encodeur, pas toute la feature map.
2. Cela supprime les artefacts liés au fond (terre, zones côtières) et améliore
   la détection des leads (fractures fines) dans les zones à fort contraste.
3. La complexité supplémentaire est faible (~+5% de paramètres), mais le gain
   qualitatif sur des images SAR bruitées est significatif.

Architecture :
  Encodeur : 4 niveaux (DoubleConv + MaxPool)
  Goulot    : 1 DoubleConv (résolution 16×16 pour patch 256×256)
  Décodeur  : 4 niveaux (Upsample + AttentionGate + DoubleConv)
  Sortie    : Conv1×1 → 1 logit par pixel (pas de sigmoid ici, fait dans la loss)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Blocs de base ────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU (avec dropout optionnel)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool2×2 → DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─── Attention Gate ────────────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Gate d'attention additive (Oktay et al. 2018).

    Paramètres
    ----------
    F_g  : canaux du signal de gating (venant du décodeur, résolution plus basse)
    F_l  : canaux du skip connection (venant de l'encodeur)
    F_int: canaux intermédiaires du gate (typiquement F_l // 2)

    Fonctionnement
    --------------
    α = σ( ψ( ReLU( W_g(g↑) + W_l(l) ) ) )  ∈ [0, 1]  par pixel
    output = α * l          ← les features encodeur filtrées spatialement
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_l = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        """
        g : signal de gating  [B, F_g, H', W']  (résolution décodeur)
        l : skip connection   [B, F_l, H,  W ]  (résolution encodeur)
        """
        # Remonter g à la résolution de l si nécessaire
        g_up = F.interpolate(self.W_g(g), size=l.shape[2:], mode="bilinear", align_corners=False)
        l_proj = self.W_l(l)
        alpha = self.psi(self.relu(g_up + l_proj))  # [B, 1, H, W]
        return alpha * l                             # gate multiplicatif


# ─── Bloc décodeur ─────────────────────────────────────────────────────────────

class Up(nn.Module):
    """Upsample → AttentionGate → concat(skip_filtré, upsampled) → DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.att = AttentionGate(F_g=in_ch // 2, F_l=skip_ch, F_int=skip_ch // 2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip_att = self.att(g=x, l=skip)
        x = torch.cat([skip_att, x], dim=1)
        return self.conv(x)


# ─── Attention U-Net ───────────────────────────────────────────────────────────

class AttentionUNet(nn.Module):
    """
    Attention U-Net complet.

    Paramètres
    ----------
    n_channels    : nombre de canaux d'entrée (4 ou 5)
    base_features : nombre de filtres du premier bloc (doublé à chaque descente)
    dropout       : taux de dropout dans les blocs DoubleConv
    """

    def __init__(
        self,
        n_channels: int = 4,
        base_features: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        f = base_features
        self.n_channels = n_channels

        # ── Encodeur ──────────────────────────────────────────────────────────
        self.enc1 = DoubleConv(n_channels, f, dropout)         # 256 → 256,  f
        self.enc2 = Down(f,     f * 2, dropout)                # 256 → 128,  2f
        self.enc3 = Down(f * 2, f * 4, dropout)                # 128 →  64,  4f
        self.enc4 = Down(f * 4, f * 8, dropout)                #  64 →  32,  8f

        # ── Goulot ────────────────────────────────────────────────────────────
        self.bottleneck = Down(f * 8, f * 16, dropout=0.0)     #  32 →  16, 16f

        # ── Décodeur ──────────────────────────────────────────────────────────
        self.up4 = Up(f * 16, f * 8,  f * 8)                   #  16 →  32
        self.up3 = Up(f * 8,  f * 4,  f * 4)                   #  32 →  64
        self.up2 = Up(f * 4,  f * 2,  f * 2)                   #  64 → 128
        self.up1 = Up(f * 2,  f,      f)                       # 128 → 256

        # ── Tête de sortie ────────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(f, 1, kernel_size=1)

        # Initialisation des poids (He pour Conv, 0/1 pour BN)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paramètre : x [B, C, H, W]
        Retourne  : logits [B, 1, H, W] (avant sigmoid)
        """
        # Encodeur
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Goulot
        b = self.bottleneck(e4)

        # Décodeur (+ attention gates sur les skip connections)
        d4 = self.up4(b,  e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        return self.out_conv(d1)   # [B, 1, H, W]

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = AttentionUNet(n_channels=4, base_features=64, dropout=0.1)
    print(f"Paramètres entraînables : {model.num_parameters:,}")

    dummy = torch.randn(2, 4, 256, 256)
    logits = model(dummy)
    print(f"Input : {dummy.shape}  →  Logits : {logits.shape}")  # [2, 1, 256, 256]

    probs = torch.sigmoid(logits)
    print(f"Probabilités min/max : {probs.min():.4f} / {probs.max():.4f}")
