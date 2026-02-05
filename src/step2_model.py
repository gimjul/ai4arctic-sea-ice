"""
step2_model.py

Reseau "U-Net" minimal pour une segmentation en supervision faible.

Entree (x):
- tenseur de forme [B, C, H, W] (un patch multi-canaux)
  Exemples de canaux: SAR HH, SAR HV, angle d'incidence, temperature AMSR, etc.

Sorties:
- pixel_map: [B, 1, H, W] : probabilite de "glace" par pixel (entre 0 et 1)
- concentration: [B, 1] : moyenne spatiale de pixel_map (concentration moyenne du patch)

Pourquoi cette tete "concentration" ?
Dans le projet, on ne dispose pas d'un masque pixel-wise fiable. En revanche, on
connait une contrainte de proportion (CT) au niveau polygone/zone. Le modele
apprend donc une carte pixel-wise dont la moyenne respecte cette contrainte.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Bloc standard U-Net: 2x (Conv3x3 -> BatchNorm -> ReLU)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNetWeak(nn.Module):
    """
    U-Net "classique" (encodeur -> goulot -> decodeur) + supervision faible.

    Parametres:
    - n_channels: nombre de canaux en entree (ex: 4)
    - n_classes: nombre de canaux en sortie (ici 1: probabilite de glace)
    """

    def __init__(self, n_channels: int = 2, n_classes: int = 1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encodeur: on descend en resolution pour capturer du contexte spatial.
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # Decodeur: on remonte en resolution et on recolle les details via les
        # skip connections (concat avec les features de l'encodeur).
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        # Sortie: 1x1 conv pour produire 1 logit par pixel.
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encodeur
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decodeur (+ skip connections)
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)

        logits = self.outc(x)
        pixel_map = torch.sigmoid(logits)

        # Supervision faible: concentration globale = moyenne de la carte pixel-wise.
        concentration = pixel_map.mean(dim=(2, 3))
        return concentration, pixel_map


if __name__ == "__main__":
    # Sanity check de formes.
    model = UNetWeak(n_channels=2, n_classes=1)
    dummy_input = torch.randn(4, 2, 256, 256)
    concentration, map_out = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output Concentration shape: {concentration.shape}")  # attendu: [4, 1]
    print(f"Output Map shape: {map_out.shape}")                  # attendu: [4, 1, 256, 256]
