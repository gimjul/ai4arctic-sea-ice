import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetWeak(nn.Module):
    def __init__(self, n_channels=2, n_classes=1):
        super(UNetWeak, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- ENCODER (Descente) ---
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # --- DECODER (Montée) ---
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        # --- SORTIE ---
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # --- Decoder ---
        x = self.up1(x5)
        # Skip connection : on concatène avec x4
        # (Attention au padding si les dimensions ne matchent pas parfaitement, 
        # mais ici 256 est pair donc ça passe)
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
        
        # Carte de probabilité (Pixel-level) : entre 0 et 1
        pixel_map = torch.sigmoid(logits)

        # --- WEAK SUPERVISION HEAD ---
        # On fait la moyenne spatiale de la carte pour obtenir la concentration globale
        # dim=(2, 3) correspond à (Height, Width)
        concentration = pixel_map.mean(dim=(2, 3)) 

        return concentration, pixel_map

if __name__ == "__main__":
    # Test rapide
    model = UNetWeak(n_channels=2, n_classes=1)
    
    # Simulation d'un batch de 4 images de 256x256 (2 canaux)
    dummy_input = torch.randn(4, 2, 256, 256)
    
    concentration, map_out = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output Concentration shape: {concentration.shape}") # Doit être [4, 1]
    print(f"Output Map shape: {map_out.shape}")                 # Doit être [4, 1, 256, 256]