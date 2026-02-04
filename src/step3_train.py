import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from tqdm import tqdm  # On suppose que tu as installé tqdm (pip install tqdm)

# Imports de tes fichiers précédents (ou coller les classes ici)
from step1_data_loader import AI4ArcticWeaklyLabeledDataset 
from step2_model import UNetWeak

# --- CONFIGURATION CHAMPION ---
DATA_DIR = "Northwest_2019/" 
BATCH_SIZE = 16  
LEARNING_RATE = 1e-4
EPOCHS = 5      
LAMBDA_REG = 0.01 

def debug_viz(inputs, pred_map, targets, epoch, batch_idx):
    """Visualisation de contrôle"""
    sar_img = inputs[0, 0, :, :].cpu().detach().numpy()
    amsr_img = inputs[0, 3, :, :].cpu().detach().numpy()
    pred_mask = pred_map[0, 0, :, :].cpu().detach().numpy()
    true_conc = targets[0].item()
    pred_conc = np.mean(pred_mask)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(sar_img, cmap='gray'); plt.title("SAR HH")
    plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(amsr_img, cmap='magma'); plt.title("AMSR2 Temp")
    plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(pred_mask, cmap='jet', vmin=0, vmax=1) 
    plt.title(f"Pred: {pred_conc:.2f} / Cible: {true_conc:.2f}")
    plt.axis('off')
    
    os.makedirs("debug_plots", exist_ok=True)
    plt.savefig(f"debug_plots/epoch_{epoch+1}_batch_{batch_idx}.png")
    plt.close()

def train():
    # 1. Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Mode Turbo : Apple Metal (MPS)")
    else:
        device = torch.device("cpu")
        print("⚠️ Mode Standard : CPU")

    # 2. Data
    nc_files = glob.glob(os.path.join(DATA_DIR, "*.nc"))
    nc_files.sort()
    
    if not nc_files:
        print(f"❌ ERREUR : Pas de fichiers dans {DATA_DIR}")
        return

    print(f"📂 Chargement de {len(nc_files)} fichiers...")
    datasets = [AI4ArcticWeaklyLabeledDataset(f, patch_size=256, augment=True) for f in nc_files]
    
    mega_dataset = ConcatDataset(datasets)
    
    # drop_last=True évite les bugs de BatchNorm sur le dernier batch incomplet
    dataloader = DataLoader(mega_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    print(f"🚀 Dataset Unifié : {len(mega_dataset)} patches.")

    # 3. Model
    # Attention : n_channels=4 (HH, HV, Angle, Temp)
    model = UNetWeak(n_channels=4, n_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mse_criterion = nn.MSELoss()

    loss_history = []

    print("🥊 Début de l'entraînement...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Barre de progression correcte
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for i, (inputs, targets) in enumerate(loop):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward
            pred_concentration, pred_map = model(inputs)

            # Loss
            loss_mse = mse_criterion(pred_concentration, targets)
            loss_bin = torch.mean(pred_map * (1 - pred_map))
            loss = loss_mse + (LAMBDA_REG * loss_bin)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Mise à jour visuelle de la Loss dans la barre
            loop.set_postfix(loss=loss.item())

            # Debug Viz (1ère image de l'époque)
            if i == 0:
                debug_viz(inputs, pred_map, targets, epoch, i)

        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        # Sauvegarde unique par époque
        torch.save(model.state_dict(), f"weights_epoch_{epoch+1}.pth")

    # Final Plot
    plt.figure()
    plt.plot(loss_history, marker='o')
    plt.title("Optimisation Loss (Full Dataset)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE + Reg)")
    plt.grid(True)
    plt.savefig("loss_curve.png")
    print("✅ Entraînement terminé.")

if __name__ == "__main__":
    train()