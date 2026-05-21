import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm

# Import our custom classes
from model import DeepMultiTaskCNN
from dataset import ConstellationDataset

# ====================================================================
# TRAINING CONFIGURATION
# ====================================================================
BATCH_SIZE = 128
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
WEIGHTS_SAVE_PATH = './cnn_best_weights.pth'

print("--- Initializing Training & Validation Datasets ---")
# Assuming datasets are in the same local directory
train_dataset = ConstellationDataset('train_images_production.npz', 'train_labels_production.csv', 'train_mapped.npy')
val_dataset = ConstellationDataset('val_images_production.npz', 'val_labels_production.csv', 'val_mapped.npy')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print("DataLoaders Ready!")

# Initialize Model and Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")
model = DeepMultiTaskCNN().to(device)

# Optimizers and Schedulers
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# Loss Functions
criterion_class = nn.CrossEntropyLoss()
criterion_reg = nn.HuberLoss()  # Robust to outliers (values 0 and 4)

best_val_loss = float('inf')
epochs_no_improve = 0
best_model_weights = copy.deepcopy(model.state_dict())

# ====================================================================
# TRAINING LOOP
# ====================================================================
print("\n--- Starting Full Hybrid Training ---")
for epoch in range(NUM_EPOCHS):
    
    # --- TRAINING PHASE ---
    model.train()
    running_train_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train", leave=False, colour='green')

    for images, labels, _ in train_bar:
        images = images.to(device)
        targets = {k: v.to(device) for k, v in labels.items()}

        optimizer.zero_grad()
        preds = model(images)

        # Classification Loss
        loss_mod = criterion_class(preds['mod'], targets['mod'])
        loss_snr = criterion_class(preds['snr'], targets['snr'])

        # Regression Loss
        loss_pn = criterion_reg(preds['pn'], targets['pn'].float().view(-1, 1))
        loss_iqi = criterion_reg(preds['iqi'], targets['iqi'].float().view(-1, 1))
        loss_amp = criterion_reg(preds['amp_dist'], targets['amp_dist'].float().view(-1, 1))
        loss_jam = criterion_reg(preds['jamming'], targets['jamming'].float().view(-1, 1))

        # Total Loss: Custom weighted balancing to prioritize modulation
        loss = (3.0 * loss_mod) + loss_snr + (0.5 * loss_pn) + (0.5 * loss_iqi) + (0.75 * loss_amp) + (0.5 * loss_jam)

        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
        train_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = running_train_loss / len(train_loader)

    # --- VALIDATION PHASE ---
    model.eval()
    running_val_loss = 0.0
    val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] Val  ", leave=False, colour='blue')

    with torch.no_grad():
        for images, labels, _ in val_bar:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in labels.items()}
            preds = model(images)

            l_mod = criterion_class(preds['mod'], targets['mod'])
            l_snr = criterion_class(preds['snr'], targets['snr'])
            l_pn = criterion_reg(preds['pn'], targets['pn'].float().view(-1, 1))
            l_iqi = criterion_reg(preds['iqi'], targets['iqi'].float().view(-1, 1))
            l_amp = criterion_reg(preds['amp_dist'], targets['amp_dist'].float().view(-1, 1))
            l_jam = criterion_reg(preds['jamming'], targets['jamming'].float().view(-1, 1))

            loss = (3.0 * l_mod) + l_snr + (0.5 * l_pn) + (0.5 * l_iqi) + (0.75 * l_amp) + (0.5 * l_jam)
            running_val_loss += loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    scheduler.step(avg_val_loss)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # --- EARLY STOPPING & SAVING ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        best_model_weights = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), WEIGHTS_SAVE_PATH)
        print(f"   -> Model improved! Weights saved to {WEIGHTS_SAVE_PATH}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n[!] Early stopping triggered after {epoch+1} epochs.")
            break

print("\nTraining Complete! Best model preserved.")