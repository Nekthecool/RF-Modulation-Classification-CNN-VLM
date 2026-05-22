import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

# Import our custom classes
from model import DeepMultiTaskCNN
from dataset import ConstellationDataset

# ====================================================================
# EVALUATION CONFIGURATION
# ====================================================================
WEIGHTS_PATH = './cnn_best_weights.pth'
PLOTS_DIR = './Evaluation_Plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepMultiTaskCNN().to(device)

if os.path.exists(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    print(f"✅ Loaded hybrid model from {WEIGHTS_PATH}")
else:
    raise FileNotFoundError(f"❌ ERROR: Model weights not found at {WEIGHTS_PATH}. Train the model first!")

# Print Computational Complexity
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"⚙️ Computational Complexity: Total Trainable Parameters = {total_params:,}")

print("\n--- Initializing Evaluation Datasets ---")
val_dataset = ConstellationDataset('val_images_production.npz', 'val_labels_production.csv', 'val_mapped.npy')
test_dataset = ConstellationDataset('test_images_generalization.npz', 'test_labels_generalization.csv', 'test_mapped.npy')

val_loader = DataLoader(val_dataset, batch_size=64, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, num_workers=2, pin_memory=True)

# ====================================================================
# INFERENCE ENGINE
# ====================================================================
def get_predictions(dataloader, restrict_to_known=False):
    model.eval()
    tasks_class = ['mod', 'snr']
    tasks_reg = ['pn', 'iqi', 'amp_dist', 'jamming']
    
    all_preds = {t: [] for t in tasks_class + tasks_reg}
    all_targets = {t: [] for t in tasks_class + tasks_reg}
    all_snr = []

    total_inference_time = 0.0
    num_batches = 0
    num_samples = 0

    torch.cuda.empty_cache()
    with torch.no_grad():
        for images, labels, snr_db in dataloader:
            images = images.to(device)

            # Inference Time Measurement
            start_time = time.time()
            outputs = model(images)
            if device.type == 'cuda': torch.cuda.synchronize()
            end_time = time.time()

            total_inference_time += (end_time - start_time)
            num_batches += 1
            num_samples += images.size(0)

            for t in tasks_class:
                all_preds[t].extend(torch.argmax(outputs[t], dim=1).cpu().numpy())
                all_targets[t].extend(labels[t].numpy())

            for t in tasks_reg:
                raw_preds = outputs[t].view(-1).clamp(0, 4).cpu().numpy()
                if restrict_to_known:
                    valid_classes = np.array([0, 2, 4])
                    closest_idx = np.abs(raw_preds[:, None] - valid_classes).argmin(axis=1)
                    final_preds = valid_classes[closest_idx]
                else:
                    final_preds = np.round(raw_preds).astype(int)
                
                all_preds[t].extend(final_preds)
                all_targets[t].extend(labels[t].numpy())

            all_snr.extend(snr_db.numpy())

    avg_batch_time = (total_inference_time / num_batches) * 1000  # ms
    avg_sample_time = (total_inference_time / num_samples) * 1000 # ms
    print(f"   ⏱️ Inference Speed: {avg_batch_time:.2f} ms/batch | {avg_sample_time:.2f} ms/sample")

    return {t: np.array(all_targets[t]) for t in all_targets}, {t: np.array(all_preds[t]) for t in all_preds}, np.array(all_snr)

print("\nRunning Inference on Validation (Knowns)...")
val_targets, val_preds, val_snr = get_predictions(val_loader, restrict_to_known=True)

print("Running Inference on Test (Generalization)...")
test_targets, test_preds, test_snr = get_predictions(test_loader, restrict_to_known=False)

# ====================================================================
# METRICS & LOGS
# ====================================================================
tasks_display = ['MOD', 'PN', 'IQI', 'SNR', 'AMP_DIST', 'JAMMING']
tasks_keys = ['mod', 'pn', 'iqi', 'snr', 'amp_dist', 'jamming']

print("\n--- Per-Task Validation Accuracy ---")
for tk, td in zip(tasks_keys, tasks_display):
    acc = accuracy_score(val_targets[tk], val_preds[tk]) * 100
    print(f"Task '{td}': {acc:.2f}%")

print("\n--- Per-Task Test (Generalization) Accuracy ---")
for tk, td in zip(tasks_keys, tasks_display):
    acc = accuracy_score(test_targets[tk], test_preds[tk]) * 100
    print(f"Task '{td}': {acc:.2f}%")

# ====================================================================
# PLOT GENERATION
# ====================================================================
sns.set_theme(style="whitegrid")

# 1. Confusion Matrix
print("\n[1/5] Generating Confusion Matrix...")
fig, ax = plt.subplots(figsize=(10, 8))
ax.grid(False)
ConfusionMatrixDisplay(
    confusion_matrix(val_targets['mod'], val_preds['mod']), 
    display_labels=val_dataset.all_mods
).plot(ax=ax, cmap='Blues', colorbar=True, xticks_rotation=90, text_kw={'fontsize': 8})
plt.title('Validation Set: Modulation Confusion Matrix', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '01_Confusion_Matrix.png'), dpi=300)
plt.close()

# 2. SNR Robustness Curve
print("[2/5] Generating SNR Robustness Curve...")
def calc_snr_acc(targets, preds, snrs):
    return {s: accuracy_score(targets['mod'][snrs==s], preds['mod'][snrs==s])*100 for s in np.unique(snrs) if len(snrs==s)>0}

val_snr_c = calc_snr_acc(val_targets, val_preds, val_snr)
test_snr_c = calc_snr_acc(test_targets, test_preds, test_snr)

plt.figure(figsize=(9, 5))
plt.plot(list(val_snr_c.keys()), list(val_snr_c.values()), 'bo-', label='Validation', linewidth=2)
plt.plot(list(test_snr_c.keys()), list(test_snr_c.values()), 'rs--', label='Test (Generalization)', linewidth=2)
plt.title('Robustness: Modulation Accuracy vs SNR', fontsize=14, weight='bold')
plt.xlabel('SNR (dB)')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 105)
plt.legend()
plt.grid(True, ls='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '02_SNR_Robustness.png'), dpi=300)
plt.close()

# 3. Severity Sensitivity (MAE)
print("[3/5] Generating Severity Sensitivity Chart (MAE)...")
sev_levels = ['none', 'low', 'medium', 'high', 'extreme']
imp_tasks = {'Phase Noise': 'pn', 'IQ Imbalance': 'iqi', 'Amp Distortion': 'amp_dist', 'Jamming': 'jamming'}
comb_t = {t: np.concatenate([val_targets[v], test_targets[v]]) for t, v in imp_tasks.items()}
comb_p = {t: np.concatenate([val_preds[v], test_preds[v]]) for t, v in imp_tasks.items()}

plt.figure(figsize=(10, 6))
for i, (name, key) in enumerate(imp_tasks.items()):
    maes = [mean_absolute_error(comb_t[name][comb_t[name]==s], comb_p[name][comb_t[name]==s]) if np.any(comb_t[name]==s) else np.nan for s in range(5)]
    plt.plot(sev_levels, maes, marker=['o', 's', '^', 'D'][i], linewidth=2.5, markersize=8, label=name)

plt.title('Sensitivity to Impairment Severity (Mean Absolute Error)', fontsize=14, weight='bold')
plt.xlabel('Severity Level')
plt.ylabel('Mean Absolute Error (Lower is Better)')
plt.ylim(0, 2.0)
plt.legend()
plt.grid(True, ls='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '03_Severity_Sensitivity_MAE.png'), dpi=300)
plt.close()

# 4. Severity Accuracy (%) - NEW PLOT
print("[4/5] Generating Severity Accuracy Chart (%)...")
plt.figure(figsize=(10, 6))
for i, (name, key) in enumerate(imp_tasks.items()):
    accs = [accuracy_score(comb_t[name][comb_t[name]==s], comb_p[name][comb_t[name]==s])*100 if np.any(comb_t[name]==s) else np.nan for s in range(5)]
    plt.plot(sev_levels, accs, marker=['o', 's', '^', 'D'][i], linewidth=2.5, markersize=8, label=name)

plt.title('Accuracy by Impairment Severity', fontsize=14, weight='bold')
plt.xlabel('Severity Level')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 105)
plt.legend()
plt.grid(True, ls='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '04_Severity_Accuracy.png'), dpi=300)
plt.close()

# 5. Generalization Gap
print("[5/5] Generating Generalization Gap Bar Chart...")
unkn_counts = test_dataset.df['Unknown_Params_Count'].values
u_vals = sorted(np.unique(unkn_counts))
gen_acc = [accuracy_score(test_targets['mod'][unkn_counts==c], test_preds['mod'][unkn_counts==c])*100 for c in u_vals]

plt.figure(figsize=(8, 5))
bars = plt.bar(u_vals, gen_acc, color='purple', alpha=0.7, edgecolor='black', linewidth=1.5)
for b in bars: 
    plt.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f'{b.get_height():.1f}%', ha='center', weight='bold')

plt.title('Generalization Capability vs Unknown Conditions', fontsize=14, weight='bold')
plt.xlabel('Number of Unknown Parameters (OOD)')
plt.ylabel('Modulation Accuracy (%)')
plt.ylim(0, 105)
plt.xticks(u_vals)
plt.grid(axis='y', ls='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '05_Generalization_Gap.png'), dpi=300)
plt.close()

print(f"\n✅ All plots saved successfully to: {os.path.abspath(PLOTS_DIR)}")
