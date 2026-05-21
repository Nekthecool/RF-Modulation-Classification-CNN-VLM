import numpy as np
import zipfile
import os
import pandas as pd
import itertools
from signal_utils import *

print("Starting Train/Val Generation...")

MODE = "PRODUCTION" # Or "DEBUG"
np.random.seed(42)

MODULATIONS = {
    "4-ASK": ask_mod(4), "8-ASK": ask_mod(8), "BPSK": psk_mod(2), "QPSK": psk_mod(4),
    "4-HQAM": hqam_mod(4), "16-HQAM": hqam_mod(16), "64-HQAM": hqam_mod(64),
    "16-QAM": qam_mod(16), "32-QAM": qam_mod(32), "64-QAM": qam_mod(64),
    "128-QAM": qam_mod(128), "256-QAM": qam_mod(256),
    "16-APSK": apsk_mod([4, 12], [0.3780, 1.1339]), "32-APSK": apsk_mod([4, 12, 16], [0.25, 0.75, 1.25]),
    "64-APSK": apsk_mod([4, 12, 16, 32], [0.1754, 0.5262, 0.8770, 1.2278]),
    "128-APSK": apsk_mod([4, 12, 16, 32, 64], [0.1327, 0.3982, 0.6638, 0.9293, 1.1947])
}

SNR_dB_levels = [0, 10, 20, 30]
SEVERITY_LEVELS = ["none", "medium", "extreme"]
N = 2000

if MODE == "PRODUCTION":
    train_reps, val_reps = 12, 3
    total_reps = train_reps + val_reps
    train_images_count = len(MODULATIONS) * len(SNR_dB_levels) * (len(SEVERITY_LEVELS)**4) * train_reps
    val_images_count = len(MODULATIONS) * len(SNR_dB_levels) * (len(SEVERITY_LEVELS)**4) * val_reps
else:
    images_per_mod = 240
    train_reps_debug = int(images_per_mod * 0.8)
    val_reps_debug = images_per_mod - train_reps_debug
    train_images_count = len(MODULATIONS) * train_reps_debug
    val_images_count = len(MODULATIONS) * val_reps_debug

total_images = train_images_count + val_images_count

# The limit for coordinates mapping
base_limit = np.max(np.abs(MODULATIONS["256-QAM"]))
global_max_limit = base_limit * 3.0

print("==================================================")
print(f" STARTING HIGH-SPEED GENERATION (Mode: {MODE})")
print(f" Total Target Images: {total_images}")
print("==================================================\n")

temp_train_file = 'temp_train_images.npy'
temp_val_file = 'temp_val_images.npy'

# ΝΕΟ: Η εικόνα 384x384 έχει 147.456 pixels. Διαιρώντας με το 8, χρειαζόμαστε 18.432 bytes!
train_dset = np.lib.format.open_memmap(temp_train_file, mode='w+', dtype=np.uint8, shape=(train_images_count, 18432))
val_dset = np.lib.format.open_memmap(temp_val_file, mode='w+', dtype=np.uint8, shape=(val_images_count, 18432))

train_labels, val_labels = [], []
train_idx, val_idx = 0, 0

if MODE == "PRODUCTION":
    iterator = itertools.product(MODULATIONS.items(), SNR_dB_levels, SEVERITY_LEVELS, SEVERITY_LEVELS, SEVERITY_LEVELS, SEVERITY_LEVELS, range(total_reps))
    progress_step = 5000
else:
    def debug_generator():
        for mod_name, constellation in MODULATIONS.items():
            for rep in range(images_per_mod):
                yield ((mod_name, constellation), np.random.choice(SNR_dB_levels), np.random.choice(SEVERITY_LEVELS), np.random.choice(SEVERITY_LEVELS), np.random.choice(SEVERITY_LEVELS), np.random.choice(SEVERITY_LEVELS), rep)
    iterator = debug_generator()
    progress_step = 500

for i, (mod_data, current_snr, pn_sev, iqi_sev, amp_sev, jam_sev, rep) in enumerate(iterator):
    mod_name, constellation = mod_data

    if (i + 1) % progress_step == 0 or i == 0:
        print(f"[{MODE}] Fast Progress: {i+1}/{total_images} images processed...")

    # Signal Logic
    tx_symbols = np.random.choice(constellation, N)
    rx_symbols = apply_iq_imbalance(tx_symbols, severity=iqi_sev)
    rx_symbols = apply_phase_noise(rx_symbols, severity=pn_sev)
    rx_symbols = apply_amplitude_distortion(rx_symbols, severity=amp_sev)
    rx_symbols = apply_interference(rx_symbols, severity=jam_sev)
    rx_symbols = apply_awgn(rx_symbols, snr_dB=current_snr)

    gray_image = render_constellation_fast(rx_symbols, img_size=384, limit=global_max_limit)

    packed_image = np.packbits(gray_image > 127)

    snr_range = "Low" if current_snr <= 10 else "Medium" if current_snr <= 20 else "High"
    is_val = (rep >= train_reps) if MODE == "PRODUCTION" else (rep >= int(images_per_mod * 0.8))

    label_data = {
        "Modulation": mod_name, "SNR_dB": current_snr, "SNR_Range": snr_range,
        "Phase_Noise_Severity": pn_sev, "IQ_Imbalance_Severity": iqi_sev,
        "Amplitude_Distortion_Severity": amp_sev, "Interference_Severity": jam_sev
    }

    if is_val:
        label_data["Image_ID"] = val_idx
        val_dset[val_idx] = packed_image   # Αποθηκεύουμε την πακεταρισμένη!
        val_labels.append(label_data)
        val_idx += 1
    else:
        label_data["Image_ID"] = train_idx
        train_dset[train_idx] = packed_image # Αποθηκεύουμε την πακεταρισμένη!
        train_labels.append(label_data)
        train_idx += 1

# Flush buffers safely to disk once done
train_dset.flush()
val_dset.flush()

print(f"\nSuccess! Fast-Saved {train_idx} Train images and {val_idx} Val images directly to disk.")

# ====================================================================
# STREAM INTO COMPRESSED NPZ & CLEANUP
# ====================================================================

print(f"\n--- Zipping Arrays into .NPZ Format ---")

train_npz_file = f'train_images_{MODE.lower()}.npz'
val_npz_file = f'val_images_{MODE.lower()}.npz'

with zipfile.ZipFile(train_npz_file, 'w', zipfile.ZIP_STORED) as zf:
    zf.write(temp_train_file, arcname='images.npy')

with zipfile.ZipFile(val_npz_file, 'w', zipfile.ZIP_STORED) as zf:
    zf.write(temp_val_file, arcname='images.npy')

# Remove temporary uncompressed files
os.remove(temp_train_file)
os.remove(temp_val_file)

# Save labels to CSV
train_lbl_file = f'train_labels_{MODE.lower()}.csv'
val_lbl_file = f'val_labels_{MODE.lower()}.csv'

pd.DataFrame(train_labels).to_csv(train_lbl_file, index=False)
pd.DataFrame(val_labels).to_csv(val_lbl_file, index=False)

# Clean success message with absolute local paths
print(f"\nSUCCESS! Files saved locally at:")
print(f" -> {os.path.abspath(train_npz_file)}")
print(f" -> {os.path.abspath(train_lbl_file)}")
print(f" -> {os.path.abspath(val_npz_file)}")
print(f" -> {os.path.abspath(val_lbl_file)}")