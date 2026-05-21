import numpy as np
import zipfile
import os
import pandas as pd
from signal_utils import *

print("Starting Test Set (Generalization) Generation...")

np.random.seed(99) # Different seed for the test set

MODULATIONS = {
    "4-ASK": ask_mod(4), "8-ASK": ask_mod(8), "BPSK": psk_mod(2), "QPSK": psk_mod(4),
    "4-HQAM": hqam_mod(4), "16-HQAM": hqam_mod(16), "64-HQAM": hqam_mod(64),
    "16-QAM": qam_mod(16), "32-QAM": qam_mod(32), "64-QAM": qam_mod(64),
    "128-QAM": qam_mod(128), "256-QAM": qam_mod(256),
    "16-APSK": apsk_mod([4, 12], [0.3780, 1.1339]), "32-APSK": apsk_mod([4, 12, 16], [0.25, 0.75, 1.25]),
    "64-APSK": apsk_mod([4, 12, 16, 32], [0.1754, 0.5262, 0.8770, 1.2278]),
    "128-APSK": apsk_mod([4, 12, 16, 32, 64], [0.1327, 0.3982, 0.6638, 0.9293, 1.1947])
}

# --- THE "KNOWN" VS "UNKNOWN" POOLS ---
KNOWN_SNR = [0, 10, 20, 30]
UNKNOWN_SNR = [5, 15, 25]

KNOWN_SEV = ["none", "medium", "extreme"]
UNKNOWN_SEV = ["low", "high"] # The intermediate impairment levels

ALL_SNR = KNOWN_SNR + UNKNOWN_SNR
ALL_SEV = KNOWN_SEV + UNKNOWN_SEV

N = 2000
images_per_mod = 1000  # 16 classes * 1000 = 16,000 images in total (matches Val set size)
total_test_images = len(MODULATIONS) * images_per_mod

base_limit = np.max(np.abs(MODULATIONS["256-QAM"]))
global_max_limit = base_limit * 3.0

print("==================================================")
print(f" STARTING TEST SET GENERATION (Generalization)")
print(f" Total Target Images: {total_test_images}")
print(" Rules: At least 1 unknown parameter, but NOT exclusively unknowns.")
print("==================================================\n")

temp_test_file = 'temp_test_images.npy'
test_dset = np.lib.format.open_memmap(temp_test_file, mode='w+', dtype=np.uint8, shape=(total_test_images, 18432))

test_labels = []
idx = 0

for mod_name, constellation in MODULATIONS.items():
    print(f"Generating 1000 hybrid OOD samples for {mod_name}...")
    for _ in range(images_per_mod):

        valid_combo = False
        while not valid_combo:
            # Randomly pick from the combined pools
            current_snr = np.random.choice(ALL_SNR)
            pn_sev = np.random.choice(ALL_SEV)
            iqi_sev = np.random.choice(ALL_SEV)
            amp_sev = np.random.choice(ALL_SEV)
            jam_sev = np.random.choice(ALL_SEV)

            # Count how many parameters are "Unseen/Unknown"
            ood_count = 0
            if current_snr in UNKNOWN_SNR: ood_count += 1
            if pn_sev in UNKNOWN_SEV: ood_count += 1
            if iqi_sev in UNKNOWN_SEV: ood_count += 1
            if amp_sev in UNKNOWN_SEV: ood_count += 1
            if jam_sev in UNKNOWN_SEV: ood_count += 1

            # Constraint Checklist:
            # 1. ood_count >= 1 -> Ensures at least one condition is completely new.
            # 2. ood_count <= 4 -> Ensures it is NOT exclusively unknown (max is 5).
            if 1 <= ood_count <= 4:
                valid_combo = True

        # Signal Logic
        tx_symbols = np.random.choice(constellation, N)
        rx_symbols = apply_iq_imbalance(tx_symbols, severity=iqi_sev)
        rx_symbols = apply_phase_noise(rx_symbols, severity=pn_sev)
        rx_symbols = apply_amplitude_distortion(rx_symbols, severity=amp_sev)
        rx_symbols = apply_interference(rx_symbols, severity=jam_sev)
        rx_symbols = apply_awgn(rx_symbols, snr_dB=current_snr)

        # Image Math rendering & Bit-Packing
        gray_image = render_constellation_fast(rx_symbols, img_size=384, limit=global_max_limit)
        packed_image = np.packbits(gray_image > 127)

        # Assign SNR Range properly (including the negative SNRs)
        if current_snr <= 5:
            snr_range = "Low"
        elif current_snr <= 20:
            snr_range = "Medium"
        else:
            snr_range = "High"

        label_data = {
            "Image_ID": idx, "Modulation": mod_name,
            "SNR_dB": current_snr, "SNR_Range": snr_range,
            "Phase_Noise_Severity": pn_sev, "IQ_Imbalance_Severity": iqi_sev,
            "Amplitude_Distortion_Severity": amp_sev, "Interference_Severity": jam_sev,
            "Unknown_Params_Count": ood_count # Helpful for deeper Pandas analysis later!
        }

        test_dset[idx] = packed_image
        test_labels.append(label_data)
        idx += 1

test_dset.flush()
print(f"\nSuccess! Fast-Saved {idx} Test images directly to disk.")

# ====================================================================
# ZIPPING & DOWNLOAD
# ====================================================================
test_npz_file = 'test_images_generalization.npz'
with zipfile.ZipFile(test_npz_file, 'w', zipfile.ZIP_STORED) as zf:
    zf.write(temp_test_file, arcname='images.npy')

# Remove temporary uncompressed file
os.remove(temp_test_file)

# Save labels to CSV
test_lbl_file = 'test_labels_generalization.csv'
pd.DataFrame(test_labels).to_csv(test_lbl_file, index=False)

# Clean success message with absolute local paths
print(f"\nSUCCESS! Test Set files saved locally at:")
print(f" -> {os.path.abspath(test_npz_file)}")
print(f" -> {os.path.abspath(test_lbl_file)}")