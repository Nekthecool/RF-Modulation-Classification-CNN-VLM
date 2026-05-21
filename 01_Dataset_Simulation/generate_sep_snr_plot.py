import numpy as np
import matplotlib.pyplot as plt
import os

# Import our custom mathematical functions
from signal_utils import *

print("\n==================================================")
print(" STARTING SEP vs SNR SIMULATION FOR QAM")
print("==================================================\n")

N_sim = 100000  # High symbol count guarantees smooth curves
sim_snr_dB = np.arange(0, 25, 1)
sigma_phase_noise = 0.08  # Wrapped Gaussian phase noise standard deviation

plt.figure(figsize=(10, 7))
colors = ['blue', 'green', 'red', 'orange', 'purple']
qam_orders = [16, 32, 64, 128, 256]

for idx, M in enumerate(qam_orders):
    print(f"Running Monte Carlo loop for {M}-QAM...")
    constellation = qam_mod(M)

    sep_empirical_pn = []    # With Phase Noise
    sep_empirical_ideal = [] # Ideal AWGN Baseline

    # 1. Baseband transmission (Generate once per SNR loop for speed)
    tx_indices = np.random.randint(0, M, N_sim)
    tx_symbols = constellation[tx_indices]

    # Generate Phase Noise profile once
    phase_noise = np.random.normal(0, sigma_phase_noise, N_sim)
    phase_noise = np.mod(phase_noise + np.pi, 2*np.pi) - np.pi

    for snr in sim_snr_dB:
        # --- SCENARIO A: AWGN + Phase Noise ---
        rx_symbols_pn = tx_symbols * np.exp(1j * phase_noise)
        rx_symbols_pn = apply_awgn(rx_symbols_pn, snr)

        # Detector for Scenario A
        dist_pn = np.abs(rx_symbols_pn[:, None] - constellation[None, :])
        rx_indices_pn = np.argmin(dist_pn, axis=1)
        sep_pn = np.sum(rx_indices_pn != tx_indices) / N_sim
        sep_empirical_pn.append(sep_pn)

        # --- SCENARIO B: Pure AWGN (Ideal Baseline) ---
        rx_symbols_ideal = apply_awgn(tx_symbols, snr)

        # Detector for Scenario B
        dist_ideal = np.abs(rx_symbols_ideal[:, None] - constellation[None, :])
        rx_indices_ideal = np.argmin(dist_ideal, axis=1)
        sep_ideal = np.sum(rx_indices_ideal != tx_indices) / N_sim
        sep_empirical_ideal.append(sep_ideal)

    # Plot empirical results (Solid line for Phase Noise)
    plt.semilogy(sim_snr_dB, sep_empirical_pn, 'o-', color=colors[idx], linewidth=2.5,
                 label=fr'{M}-QAM (Phase Noise $\sigma$={sigma_phase_noise})')

    # Plot baseline results (Dashed line for Ideal AWGN)
    plt.semilogy(sim_snr_dB, sep_empirical_ideal, '^--', color=colors[idx], linewidth=1.5, alpha=0.6,
                 label=f'{M}-QAM (Ideal AWGN)')

# Format the final plot to academic standards
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.xlabel('SNR (dB)', fontsize=12, fontweight='bold')
plt.ylabel('Symbol Error Probability (SEP)', fontsize=12, fontweight='bold')
plt.title('QAM SEP vs SNR: Ideal AWGN vs Wrapped Gaussian Phase Noise', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylim([1e-4, 1])
plt.xlim([0, 24])

# Save the figure in high resolution for the report
plt.tight_layout()
output_filename = 'SEP_vs_SNR_PhaseNoise.png'
plt.savefig(output_filename, dpi=300)

# Clean success message with absolute local path
print(f"\nSUCCESS! Plot saved locally at:")
print(f" -> {os.path.abspath(output_filename)}")

# Display the plot at the end
plt.show()