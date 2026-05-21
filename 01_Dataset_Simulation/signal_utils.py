import numpy as np
import cv2

# ====================================================================
# 1. MODULATION GENERATORS
# ====================================================================

def ask_mod(M):
    symbols = np.arange(-(M-1), M+1, 2)
    return symbols / np.sqrt(np.mean(symbols**2))

def psk_mod(M):
    phases = np.linspace(0, 2*np.pi, M, endpoint=False)
    return np.exp(1j * phases)

def qam_mod(M):
    bits_per_symbol = int(np.log2(M))
    if bits_per_symbol % 2 == 0:
        limit = int(np.sqrt(M))
        coords = np.arange(-limit + 1, limit, 2)
        x, y = np.meshgrid(coords, coords)
        symbols = x.flatten() + 1j * y.flatten()
    else:
        if M == 32:
            coords = np.arange(-5, 6, 2)
            x, y = np.meshgrid(coords, coords)
            mask = ~((np.abs(x.flatten()) == 5) & (np.abs(y.flatten()) == 5))
            symbols = x.flatten()[mask] + 1j * y.flatten()[mask]
        elif M == 128:
            coords = np.arange(-11, 12, 2)
            x, y = np.meshgrid(coords, coords)
            mask = ~((np.abs(x.flatten()) >= 9) & (np.abs(y.flatten()) >= 9))
            symbols = x.flatten()[mask] + 1j * y.flatten()[mask]
        else:
            raise ValueError(f"Cross-QAM for M={M} is not explicitly defined.")
    return symbols / np.sqrt(np.mean(np.abs(symbols)**2))

def hqam_mod(M):
    side = int(np.sqrt(M))
    assert side * side == M
    v1 = 1.0 + 0j
    v2 = 0.0 + 1j * np.sqrt(3)
    symbols = []
    for r in range(side):
        c2 = r
        for c in range(side):
            c1 = 2 * c + (r % 2)
            symbols.append(c1 * v1 + c2 * v2)
    symbols = np.array(symbols)
    mid_re = (np.max(symbols.real) + np.min(symbols.real)) / 2.0
    mid_im = (np.max(symbols.imag) + np.min(symbols.imag)) / 2.0
    symbols -= (mid_re + 1j * mid_im)
    return symbols / np.sqrt(np.mean(np.abs(symbols)**2))

def apsk_mod(rings, radii):
    symbols_list = []
    for i, (num_points, radius) in enumerate(zip(rings, radii)):
        phases = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        if i % 2 != 0:
            phases += np.pi / num_points
        symbols_list.append(radius * np.exp(1j * phases))
    symbols = np.concatenate(symbols_list)
    return symbols / np.sqrt(np.mean(np.abs(symbols)**2))

# ====================================================================
# 2. CHANNEL IMPAIRMENT MODELS
# ====================================================================

def apply_awgn(symbols, snr_dB):
    Es = np.mean(np.abs(symbols)**2)
    No = Es / (10**(snr_dB / 10))
    noise = np.sqrt(No / 2) * np.random.randn(*symbols.shape) + 1j * np.sqrt(No / 2) * np.random.randn(*symbols.shape)
    return symbols + noise

def apply_phase_noise(symbols, severity="none"):
    kappa_mapping = {"none": None, "low": 50.0, "medium": 20.0, "high": 10.0, "extreme": 2.0}
    kappa = kappa_mapping[severity]
    if kappa is None: return symbols
    return symbols * np.exp(1j * np.random.vonmises(mu=0.0, kappa=kappa, size=symbols.shape))

def apply_iq_imbalance(symbols, severity="none"):
    imbalance_mapping = {"none": (0.0, 0.0), "low": (2.0, 3.75), "medium": (4.0, 7.5), "high": (6.0, 11.25), "extreme": (8.0, 15.0)}
    gain_db, phase_deg = imbalance_mapping[severity]
    if gain_db == 0.0 and phase_deg == 0.0: return symbols
    factor = 10**(gain_db / 20.0)
    epsilon = (factor - 1) / (factor + 1)
    delta_phi = np.deg2rad(phase_deg)
    I, Q = symbols.real, symbols.imag
    I_imb = (1 + epsilon) * I
    Q_imb = (1 - epsilon) * (Q * np.cos(delta_phi) - I * np.sin(delta_phi))
    return I_imb + 1j * Q_imb

def apply_amplitude_distortion(symbols, severity="none"):
    severity_mapping = {"none": {"p": None}, "low": {"p": 4.0, "ibo_db": 10.0}, "medium": {"p": 2.5, "ibo_db": 7.0}, "high": {"p": 1.5, "ibo_db": 5.0}, "extreme": {"p": 1.0, "ibo_db": 3.0}}
    params = severity_mapping[severity]
    if params["p"] is None: return symbols
    amplitude, phase = np.abs(symbols), np.angle(symbols)
    p_avg = np.mean(amplitude**2)
    a_sat = np.sqrt(p_avg * (10**(params["ibo_db"] / 10.0)))
    normalized_amplitude = amplitude / a_sat
    distorted_amplitude = amplitude / (1 + normalized_amplitude**(2 * params["p"]))**(1 / (2 * params["p"]))
    return distorted_amplitude * np.exp(1j * phase)

def apply_interference(symbols, severity="none"):
    overlap_mapping = {"none": 0.00, "low": 0.15, "medium": 0.45, "high": 0.70, "extreme": 1.00}
    r = overlap_mapping[severity]
    if r == 0.0: return symbols
    N = len(symbols)
    A_j = np.sqrt(np.mean(np.abs(symbols)**2))
    f_j, theta_j = np.random.uniform(0.01, 0.1), np.random.uniform(0, 2 * np.pi)
    jammer_signal = A_j * np.exp(1j * (2 * np.pi * f_j * np.arange(N) + theta_j))
    pulse_length = int(r * N)
    start_idx = np.random.randint(0, N - pulse_length + 1) if N > pulse_length else 0
    mask = np.zeros(N)
    mask[start_idx : start_idx + pulse_length] = 1
    return symbols + (jammer_signal * mask)


# ====================================================================
# 3. HIGH SPEED IMAGE RENDERING (BYPASSES MATPLOTLIB COMPLETELY)
# ====================================================================

def render_constellation_fast(symbols, img_size=384, limit=3.0):
    """
    Lightning fast direct-to-pixel rendering. Replaces Matplotlib for 100x faster generation.
    """
    img = np.zeros((img_size, img_size), dtype=np.uint8)

    # Map Re/Im values linearly to pixel indices (0 to 383)
    x_px = np.round((symbols.real + limit) / (2 * limit) * (img_size - 1))
    y_px = np.round((limit - symbols.imag) / (2 * limit) * (img_size - 1)) # Invert Y so positive imag is top

    # Clip limits to safely stay within bounds
    x_px = np.clip(x_px, 0, img_size - 1).astype(np.int32)
    y_px = np.clip(y_px, 0, img_size - 1).astype(np.int32)

    # Light up the pixels
    img[y_px, x_px] = 255

    # Add minor dilation to emulate Matplotlib's 'markersize=2' dots (make dots slightly thicker)
    img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)

    return img