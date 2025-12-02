import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.interpolate import interp1d
import wfdb

# Page configuration
st.set_page_config(page_title="DWT Signal Analyzer (Per-level waverec)", layout="wide")
st.title("Discrete Wavelet Transform Analyzer - Approximation and Detail Coefficients")


st.image(
        "DWT_Implementation.png",
        #width=800,  # fixed width, won’t scale larger
        use_column_width=True,  # scales to the page/column width
        caption="DWT Implementation Diagram"
      )



# ----------------- Generate ECG -----------------
@st.cache_data
def generate_ecg():
    record = wfdb.rdrecord("100", sampto=2048, pn_dir="mitdb")
    ecg_sig = record.p_signal[:, 0]
    ecg_fs = record.fs
    ecg_t = np.arange(len(ecg_sig)) / ecg_fs

    ds_factor = 2
    ecg_sig = ecg_sig[::ds_factor]
    ecg_fs = ecg_fs / ds_factor
    ecg_t = np.arange(len(ecg_sig)) / ecg_fs
    N_ds = len(ecg_sig)
    duration = N_ds / ecg_fs
    return ecg_sig, ecg_t, ecg_fs, duration

# ----------------- Helpers -----------------
def interpolate_to_length(x, target_length):
    x = np.asarray(x).ravel()
    if len(x) == target_length:
        return x
    if len(x) == 1:
        return np.full(target_length, float(x[0]))
    x_old = np.linspace(0, 1, len(x))
    x_new = np.linspace(0, 1, target_length)
    return interp1d(x_old, x, kind='linear', fill_value='extrapolate')(x_new)

def compute_spectrum(sig):
    n = len(sig)
    spectrum = np.fft.fft(sig)
    return np.abs(spectrum[:n//2])

# ----------------- Generate -----------------
ecg_sig, ecg_t, fs, duration = generate_ecg()
n_samples = len(ecg_sig)

# ----------------- Sidebar -----------------
st.sidebar.header("Controls")
wavelet_families = {
    'Haar': 'haar',
    'Daubechies 4 (db4)': 'db4',
    'Daubechies 8 (db8)': 'db8',
    'Symlet 4 (sym4)': 'sym4',
    'Symlet 8 (sym8)': 'sym8',
    'Coiflet 1 (coif1)': 'coif1',
    'Coiflet 2 (coif2)': 'coif2'
}
selected_wavelet_name = st.sidebar.selectbox("Wavelet Family", options=list(wavelet_families.keys()), index=0)
wavelet = wavelet_families[selected_wavelet_name]
max_level = pywt.dwt_max_level(n_samples, pywt.Wavelet(wavelet))
level = st.sidebar.number_input(f"Decomposition Level (Max: {max_level})", min_value=1, max_value=max_level, value=min(3, max_level), step=1)





# ----------------- Original Signal Plots -----------------
original_spectrum = compute_spectrum(ecg_sig)
freq_axis = np.linspace(0, fs/2, len(original_spectrum))

fig1, ax1 = plt.subplots(figsize=(12, 3))
ax1.plot(ecg_t, ecg_sig, '-', linewidth=1.5)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.set_xlim(0, duration)
ax1.set_title('Original ECG Signal - Time Domain')
ax1.grid(alpha=0.3)
st.pyplot(fig1)
plt.close()

fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.plot(freq_axis, original_spectrum, '-', linewidth=1.5)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude')
ax2.set_xlim(-1, fs/2)
ax2.set_title('Original ECG Signal - Frequency Domain')
ax2.grid(alpha=0.3)
st.pyplot(fig2)
plt.close()

# ----------------- Single Wavelet Decomposition -----------------
# Compute full decomposition once (ONLY ONCE!!)
coeffs_full = pywt.wavedec(ecg_sig, wavelet, level=level)
# coeffs_full: [cA_level, cD_level, ..., cD1]
cA = coeffs_full[0]
details = coeffs_full[1:]  # list of cD_l,...,cD1

# --- Approximation reconstruction ---
approx_rec_raw = interpolate_to_length(cA, n_samples)
try:
    coeffs_approx = [cA] + [np.zeros_like(c) for c in details]
    approx_rec = pywt.waverec(coeffs_approx, wavelet)
except Exception:
    approx_rec = interpolate_to_length(cA, n_samples)
    
approx_interp = interpolate_to_length(approx_rec, n_samples)

#---Detail reconstructions---
display_order = list(range(len(details)-1, -1, -1))  

for idx, j in enumerate(display_order, start=1):
    # --- pick the correct coefficient for this level ---
    cD = details[j]
    detail_rec_raw = interpolate_to_length(cD, n_samples)
    try:
        # isolate ONLY the current detail cD based on its real index
        coeffs_detail = [np.zeros_like(cA)] + [
            d if i == j else np.zeros_like(d)
            for i, d in enumerate(details)
        ]
        detail_rec = pywt.waverec(coeffs_detail, wavelet)

    except Exception:
        # still falls back only when wavedec fails (rare)
        detail_rec = interpolate_to_length(cD, n_samples)


    detail_interp = interpolate_to_length(detail_rec, n_samples)

    st.subheader(f"Detail Level {idx}")
    colD_left, colD_right = st.columns(2)
    

# --- Plot ALL DETAILS COEFFS -----------------------------

    # --- LEFT: TIME DOMAIN ---
    with colD_left:
        st.write(f"**Detail D{idx}**: {len(cD)} coefficients")
        fig_dt, ax_dt = plt.subplots(figsize=(6, 3))
        #ax_dt.plot(ecg_t, detail_interp, '-', linewidth=1.5)
        ax_dt.plot(ecg_t, detail_rec_raw, '-', linewidth=1.5)
        ax_dt.set_xlabel('Time (s)')
        ax_dt.set_ylabel('Amplitude')
        ax_dt.set_xlim(0, duration)
        ax_dt.grid(alpha=0.3)
        st.pyplot(fig_dt)
        plt.close()

    # --- RIGHT: SPECTRUM ---
    with colD_right:
        detail_spectrum = compute_spectrum(detail_interp)
        st.write("**Detail Spectrum**")
        fig_df, ax_df = plt.subplots(figsize=(6, 3))
        ax_df.plot(np.linspace(0, fs/2, len(detail_spectrum)), detail_spectrum, '-', linewidth=1.5)
        ax_df.set_xlabel('Frequency (Hz)')
        ax_df.set_ylabel('Magnitude')
        ax_df.set_xlim(-2, fs/2)
        ax_df.grid(alpha=0.3)
        st.pyplot(fig_df)
        plt.close()

# --- Plot Approximation (Time LEFT, Spectrum RIGHT) ---
st.subheader(f"Approximation Level {idx}")

colA_left, colA_right = st.columns(2)

with colA_left:
    st.write(f"**Approximation A{idx}**: {len(cA)} coefficients")
    fig_at, ax_at = plt.subplots(figsize=(6, 3))
    #ax_at.plot(ecg_t, approx_interp, '-', linewidth=1.5)
    ax_at.plot(ecg_t, approx_rec_raw, '-', linewidth=1.5)
    ax_at.set_xlabel('Time (s)')
    ax_at.set_ylabel('Amplitude')
    ax_at.set_xlim(0, duration)
    ax_at.grid(alpha=0.3)
    st.pyplot(fig_at)
    plt.close()

with colA_right:
    approx_spectrum = compute_spectrum(approx_interp)
    st.write("**Approximation Spectrum**")
    fig_af, ax_af = plt.subplots(figsize=(6, 3))
    ax_af.plot(np.linspace(0, fs/2, len(approx_spectrum)), approx_spectrum, '-', linewidth=1.5)
    ax_af.set_xlabel('Frequency (Hz)')
    ax_af.set_ylabel('Magnitude')
    ax_af.set_xlim(-2, fs/2)
    ax_af.grid(alpha=0.3)
    st.pyplot(fig_af)
    plt.close()

# ----------------- Sidebar Info -----------------
st.sidebar.divider()
st.sidebar.header("Information")
st.sidebar.write(f"**Signal Length:** {n_samples} samples")
st.sidebar.write(f"**Wavelet Length:** {pywt.Wavelet(wavelet).dec_len} samples")

# # Prepare the figure
# fig, axes = plt.subplots(1, 2, figsize=(6, 3))
# wavelet_obj = pywt.Wavelet(wavelet)
# L = wavelet_obj.dec_len 

# # Get the scaling (phi) and wavelet (psi) functions
# (phi, psi, x) = wavelet_obj.wavefun(level=5) # level=5 for smoothness

# # Plot the  functions
# axes[0].plot(x, psi)
# axes[0].set_title('Wavelet Function (ψ)')

# axes[1].plot(x, phi)
# axes[1].set_title('Father Wavelet (φ)')
# plt.tight_layout()
# # 4. Display 
# st.sidebar.pyplot(fig)

st.sidebar.write(f"**Duration:** {np.round(duration,2)} seconds")
st.sidebar.write(f"**Sampling Frequency:** {fs:.1f} Hz")
st.sidebar.write(f"**Frequency Resolution:** {fs/n_samples:.3f} Hz")
st.sidebar.write(f"**Nyquist Frequency:** {fs/2:.1f} Hz")
