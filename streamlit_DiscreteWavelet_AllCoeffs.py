import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.interpolate import interp1d
import wfdb

# Page configuration
st.set_page_config(page_title="DWT Signal Analyzer (Per-level waverec)", layout="wide")
st.title("Discrete Wavelet Transform Analyzer - Visualizing Approximation and Detail Coefficients at Each Decomposition Level")

@st.cache_data
def generate_ecg():
    record = wfdb.rdrecord("100", sampto=2048, pn_dir="mitdb")  # first 2000 samples
    ecg_sig = record.p_signal[:, 0]
    ecg_fs = record.fs
    ecg_t = np.arange(len(ecg_sig)) / ecg_fs
    
    ds_factor = 2
    ecg_sig = ecg_sig[::ds_factor]
    ecg_fs= ecg_fs / ds_factor
    ecg_t= np.arange(len(ecg_sig)) / ecg_fs
    N_ds = len(ecg_sig)
    duration = N_ds / ecg_fs
    
    return ecg_sig, ecg_t, ecg_fs, duration
# Helpers

def interpolate_to_length(x, target_length):
    x = np.asarray(x)
    if x.ndim != 1:
        x = x.ravel()
    if len(x) == target_length:
        return x
    if len(x) == 1:
        return np.full(target_length, float(x[0]))
    x_old = np.linspace(0, 1, len(x))
    x_new = np.linspace(0, 1, target_length)
    f = interp1d(x_old, x, kind='linear', fill_value='extrapolate')
    return f(x_new)


def compute_spectrum(sig):
    sig = np.asarray(sig)
    n = len(sig)
    spectrum = np.fft.fft(sig)
    magnitude = np.abs(spectrum[:n//2])
    return magnitude

# Generate
ecg_sig, ecg_t, fs,duration = generate_ecg()
n_samples = len(ecg_sig)

# Sidebar
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

# Original spectrum
original_spectrum = compute_spectrum(ecg_sig)
freq_axis = np.linspace(0, fs/2, len(original_spectrum))

# Plots - original
#st.header("Original Signal - Time Domain")
fig1, ax1 = plt.subplots(figsize=(12, 3))
ax1.plot(ecg_t, ecg_sig, '-', linewidth=1.5)
ax1.set_xlabel('Time (s)')
ax1.set_title('Original ECG Signal - Time Domain')
ax1.set_ylabel('Amplitude')
ax1.set_xlim(0, duration)
ax1.grid(alpha=0.3)
st.pyplot(fig1)
plt.close()

#st.header("Original Signal - Frequency Domain")
fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.plot(freq_axis, original_spectrum, '-', linewidth=1.5)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude')
ax2.set_title('Original ECG Signal - Frequency Domain')
ax2.set_xlim(-1, fs/2)
ax2.grid(alpha=0.3)
st.pyplot(fig2)
plt.close()

# Compute full decomposition once (up to requested 'level')
coeffs_full = pywt.wavedec(ecg_sig, wavelet, level=level)
# coeffs_full: [cA_level, cD_level, ..., cD1]

# For each requested level, extract A_l and D_l correctly
for l in range(1, level + 1):
    #st.header(f"Level {l} Decomposition (A_l and D_l)")

    # To get A_l and D_l correctly, compute wavedec at level=l (gives cA_l, cD_l, ..., cD1)
    coeffs_l = pywt.wavedec(ecg_sig, wavelet, level=l)
    cA_l = coeffs_l[0]
    # cD_l (the detail corresponding to level l's scale) is at index 1 in coeffs_l
    cD_l = coeffs_l[1]

    # Reconstruct only the approximation at level l using waverec with zeros for details
    try:
        # build coeffs list with only cA_l kept
        coeffs_approx = [cA_l] + [np.zeros_like(c) for c in coeffs_l[1:]]
        approx_rec = pywt.waverec(coeffs_approx, wavelet)
    except Exception:
        approx_rec = interpolate_to_length(cA_l, n_samples)

    # Reconstruct only detail D_l by zeroing everything except coeffs_l[1]
    try:
        coeffs_detail = [np.zeros_like(cA_l)] + [cD_l] + [np.zeros_like(c) for c in coeffs_l[2:]]
        detail_rec = pywt.waverec(coeffs_detail, wavelet)
    except Exception:
        detail_rec = interpolate_to_length(cD_l, n_samples)

    # Handle edge cases: waverec may return different length -> trim or interpolate
    def ensure_full_length(x):
        x = np.asarray(x)
        if x.size == n_samples:
            return x
        if x.size > n_samples:
            return x[:n_samples]
        return interpolate_to_length(x, n_samples)

    # If coefficient arrays are single values (lowest-level), extend as constant
    if len(cA_l) == 1:
        approx_interp = np.full(n_samples, float(cA_l[0]))
    else:
        approx_interp = ensure_full_length(approx_rec)

    if len(cD_l) == 1:
        detail_interp = np.full(n_samples, float(cD_l[0]))
    else:
        detail_interp = ensure_full_length(detail_rec)

    # Quick sanity: show coefficient counts to confirm expected halving behavior
    #st.write(f"cA_l length: {len(cA_l)}, cD_l length: {len(cD_l)}")

    # Compute spectra
    approx_spectrum = compute_spectrum(approx_interp)
    detail_spectrum = compute_spectrum(detail_interp)

    # Time domain plots
    st.subheader(f"Level {l} - Time Domain")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Approximation (A{l})**: {len(cA_l)} coefficients")
        fig_at, ax_at = plt.subplots(figsize=(6, 3))
        ax_at.plot(ecg_t, approx_interp, '-', linewidth=1.5)
        ax_at.set_xlabel('Time (s)')
        ax_at.set_ylabel('Amplitude')
        ax_at.set_xlim(0, duration)
        ax_at.grid(alpha=0.3)
        st.pyplot(fig_at)
        plt.close()

    with col2:
        st.write(f"**Detail (D{l})**: {len(cD_l)} coefficients")
        fig_dt, ax_dt = plt.subplots(figsize=(6, 3))
        ax_dt.plot(ecg_t, detail_interp, '-', linewidth=1.5)
        ax_dt.set_xlabel('Time (s)')
        ax_dt.set_ylabel('Amplitude')
        ax_dt.set_xlim(0, duration)
        ax_dt.grid(alpha=0.3)
        st.pyplot(fig_dt)
        plt.close()

    # Frequency domain plots
    st.subheader(f"Level {l} - Frequency Domain")
    col3, col4 = st.columns(2)
    with col3:
        st.write("**Approximation Spectrum**")
        fig_af, ax_af = plt.subplots(figsize=(6, 3))
        ax_af.plot(np.linspace(0, fs/2, len(approx_spectrum)), approx_spectrum, '-', linewidth=1.5)
        ax_af.set_xlabel('Frequency (Hz)')
        ax_af.set_ylabel('Magnitude')
        ax_af.set_xlim(-2, fs/2)
        ax_af.grid(alpha=0.3)
        st.pyplot(fig_af)
        plt.close()

    with col4:
        st.write("**Detail Spectrum**")
        fig_df, ax_df = plt.subplots(figsize=(6, 3))
        ax_df.plot(np.linspace(0, fs/2, len(detail_spectrum)), detail_spectrum, '-', linewidth=1.5)
        ax_df.set_xlabel('Frequency (Hz)')
        ax_df.set_ylabel('Magnitude')
        ax_df.set_xlim(-2, fs/2)
        ax_df.grid(alpha=0.3)
        st.pyplot(fig_df)
        plt.close()

    st.divider()

# Sidebar info
st.sidebar.divider()
st.sidebar.header("Information")
st.sidebar.write(f"**Signal Length:** {n_samples} samples")
st.sidebar.write(f"**Duration:** {duration} seconds")
st.sidebar.write(f"**Sampling Frequency:** {fs:.1f} Hz")
st.sidebar.write(f"**Frequency Resolution:** {fs/n_samples:.3f} Hz")
st.sidebar.write(f"**Nyquist Frequency:** {fs/2:.1f} Hz")
