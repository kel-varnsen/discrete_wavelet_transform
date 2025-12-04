import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

def generate_signal(amplitude, frequency, time):
    return amplitude * np.sin(2 * np.pi * frequency * time)

def plot_signal(time, signal, title, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, signal)
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True)
    return fig

def plot_spectrum(freq, spectrum, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freq, np.abs(spectrum))
    ax.set_title('Spectrum of Combined Signals')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True)
    return fig

st.title('Signal Analyzer')

# Time array
duration = 1.0
sampling_rate = 1000
time = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Generate and plot individual signals
signals = []
for i in range(3):
    st.subheader(f'Signal {i+1}')
    col1, col2 = st.columns([3, 1])
    
    with col2:
        amp = st.slider(f'Amplitude {i+1}', 0.0, 10.0, 5.0, 0.1)
        freq = st.slider(f'Frequency {i+1} (Hz)', 0.1, 50.0, 10.0, 0.1)
    
    signal = generate_signal(amp, freq, time)
    signals.append(signal)
    
    with col1:
        fig = plot_signal(time, signal, f'Signal {i+1}', ylim=(-10, 10))
        st.pyplot(fig)

# Plot combined signal
combined_signal = sum(signals)
st.subheader('Combined Signal')
fig_combined = plot_signal(time, combined_signal, 'Combined Signal')
st.pyplot(fig_combined)

# Compute and plot normalized spectrum
freq = np.fft.fftfreq(len(time), 1/sampling_rate)
spectrum = fft(combined_signal) / (len(time)/2)  # Normalize by dividing by N
fig_spectrum = plot_spectrum(freq[:len(freq)//2], spectrum[:len(spectrum)//2])
st.subheader('Spectrum of Combined Signals')
st.pyplot(fig_spectrum)