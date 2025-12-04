import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Dirac Pulse Filter Response",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .filter-box {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<h1 class="main-header">Filter Impulse and Frequency Response</h1>', unsafe_allow_html=True)
#st.markdown("**Interactive demonstration of impulse responses through different filters**")

# Sidebar for filter selection and parameters
st.sidebar.header("Filter Controls")

filter_type = st.sidebar.selectbox(
    "Select Filter Type",
    ["Identity Filter", "Moving Average", "IIR Filter (Exponential)"],
    index=0
)

# Initialize session state for filter parameters
if 'window_size' not in st.session_state:
    st.session_state.window_size = 5
if 'alpha' not in st.session_state:
    st.session_state.alpha = 0.1

# Filter parameters
if filter_type == "Moving Average":
    st.session_state.window_size = st.sidebar.slider("Window Size", 3, 31, st.session_state.window_size)
    window_size = st.session_state.window_size
elif filter_type == "IIR Filter (Exponential)":
    st.session_state.alpha = st.sidebar.slider("Alpha (decay factor)", 0.01, 0.3, st.session_state.alpha, 0.01)
    alpha = st.session_state.alpha

# Set variables for use in frequency response (use stored values)
window_size = st.session_state.window_size
alpha = st.session_state.alpha

# Signal parameters
st.sidebar.subheader("Signal Parameters")
n_samples = st.sidebar.slider("Number of samples", 30, 100, 50)
impulse_position = st.sidebar.slider("Impulse position", 0, 5, 2)

# Sine wave parameters for additional demo
st.sidebar.subheader("Sine Wave Demo")
sine_freq = st.sidebar.slider("Sine frequency", 0.01, 0.25, 0.02, 0.01)
noise_level = st.sidebar.slider("Noise level", 0.0, 0.5, 0.2, 0.05)

# Generate signals
n = np.arange(n_samples)
dirac_pulse = np.zeros(n_samples)
dirac_pulse[impulse_position] = 1.0

# Generate noisy sine wave
np.random.seed(42)  # For reproducible noise
sine_wave = np.sin(2 * np.pi * sine_freq * n)
noise = np.random.normal(0, noise_level, n_samples)
noisy_sine = sine_wave + noise

# Filter implementations
@st.cache_data
def identity_filter(input_signal):
    return input_signal.copy()

@st.cache_data
def moving_average_filter(input_signal, window_size):
    """Moving average with constant response (rectangular window)"""
    output = np.zeros_like(input_signal)
    for i in range(len(input_signal)):
        if input_signal[i] != 0:  # Only process when there's an impulse
            # Create constant response for window_size samples
            for j in range(window_size):
                if i + j < len(output):
                    output[i + j] = 1.0 / window_size  # Normalized constant value
    return output

@st.cache_data
def moving_average_filter_continuous(input_signal, window_size):
    """Standard moving average for continuous signals"""
    output = np.zeros_like(input_signal, dtype=float)
    for i in range(len(input_signal)):
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        output[i] = np.mean(input_signal[start_idx:end_idx])
    return output

@st.cache_data
def iir_filter(input_signal, alpha):
    """IIR filter with exponential decay, normalized to preserve energy"""
    output = np.zeros_like(input_signal, dtype=float)
    for i in range(len(input_signal)):
        if i == 0:
            output[i] = alpha * input_signal[i]
        else:
            output[i] = alpha * input_signal[i] + (1 - alpha) * output[i-1]
    return output

# Apply selected filter to both signals
if filter_type == "Identity Filter":
    filtered_output = identity_filter(dirac_pulse)
    filtered_sine = identity_filter(noisy_sine)
    filter_description = "Identity Filter: Output equals input (δ[n] → δ[n])"
elif filter_type == "Moving Average":
    filtered_output = moving_average_filter(dirac_pulse, window_size)
    filtered_sine = moving_average_filter_continuous(noisy_sine, window_size)
    filter_description = f"Moving Average Filter: Rectangular impulse response with width {window_size}, amplitude 1/{window_size}"
else:  # IIR Filter
    filtered_output = iir_filter(dirac_pulse, alpha)
    filtered_sine = iir_filter(noisy_sine, alpha)
    filter_description = f"IIR Filter: Exponential decay with α={alpha:.2f} (slow decay, long memory)"

# Display filter info
st.markdown(f'<div class="filter-box">{filter_description}</div>', unsafe_allow_html=True)

# Create three columns layout for impulse response
st.subheader("Impulse Response Analysis")
col1, col2, col3 = st.columns([2, 1, 2])

# Input plot (left column)
with col1:
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    ax1.stem(n[0:6], dirac_pulse[0:6], linefmt='red', markerfmt='ro', basefmt='k-', label='Input: δ[n]')
    ax1.set_title('Input: Dirac Pulse δ[n]', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Sample (n)', fontsize=10)
    ax1.set_ylabel('Amplitude', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.2)
    ax1.tick_params(labelsize=9)
    plt.tight_layout()
    st.pyplot(fig1)

# Filter box (middle column)
with col2:
    # Add arrow indicators at the TOP to align with plots
    st.markdown("""
    <div style="text-align: center; font-size: 28px; color: #ff6b6b; margin: -5px 0 10px 0; font-weight: bold;">
        → FILTER →
    </div>
    """, unsafe_allow_html=True)
    
    # Create a more prominent filter box
    filter_box_html = f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        margin: 20px 0;
        animation: pulse 2s infinite;
        min-height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
    ">
        <div style="font-size: 14px; opacity: 0.9; margin-bottom: 10px;">FILTER</div>
        <div style="font-size: 16px; line-height: 1.2;">{filter_type}</div>
    </div>
    
    <style>
    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); }}
        50% {{ transform: scale(1.02); }}
    }}
    </style>
    """
    st.markdown(filter_box_html, unsafe_allow_html=True)

# Output plot (right column)
with col3:
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    ax2.stem(n, filtered_output, linefmt='blue', markerfmt='bo', basefmt='k-', label='Filtered Output')
    ax2.set_title(f'{filter_type} Impulse Response', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Sample (n)', fontsize=10)
    ax2.set_ylabel('Amplitude', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=9)
    
    # Adjust y-axis for better visualization
    if filter_type == "Moving Average":
        max_val = max(filtered_output) if np.any(filtered_output) else 1.0
        ax2.set_ylim(-0.02, max_val + 0.1)
    elif filter_type == "IIR Filter (Exponential)":
        max_val = max(filtered_output) if np.any(filtered_output) else 1.0
        ax2.set_ylim(-0.02, max_val + 0.1)
    else:
        ax2.set_ylim(-0.1, 1.2)
    
    plt.tight_layout()
    st.pyplot(fig2)

# Add the new sine wave demonstration below
st.subheader("Real Signal Processing Example")
col4, col5 = st.columns(2)

# Noisy sine wave input (left)
with col4:
    fig3, ax3 = plt.subplots(1, 1, figsize=(7, 4))
    #ax3.plot(n, sine_wave, 'g--', alpha=0.6, linewidth=2, label='Clean sine wave')
    ax3.plot(n, noisy_sine, 'r-', alpha=0.8, linewidth=1.5, label='Noisy input signal')
    ax3.set_title('Input Signal: Noisy Sine Wave', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Sample (n)', fontsize=10)
    ax3.set_ylabel('Amplitude', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.tick_params(labelsize=9)
    plt.tight_layout()
    st.pyplot(fig3)

# Filtered sine wave output (right)  
with col5:
    fig4, ax4 = plt.subplots(1, 1, figsize=(7, 4))
    #ax4.plot(n, sine_wave, 'g--', alpha=0.6, linewidth=2, label='Original clean signal')
    ax4.plot(n, filtered_sine, 'b-', linewidth=2, label='Filtered output')
    ax4.set_title(f'Filtered Output: {filter_type}', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Sample (n)', fontsize=10)
    ax4.set_ylabel('Amplitude', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.tick_params(labelsize=9)
    plt.tight_layout()
    st.pyplot(fig4)

# NEW SECTION: Frequency Response
st.subheader("Frequency Response")

# Create frequency response plots
fig_freq, ((ax_mag_left, ax_mag_right), (ax_phase_left, ax_phase_right)) = plt.subplots(2, 2, figsize=(14, 8))

# Calculate frequency response for both filters
N_fft = 512
frequencies = np.linspace(0, 0.5, N_fft//2)  # Normalized frequency from 0 to 0.5 (Nyquist)

# Moving Average filter impulse response (use stored window_size)
ma_impulse_response = np.ones(window_size) / window_size

# IIR Filter impulse response (use stored alpha)
delta_input = np.zeros(50)
delta_input[0] = 1.0
iir_impulse_response = iir_filter(delta_input, alpha)[:20]

# Compute frequency responses
ma_fft = np.fft.fft(ma_impulse_response, N_fft)
ma_magnitude = np.abs(ma_fft)[:N_fft//2]
ma_phase = np.angle(ma_fft)[:N_fft//2]

iir_fft = np.fft.fft(iir_impulse_response, N_fft)
iir_magnitude = np.abs(iir_fft)[:N_fft//2]
iir_phase = np.angle(iir_fft)[:N_fft//2]

# Top row: Magnitude responses
ax_mag_left.plot(frequencies, 20*np.log10(ma_magnitude + 1e-10), 'b-', linewidth=2)
ax_mag_left.set_title(f"Moving Average Filter\nFrequency Response - Magnitude (Window={window_size})", fontsize=12, fontweight='bold')
ax_mag_left.set_xlabel("Normalized Frequency", fontsize=10)
ax_mag_left.set_ylabel("Magnitude (dB)", fontsize=10)
ax_mag_left.grid(True, alpha=0.3)
ax_mag_left.set_xlim(0, 0.5)

ax_mag_right.plot(frequencies, 20*np.log10(iir_magnitude + 1e-10), 'r-', linewidth=2)
ax_mag_right.set_title(f"IIR (Exponential) Filter\nFreqeuncy Response - Magnitude (α={alpha:.2f})", fontsize=12, fontweight='bold')
ax_mag_right.set_xlabel("Normalized Frequency", fontsize=10)
ax_mag_right.set_ylabel("Magnitude (dB)", fontsize=10)
ax_mag_right.grid(True, alpha=0.3)
ax_mag_right.set_xlim(0, 0.5)

# Bottom row: Phase responses
ax_phase_left.plot(frequencies, np.unwrap(ma_phase), 'b-', linewidth=2)
ax_phase_left.set_title(f"Moving Average Filter\nFrequency Response - Phase", fontsize=12, fontweight='bold')
ax_phase_left.set_xlabel("Normalized Frequency", fontsize=10)
ax_phase_left.set_ylabel("Phase (radians)", fontsize=10)
ax_phase_left.grid(True, alpha=0.3)
ax_phase_left.set_xlim(0, 0.5)

ax_phase_right.plot(frequencies, np.unwrap(iir_phase), 'r-', linewidth=2)
ax_phase_right.set_title(f"IIR (Exponential) Filter\nFreqeuncy Response - Phase", fontsize=12, fontweight='bold')
ax_phase_right.set_xlabel("Normalized Frequency", fontsize=10)
ax_phase_right.set_ylabel("Phase (radians)", fontsize=10)
ax_phase_right.grid(True, alpha=0.3)
ax_phase_right.set_xlim(0, 0.5)

plt.tight_layout()
st.pyplot(fig_freq)

# Signal flow diagram
st.subheader("Signal Flow Diagram")
col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 1, 2])

with col1:
    st.metric("Input Signal", "Dirac Pulse", "δ[n]")
    
with col2:
    st.markdown("**→**")
    
with col3:
    st.metric("Filter Type", filter_type.split()[0], "Processing")
    
with col4:
    st.markdown("**→**")
    
with col5:
    max_output = np.max(filtered_output)
    st.metric("Output Peak", f"{max_output:.3f}", f"{max_output/1.0:.1%} of input")

# Performance analysis for sine wave
st.subheader("Filter Response")
col1, col2, col3 = st.columns(3)

with col1:
    # Calculate noise reduction
    input_noise_power = np.var(noisy_sine - sine_wave)
    output_noise_power = np.var(filtered_sine - sine_wave)
    noise_reduction_db = 10 * np.log10(input_noise_power / (output_noise_power + 1e-10))
    st.metric("Noise Reduction", f"{noise_reduction_db:.1f} dB", "Lower is better")

with col2:
    # Calculate signal preservation
    signal_correlation = np.corrcoef(sine_wave, filtered_sine)[0, 1]
    st.metric("Signal Preservation", f"{signal_correlation:.3f}", "Correlation with original")

with col3:
    # Calculate delay/distortion
    mse = np.mean((sine_wave - filtered_sine)**2)
    st.metric("Signal Distortion", f"{mse:.4f}", "Mean squared error")

# Additional analysis
st.subheader("Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Impulse Response Properties")
    impulse_df = pd.DataFrame({
        'Sample': n[:15],
        'Impulse Input': dirac_pulse[:15],
        'Filter Response': filtered_output[:15]
    })
    st.dataframe(impulse_df)

with col2:
    st.subheader("Signal Processing Results")
    signal_df = pd.DataFrame({
        'Sample': n[:15],
        'Noisy Input': noisy_sine[:15],
        'Filtered Output': filtered_sine[:15]
    })
    st.dataframe(signal_df)

# Mathematical explanation
with st.expander("Mathematical Background"):
    if filter_type == "Identity Filter":
        st.latex(r"y[n] = x[n]")
        st.write("The identity filter simply passes the input unchanged.")
        
    elif filter_type == "Moving Average":
        st.latex(f"h[n] = \\begin{{cases}} \\frac{{1}}{{{window_size}}} & \\text{{for }} 0 \\leq n < {window_size} \\\\ 0 & \\text{{otherwise}} \\end{{cases}}")
        st.write(f"Impulse response: rectangular window of width {window_size} with amplitude 1/{window_size}")
        st.write("**Effect on signals:** Smooths out rapid changes and reduces high-frequency noise")
        
    else:  # IIR Filter
        st.latex(f"h[n] = {alpha:.2f} \\cdot ({1-alpha:.2f})^n \\cdot u[n]")
        st.write(f"Impulse response: exponential decay starting at {alpha:.2f}")
        st.write("**Effect on signals:** Low-pass filtering with infinite memory, smooth transitions")

# Tips
st.sidebar.markdown("---")
st.sidebar.subheader("Tips")
st.sidebar.info("""
- **Identity**: No change to signal
- **Moving Average**: Smooths signals, reduces noise
- **IIR**: Exponential smoothing, good for trends
- Adjust noise level to see filter effectiveness
- Try different frequencies to see filter response
- Check frequency response to understand filtering behavior
""")

