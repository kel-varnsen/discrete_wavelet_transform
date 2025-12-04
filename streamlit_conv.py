# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt

# # Initialize session state
# if 'step' not in st.session_state:
#     st.session_state.step = 0
# if 'playing' not in st.session_state:
#     st.session_state.playing = False

# st.title("Step-by-Step Convolution Process")

# # Define the signal: length 11, with positive spikes (Dirac-like impulses with varying heights)
# signal = np.array([0, 3, 2, 1, 0, 4, 0, 2, 0, 5, 0, 0, 0])

# # Define the kernel: size 3, decreasing exponential (e^(-n), scaled to sum ~4)
# kernel = np.array([2.661, 0.979, 0.360])

# # Flipped kernel for visualization (convolution involves flipping the kernel)
# flipped_kernel = np.flip(kernel)

# # Compute the full convolution result
# conv_result = np.convolve(signal, kernel, mode='full')

# # Signal length N=11, kernel M=3, display output up to index 10
# N = len(signal)
# M = len(kernel)
# output_len = 13  # Display indices 0 to 10

# # Slider to select the convolution step (position n)
# st.session_state.step = st.slider("Convolution Step (position n)", 0, output_len - 1, st.session_state.step)



# # Create the figure with four subplots
# fig, axs = plt.subplots(4, 1, figsize=(6, 8))

# # First plot: Input Signal
# axs[0].stem(range(N), signal, linefmt='b-', markerfmt='bo', basefmt='gray', label='Input Signal')
# axs[0].set_title("Input Signal")
# axs[0].set_xlabel("Index")
# axs[0].set_ylabel("Value")
# axs[0].set_xlim(-0.5, N - 0.5)
# axs[0].set_ylim(0, np.max(signal) * 1.1)
# axs[0].legend()
# axs[0].grid(True)

# # Second plot: Shifted Flipped Kernel
# kernel_positions = range(st.session_state.step - M + 1, st.session_state.step + 1)
# axs[1].stem(kernel_positions, flipped_kernel, linefmt='g-', markerfmt='go', basefmt='gray', label='Flipped Kernel')
# axs[1].set_title("Flipped Kernel (Shifted)")
# axs[1].set_xlabel("Index")
# axs[1].set_ylabel("Value")
# axs[1].set_xlim(-0.5, output_len - 0.5)
# axs[1].set_ylim(0, np.max(kernel) * 1.1)
# axs[1].legend()
# axs[1].grid(True)

# # Third plot: Element-wise Products
# # Zero-pad signal for indices < 0 and >= N
# padded_signal = np.pad(signal, (M-1, M-1), mode='constant', constant_values=0)
# products = padded_signal[st.session_state.step:st.session_state.step + M] * flipped_kernel
# axs[2].stem(kernel_positions, products, linefmt='r-', markerfmt='ro', basefmt='gray', label='Product')
# axs[2].set_title(f"Element-wise Product (Sum = {np.sum(products):.2f})")
# axs[2].set_xlabel("Index")
# axs[2].set_ylabel("Value")
# axs[2].set_xlim(-0.5, output_len - 0.5)
# axs[2].set_ylim(0, np.max(products) * 1.1 if np.max(products) > 0 else 1)
# axs[2].legend()
# axs[2].grid(True)

# # Fourth plot: Convolution output
# # Plot the full convolution in light color for reference
# axs[3].stem(range(len(conv_result)), conv_result, linefmt='c-', markerfmt='co', basefmt='gray', label='Full Output (reference)')
# # Plot the output up to the current step in bold
# axs[3].stem(range(st.session_state.step + 1), conv_result[:st.session_state.step + 1], linefmt='b-', markerfmt='bo', basefmt='gray', label='Current Output')
# axs[3].set_title("Convolution Output")
# axs[3].set_xlabel("Index n")
# axs[3].set_ylabel("Value")
# axs[3].set_xlim(-0.5, output_len - 0.5)
# axs[3].set_ylim(0, np.max(conv_result) * 1.1)
# axs[3].legend()
# axs[3].grid(True)

# # Adjust layout to prevent overlapping
# plt.tight_layout()

# # Display the plot in Streamlit
# st.pyplot(fig)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'playing' not in st.session_state:
    st.session_state.playing = False

st.title("Step-by-Step Convolution Process")

# Define the signal: length 11, with positive spikes (Dirac-like impulses with varying heights)
signal = np.array([0, 3, 2, 1, 0, 4, 1, 2, 0, 5, 0, 0, 0])

# Define the kernel: size 3, decreasing exponential (e^(-n), scaled to sum ~4)
kernel = np.array([2.661, 0.979, 0.360])

# Flipped kernel for visualization (convolution involves flipping the kernel)
flipped_kernel = np.flip(kernel)

# Compute the full convolution result
conv_result = np.convolve(signal, kernel, mode='full')

# Signal length N=11, kernel M=3, display output up to index 10
N = len(signal)
M = len(kernel)
output_len = 13  # Display indices 0 to 10

# Slider to select the convolution step (position n)
st.session_state.step = st.slider("Convolution Step (position n)", 0, output_len - 1, st.session_state.step)

# Create the figure with 4 plots on the left and 2 plots on the right
fig = plt.figure(figsize=(14, 8))

# Create a grid layout: 4 rows, 3 columns
# Left side will span 2 columns, right side will use 1 column

# Left side plots (spanning first 2 columns)
ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=2)  # Input Signal
ax2 = plt.subplot2grid((4, 3), (1, 0), colspan=2)  # Shifted Flipped Kernel
ax3 = plt.subplot2grid((4, 3), (2, 0), colspan=2)  # Element-wise Products
ax4 = plt.subplot2grid((4, 3), (3, 0), colspan=2)  # Convolution Output

# Right side plots (third column)
ax5 = plt.subplot2grid((4, 3), (0, 2), rowspan=2)  # Original Kernel
ax6 = plt.subplot2grid((4, 3), (2, 2), rowspan=2)  # Kernel Comparison

axs = [ax1, ax2, ax3, ax4]

# First plot: Input Signal
axs[0].stem(range(N), signal, linefmt='b-', markerfmt='bo', basefmt='gray', label='Input Signal')
axs[0].set_title("Input Signal")
axs[0].set_xlabel("Index")
axs[0].set_ylabel("Value")
axs[0].set_xlim(-0.5, N - 0.5)
axs[0].set_ylim(0, np.max(signal) * 1.1)
axs[0].legend()
axs[0].grid(True)

# Second plot: Shifted Flipped Kernel
kernel_positions = range(st.session_state.step - M + 1, st.session_state.step + 1)
axs[1].stem(kernel_positions, flipped_kernel, linefmt='g-', markerfmt='go', basefmt='gray', label='Flipped Kernel')
axs[1].set_title("Flipped Kernel (Shifted)")
axs[1].set_xlabel("Index")
axs[1].set_ylabel("Value")
axs[1].set_xlim(-0.5, output_len - 0.5)
axs[1].set_ylim(0, np.max(kernel) * 1.1)
axs[1].legend()
axs[1].grid(True)

# Third plot: Element-wise Products
# Zero-pad signal for indices < 0 and >= N
padded_signal = np.pad(signal, (M-1, M-1), mode='constant', constant_values=0)
products = padded_signal[st.session_state.step:st.session_state.step + M] * flipped_kernel
axs[2].stem(kernel_positions, products, linefmt='r-', markerfmt='ro', basefmt='gray', label='Product')
axs[2].set_title(f"Element-wise Product (Sum = {np.sum(products):.2f})")
axs[2].set_xlabel("Index")
axs[2].set_ylabel("Value")
axs[2].set_xlim(-0.5, output_len - 0.5)
axs[2].set_ylim(0, np.max(products) * 1.1 if np.max(products) > 0 else 1)
axs[2].legend()
axs[2].grid(True)

# Fourth plot: Convolution output
# Plot the full convolution in light color for reference
axs[3].stem(range(len(conv_result)), conv_result, linefmt='c-', markerfmt='co', basefmt='gray', label='Full Output (reference)')
# Plot the output up to the current step in bold
axs[3].stem(range(st.session_state.step + 1), conv_result[:st.session_state.step + 1], linefmt='b-', markerfmt='bo', basefmt='gray', label='Current Output')
axs[3].set_title("Convolution Output")
axs[3].set_xlabel("Index n")
axs[3].set_ylabel("Value")
axs[3].set_xlim(-0.5, output_len - 0.5)
axs[3].set_ylim(0, np.max(conv_result) * 1.1)
axs[3].legend()
axs[3].grid(True)

# Fifth plot: Original Kernel (top right)
ax5.stem(range(M), kernel, linefmt='purple', markerfmt='s', basefmt='gray')
ax5.set_title("Original Kernel h[k]")
ax5.set_xlabel("Index k")
ax5.set_ylabel("Value")
ax5.set_xlim(-M-0.5, M - 0.5)
ax5.set_ylim(0, np.max(kernel) * 1.1)
ax5.grid(True)

# Add numerical values as text annotations
for i, val in enumerate(kernel):
    ax5.text(i, val + 0.1, f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

arr = np.arange(M) 
negated = -arr      
x_flip = np.flip(negated) 

# Sixth plot: Kernel Comparison (bottom right)
ax6.stem(x_flip, np.flip(kernel), linefmt='purple', markerfmt='s', basefmt='gray')
ax6.set_title("Flipped Kernel h[-k]")
ax6.set_xlabel("Index k")
ax6.set_ylabel("Value")
ax6.set_xlim(-M-0.5, M - 0.5)
ax6.set_ylim(0, np.max(kernel) * 1.1)
ax6.grid(True)




# Adjust layout to prevent overlapping
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)

# Add explanatory text
st.markdown("""
### Kernel Information
**Original Kernel**: h[k] = [2.661, 0.979, 0.360] (decreasing exponential)  
**Flipped Kernel**: h[M-1-k] = [0.360, 0.979, 2.661] (used in convolution)

The right-side plots show:
- **Top**: Original kernel values with exact numerical values
- **Bottom**: Side-by-side comparison of original vs. flipped kernel

The convolution operation uses the flipped version of the kernel, which slides across the signal.
""")