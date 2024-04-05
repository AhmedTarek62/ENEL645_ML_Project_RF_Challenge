import matplotlib.pyplot as plt
import numpy as np


def plot_complex_signal(complex_signal):
    """
    Plot the In-phase (I) and Quadrature (Q) components of a complex signal.

    Args:
    - complex_signal (array): The complex signal to be plotted.
    """
    time_axis = np.arange(0, complex_signal.shape[0])
    # Plot the In-phase (I) and Quadrature (Q) components
    plt.figure(figsize=(10, 6))

    # In-phase (I) component (Real part)
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, np.real(complex_signal), label='In-phase (I)')
    plt.title('In-phase (I) Component of Complex Signal')
    plt.xlabel('timestep')
    plt.ylabel('Amplitude')

    # Quadrature (Q) component (Imaginary part)
    plt.subplot(3, 1, 2)
    plt.plot(time_axis, np.imag(complex_signal), label='Quadrature (Q)', color='orange')
    plt.title('Quadrature (Q) Component of Complex Signal')
    plt.xlabel('timestep')
    plt.ylabel('Amplitude')

    # Envelope
    plt.subplot(3, 1, 3)
    plt.plot(time_axis, np.abs(complex_signal),  label='Envelope', color='m')
    plt.title('Envelope of Complex Signal')
    plt.xlabel('timestep')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()


def plot_spectrogram(spectrogram, frequency_axis, time_axis):
    # Plot the spectrogram as a heatmap
    plt.pcolormesh(time_axis, frequency_axis, spectrogram, shading='gouraud')
    plt.title('Spectrogram')
    plt.xlabel(r'Time $(\frac{1}{f_s})$')
    plt.ylabel('Normalized Frequency')
    plt.colorbar(label='Amplitude')
    plt.show()


def plot_constellation(signal):
    """
    Plots a scatter constellation plot for a complex signal.

    Parameters:
    signal (array-like): Complex signal (array of complex numbers).
    """
    # Extract real and imaginary parts of the signal
    real_part = signal.real
    imag_part = signal.imag

    # Create scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(real_part, imag_part, color='b', marker='o')

    # Set plot labels and title
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('Constellation Plot')

    # Show plot
    plt.grid(True)
    plt.show()
