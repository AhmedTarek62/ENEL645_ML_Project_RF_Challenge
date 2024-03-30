from scipy.signal import spectrogram, get_window
from scipy.fft import fftshift


def compute_spectrogram(signal, nfft, overlap):
    hann_window = get_window('hann', nfft)
    f, t, Sxx = spectrogram(signal, fs=1.0, window=hann_window, noverlap=overlap, nfft=nfft, return_onesided=False)
    return fftshift(f), t, fftshift(Sxx, axes=0)
