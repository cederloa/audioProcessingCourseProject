# Separation of drums from music signals
# Antti Cederlöf, 283233

import numpy as np
import sounddevice as sd
import librosa
import librosa.display
import itertools
from scipy import signal
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt


def stft(audio, t_w, g, n_fft, hop_size):
    """
    A function for calculating short time fourier transform. The power spectrogram this function returns is
    range-compressed.
    :param hop_size: The distance in samples between the start of adjacent windows
    :param n_fft: Points in fft
    :param t_w: Window size
    :param g: Range-compression factor
    :param audio: The audio signal array
    :return: The power spectrogram (y_spect) and the one-sided fourier transform spectrogram (F)
    """

    for i in itertools.count(start=0):
        w_start = int(i * hop_size)  # The time point at the start of the window
        x = audio[w_start:w_start + t_w]
        x_w = x * signal.hann(len(x))

        # Stop iterating when the signal ends
        if i * hop_size + t_w >= len(audio):
            break

        # DFT
        y = fft(x_w, n_fft)
        y1 = y[0:int(n_fft / 2)]  # Single-sided spectrum
        y1T = y1.reshape(len(y1), 1)  # Transpose
        yp = np.power(abs(y1), 2 * g)  # Range-compressed power spectrum
        ypT = yp.reshape(len(yp), 1)  # Transpose

        if i == 0:
            # Initialize spectrogram
            y_spect = ypT
            F = y1T
        else:
            y_spect = np.concatenate((y_spect, ypT), axis=1)
            F = np.concatenate((F, y1T), axis=1)

    return y_spect, F


def istft(spect, t_w, hop_size):
    """
    A function for calculating the inverse short time fourier transform.
    :param hop_size: The distance in samples between the start of adjacent windows
    :param t_w: Window size
    :param spect: The amplitude spectrogram
    :return:
    """
    for i in range(np.shape(spect)[1]):
        # Take ifft of the signals at different time windows
        y = spect[:, i]
        x = np.real(ifft(y, t_w))
        x = x * signal.hann(len(x))

        # Put the ifft transformed signal parts together
        startpoint = int(i * hop_size)
        endpoint = startpoint + t_w
        if i == 0:
            sgn = np.zeros(int(hop_size * (np.shape(spect)[1] + (t_w/hop_size - 1))))
        sgn[startpoint:endpoint] += x

    return sgn


def divide_harmonic_percussive(spect, k_max, a):
    """
    A function for separating an audio signal spectrogram
    :param spect:
    :param k_max:
    :param a:
    :return: Harmonic and percussive signal spectrograms
    """
    # Initial matrices
    H = 1/2 * spect
    P = 1/2 * spect

    # Add zero padding to keep the same shape
    H_buffer = np.zeros((np.shape(H)[0], 1))
    P_buffer = np.zeros((1, np.shape(P)[1]))

    for k in range(k_max):
        H_delay = np.concatenate((H_buffer, H[:, :-1]), axis=1)  # H values with a delay in time
        H_1 = np.concatenate((H[:, 1:], H_buffer), axis=1)  # H values with an advancement in time
        P_delay = np.concatenate((P_buffer, P[:-1, :]), axis=0)  # P values with a delay in freq
        P_1 = np.concatenate((P[1:, :], P_buffer), axis=0)  # P values with an advancement in freq
        update = a * (H_delay - 2 * H + H_1)/4 - (1 - a) * (P_delay - 2 * P + P_1)/4

        H = np.minimum(np.maximum(H + update, 0), spect)
        P = spect - H

    # Pick every value of the spectrogram to either H_bin or P_bin, and everywhere else H_bin and P_bin are zero.
    H_bin = np.where(H - P >= 0, 1, 0)
    P_bin = np.where(H - P < 0, 1, 0)
    H_bin = np.multiply(spect, H_bin)
    P_bin = np.multiply(spect, P_bin)

    return H_bin, P_bin


def main():
    # audio, fs = librosa.load('police03short.wav')
    audio, fs = librosa.load('project_test1.wav')

    g = 0.3  # Range-compression factor
    a = 0.5  # The balance parameter used in separating the harmonic and percussive spectrograms
    n_fft = 2048  # Points in fft
    t_w = n_fft  # Window size in samples
    hop_size = t_w / 5

    # First, perform STFT
    spect, F = stft(audio, t_w, g, n_fft, hop_size)

    # Divide spectrogram into harmonic and percussive parts
    H, P = divide_harmonic_percussive(spect, 50, a)

    # Bring the spectrograms back to the complex forms
    H_c = np.power(H, 1 / (2 * g)) * np.exp(np.angle(F) * 1j)
    P_c = np.power(P, 1 / (2 * g)) * np.exp(np.angle(F) * 1j)

    # Take ISTFT to get the harmonic and percussive signals and their combination
    h = istft(H_c, t_w, hop_size)
    p = istft(P_c, t_w, hop_size)
    rec = np.add(p, h)  # reconstructed signal

    # Uncomment the rows below to listen the original signal, reconstructed signal, and harmonic and percussive
    # components separately.
    '''
    sd.play(audio, fs)
    sd.wait()
    sd.play(rec, fs)
    sd.wait()
    '''
    sd.play(h, fs)
    sd.wait()
    sd.play(p, fs)
    sd.wait()
    
    e = audio[:len(rec)] - rec[:len(audio)]  # original - reconstructed, for snr
    snr = 10 * np.log10(np.sum(np.square(audio)) / np.sum(np.square(e)))

    t_max = len(audio)/fs  # Length of the audio in s
    t_max_rec = len(rec)/fs  # Length of the reconstructed audio (slightly different)
    t = np.linspace(0, t_max, len(audio))
    t_rec = np.linspace(0, t_max_rec, len(rec))
    plt.plot(t, audio,  label='Original')
    plt.plot(t_rec, rec, label='reconstructed')
    plt.title(f'Original vs reconstructed signal. SNR={snr}')
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    plt.legend()
    plt.show()


main()
