"""
fft_algorithms.py -- from-scratch FFT/IFFT used by the audio editor's
DSP code (Discrete_Signal.fft, remove_noise's per-frame spectra).

Never calls numpy.fft, scipy.fft, numpy.convolve, or scipy.signal.
"""

import numpy as np


def _is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def _next_power_of_two(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _bit_reversal_permutation(N):
    bits = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for i in range(bits):
        rev |= ((idx >> i) & 1) << (bits - 1 - i)
    return rev


class CooleyTukeyFFT:
    """
    Iterative radix-2 Cooley-Tukey FFT/IFFT, O(N log N).

    Only valid for lengths that are an exact power of two -- it is the
    inner engine BluesteinFFT calls after padding, not something
    Discrete_Signal should use directly (a signal length is rarely a
    power of two).
    """

    @staticmethod
    def fft(x):
        x = np.asarray(x, dtype=complex)
        N = len(x)
        if not _is_power_of_two(N):
            raise ValueError("CooleyTukeyFFT.fft requires a power-of-two length.")

        spectrum = x[_bit_reversal_permutation(N)].copy()

        for s in range(1, int(np.log2(N)) + 1):
            M = 1 << s
            half = M // 2
            twiddles = np.exp(-1j * 2 * np.pi * np.arange(half) / M)
            blocks = spectrum.reshape(-1, M)
            even = blocks[:, :half].copy()
            odd = blocks[:, half:] * twiddles
            blocks[:, :half] = even + odd
            blocks[:, half:] = even - odd

        return spectrum

    @staticmethod
    def ifft(spectrum):
        spectrum = np.asarray(spectrum, dtype=complex)
        N = len(spectrum)
        if not _is_power_of_two(N):
            raise ValueError("CooleyTukeyFFT.ifft requires a power-of-two length.")

        x = spectrum[_bit_reversal_permutation(N)].copy()

        for s in range(1, int(np.log2(N)) + 1):
            M = 1 << s
            half = M // 2
            twiddles = np.exp(1j * 2 * np.pi * np.arange(half) / M)
            blocks = x.reshape(-1, M)
            even = blocks[:, :half].copy()
            odd = blocks[:, half:] * twiddles
            blocks[:, :half] = even + odd
            blocks[:, half:] = even - odd

        return x / N


class BluesteinFFT:
    """
    Chirp-z (Bluestein) transform: O(N log N) DFT/IDFT for ANY length N,
    built on top of CooleyTukeyFFT. This is the length-preserving,
    drop-in replacement for numpy.fft.fft / numpy.fft.ifft -- this is
    what the rest of the project should call.
    """

    @staticmethod
    def _chirp(N):
        n = np.arange(N)
        return np.exp(-1j * np.pi * (n ** 2) / N)

    @staticmethod
    def fft(x):
        x = np.asarray(x, dtype=complex)
        N = len(x)
        if N <= 1:
            return x.copy()
        if _is_power_of_two(N):
            return CooleyTukeyFFT.fft(x)

        w = BluesteinFFT._chirp(N)
        M = _next_power_of_two(2 * N - 1)  # avoids circular-conv aliasing

        a = np.zeros(M, dtype=complex)
        a[:N] = x * w

        b = np.zeros(M, dtype=complex)
        b[:N] = np.conj(w)
        b[M - N + 1:] = np.conj(w[1:][::-1])  # circularly-symmetric tail

        convolved = CooleyTukeyFFT.ifft(CooleyTukeyFFT.fft(a) * CooleyTukeyFFT.fft(b))
        return convolved[:N] * w

    @staticmethod
    def ifft(spectrum):
        spectrum = np.asarray(spectrum, dtype=complex)
        N = len(spectrum)
        if N <= 1:
            return spectrum.copy()
        if _is_power_of_two(N):
            return CooleyTukeyFFT.ifft(spectrum)
        # IDFT(X) = conj(DFT(conj(X))) / N -- reuses the forward chirp-z
        # transform instead of duplicating it with flipped twiddle signs.
        return np.conj(BluesteinFFT.fft(np.conj(spectrum))) / N