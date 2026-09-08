"""MFCC and vector-quantization based speaker similarity."""

import numpy as np

from backend.fft_algorithms import BluesteinFFT


_FRAME_LENGTH_MS = 25
_FRAME_STEP_MS = 10
_NUM_MEL_FILTERS = 26
_NUM_MFCC = 13
_CODEBOOK_SIZE = 64


def rank_sample_speakers(target_clip, sample_speakers):
    """Return sample speakers ordered by increasing VQ distortion.

    The target speaker provides the codebook. No threshold is used: this is
    a ranking operation, so lower distortion always means greater similarity.
    """
    if not sample_speakers:
        return []

    target_features = extract_features(target_clip)
    codebook = _build_codebook(target_features)
    ranked = []
    for speaker in sample_speakers:
        features = extract_features(speaker.clip, target_rate=target_clip.sample_rate)
        distortion = _quantization_distortion(features, codebook)
        ranked.append((speaker, distortion))

    ranked.sort(key=lambda result: result[1])
    return ranked


def extract_features(clip, target_rate=None):
    """Extract CMVN-normalized 39-dimensional MFCC feature vectors."""
    signal = clip.to_mono().channels[0]
    samples = signal.samples.astype(np.float64)
    sample_rate = signal.sample_rate
    if target_rate is not None and sample_rate != target_rate:
        signal = signal.resample(target_rate)
        samples = signal.samples.astype(np.float64)
        sample_rate = signal.sample_rate

    samples = _remove_silence(samples)
    if samples.size == 0:
        samples = np.zeros(max(1, int(sample_rate * 0.025)), dtype=np.float64)

    samples = np.append(samples[0], samples[1:] - 0.97 * samples[:-1])
    frame_length = max(1, round(sample_rate * _FRAME_LENGTH_MS / 1000))
    frame_step = max(1, round(sample_rate * _FRAME_STEP_MS / 1000))
    frames = _frame_signal(samples, frame_length, frame_step)
    window = np.hamming(frame_length)
    n_fft = 1 << (frame_length - 1).bit_length()

    filter_bank = _mel_filter_bank(sample_rate, n_fft)
    mfcc = []
    for frame in frames:
        windowed = np.pad(frame * window, (0, n_fft - frame_length))
        spectrum = BluesteinFFT.fft(windowed)
        power = np.abs(spectrum[: n_fft // 2 + 1]) ** 2 / n_fft
        mel_energy = np.maximum(filter_bank @ power, np.finfo(float).eps)
        mfcc.append(_dct(np.log(mel_energy))[:_NUM_MFCC])

    static = np.asarray(mfcc, dtype=np.float64)
    delta = _delta(static)
    delta_delta = _delta(delta)
    features = np.concatenate((static, delta, delta_delta), axis=1)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    return (features - mean) / np.maximum(std, 1e-8)


def _remove_silence(samples):
    if samples.size == 0:
        return samples
    frame_size = max(1, min(len(samples), round(0.025 * 16000)))
    energies = []
    for start in range(0, len(samples), frame_size):
        frame = samples[start:start + frame_size]
        energies.append(np.mean(frame * frame))
    threshold = max(np.max(energies) * 0.01, np.finfo(float).eps)
    active = []
    for start, energy in zip(range(0, len(samples), frame_size), energies):
        if energy >= threshold:
            active.extend(range(start, min(start + frame_size, len(samples))))
    return samples[active] if active else np.array([], dtype=np.float64)


def _frame_signal(samples, frame_length, frame_step):
    if len(samples) <= frame_length:
        padded = np.pad(samples, (0, frame_length - len(samples)))
        return padded[None, :]
    count = 1 + int(np.ceil((len(samples) - frame_length) / frame_step))
    padded_length = (count - 1) * frame_step + frame_length
    padded = np.pad(samples, (0, padded_length - len(samples)))
    starts = np.arange(count) * frame_step
    return np.asarray([padded[start:start + frame_length] for start in starts])


def _mel_filter_bank(sample_rate, n_fft):
    low_mel = 2595 * np.log10(1 + 0 / 700)
    high_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    points = np.linspace(low_mel, high_mel, _NUM_MEL_FILTERS + 2)
    frequencies = 700 * (10 ** (points / 2595) - 1)
    bins = np.floor((n_fft + 1) * frequencies / sample_rate).astype(int)
    filters = np.zeros((_NUM_MEL_FILTERS, n_fft // 2 + 1))
    for index in range(_NUM_MEL_FILTERS):
        left, center, right = bins[index:index + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for point in range(left, min(center, filters.shape[1])):
            filters[index, point] = (point - left) / (center - left)
        for point in range(center, min(right, filters.shape[1])):
            filters[index, point] = (right - point) / (right - center)
    return filters


def _dct(values):
    length = len(values)
    indices = np.arange(length)
    return np.asarray([
        np.sum(values * np.cos(np.pi * coefficient * (2 * indices + 1) / (2 * length)))
        for coefficient in range(length)
    ])


def _delta(features):
    if len(features) == 1:
        return np.zeros_like(features)
    padded = np.pad(features, ((1, 1), (0, 0)), mode="edge")
    return (padded[2:] - padded[:-2]) / 2


def _build_codebook(features):
    codebook_size = min(_CODEBOOK_SIZE, len(features))
    centers = features[np.linspace(0, len(features) - 1, codebook_size).astype(int)].copy()
    for _ in range(20):
        distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        updated = centers.copy()
        for index in range(codebook_size):
            members = features[labels == index]
            if len(members):
                updated[index] = members.mean(axis=0)
        if np.allclose(updated, centers):
            break
        centers = updated
    return centers


def _quantization_distortion(features, codebook):
    distances = ((features[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=2)
    return float(distances.min(axis=1).mean())
