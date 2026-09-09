import numpy as np

from backend.fft_algorithms import BluesteinFFT

class Discrete_Signal:
    """
    Represents a single-channel (mono) discrete-time audio signal.

    samples      : 1-D NumPy array of shape (N,)
    sample_rate  : samples per second
    start_index  : global index of samples[0]
    """

    def __init__(self, samples, sample_rate, start_index=0):
        samples = np.asarray(samples)

        if samples.ndim != 1:
            raise ValueError("samples must be a 1-D NumPy array.")

        self.samples = samples.copy()
        self.sample_rate = int(sample_rate)
        self.start_index = int(start_index)

    def end_index(self):
        return self.start_index + len(self.samples) - 1

    def get_index(self, seconds):
        return int(seconds * self.sample_rate)

    def get_time(self, index):
        return index / self.sample_rate

    def get_sample(self, index):
        if index < self.start_index or index > self.end_index():
            return 0

        return self.samples[index - self.start_index]

    def set_sample(self, value, index):
        if index < self.start_index or index > self.end_index():
            raise ValueError("Invalid index in set_sample/Discrete_Signal.")

        self.samples[index - self.start_index] = value

    def copy(self):
        return Discrete_Signal(
            self.samples.copy(),
            self.sample_rate,
            self.start_index
        )

    def resample(self, new_sample_rate):
        new_sample_rate = int(new_sample_rate)

        if new_sample_rate <= 0:
            raise ValueError("Invalid sample rate in resample/Discrete_Signal.")

        if new_sample_rate == self.sample_rate:
            return Discrete_Signal(
                self.samples.copy(),
                self.sample_rate,
                self.start_index
            )

        old_length = len(self.samples)
        new_length = max(
            1,
            round(old_length * new_sample_rate / self.sample_rate)
        )

        old_pos = np.arange(old_length)
        new_pos = np.linspace(0, old_length - 1, new_length)

        new_samples = np.interp(new_pos, old_pos, self.samples)

        return Discrete_Signal(
            new_samples,
            new_sample_rate,
            self.start_index
        )

    def trim(self, start_index, end_index):
        if start_index < self.start_index or end_index > self.end_index():
            raise ValueError("Invalid trimming range in trim/Discrete_Signal.")

        if start_index > end_index:
            raise ValueError("Invalid trimming range in trim/Discrete_Signal.")

        local_start = start_index - self.start_index
        local_end = end_index - self.start_index + 1

        return Discrete_Signal(
            self.samples[local_start:local_end],
            self.sample_rate,
            start_index
        )

    def concatenate(self, other, index=None):
        other = other.resample(self.sample_rate)

        if index is None:
            index = self.end_index() + 1

        if index < self.start_index or index > self.end_index() + 1:
            raise ValueError("Invalid insertion index in concatenate/Discrete_Signal.")

        local_index = index - self.start_index

        new_samples = np.concatenate(
            (
                self.samples[:local_index],
                other.samples,
                self.samples[local_index:]
            )
        )

        return Discrete_Signal(
            new_samples,
            self.sample_rate,
            self.start_index
        )

    def time_scale(self, factor):
        factor = float(factor)

        if factor <= 0:
            raise ValueError("Invalid scaling factor in time_scale/Discrete_Signal.")

        old_length = len(self.samples)
        new_length = max(1, round(old_length * factor))

        old_pos = np.arange(old_length)
        new_pos = np.linspace(0, old_length - 1, new_length)

        new_samples = np.interp(
            new_pos,
            old_pos,
            self.samples
        )

        return Discrete_Signal(
            new_samples,
            self.sample_rate,
            round(self.start_index * factor)
        )

    def vertical_scale(self, factor, start_index=None, end_index=None):
        if start_index is None:
            start_index = self.start_index

        if end_index is None:
            end_index = self.end_index()

        factor = float(factor)

        if start_index < self.start_index or end_index > self.end_index():
            raise ValueError("Invalid scaling range in vertical_scale/Discrete_Signal.")

        if start_index > end_index:
            raise ValueError(
                "start_index must not be greater than end_index in vertical_scale/Discrete_Signal."
            )

        new_samples = self.samples.copy()

        local_start = start_index - self.start_index
        local_end = end_index - self.start_index + 1

        new_samples[local_start:local_end] *= factor

        return Discrete_Signal(
            new_samples,
            self.sample_rate,
            self.start_index
        )

    def reverse(self, start_index=None, end_index=None):
        if start_index is None:
            start_index = self.start_index

        if end_index is None:
            end_index = self.end_index()

        if start_index < self.start_index or end_index > self.end_index():
            raise ValueError("Invalid reversing range in reverse/Discrete_Signal.")

        if start_index > end_index:
            raise ValueError(
                "start_index must not be greater than end_index in reverse/Discrete_Signal."
            )

        new_samples = self.samples.copy()

        local_start = start_index - self.start_index
        local_end = end_index - self.start_index + 1

        new_samples[local_start:local_end] = (
            new_samples[local_start:local_end][::-1]
        )

        return Discrete_Signal(
            new_samples,
            self.sample_rate,
            self.start_index
        )

    def reverse_time_domain(self):
        return Discrete_Signal(
            self.samples[::-1].copy(),
            self.sample_rate,
            -self.end_index()
        )

    def add(self, other, a=1, b=1):
        other = other.resample(self.sample_rate)

        new_start = min(self.start_index, other.start_index)
        new_end = max(self.end_index(), other.end_index())

        new_length = new_end - new_start + 1

        new_samples = np.zeros(
            new_length,
            dtype=self.samples.dtype
        )

        self_start = self.start_index - new_start
        self_end = self_start + len(self.samples)

        new_samples[self_start:self_end] += a * self.samples

        other_start = other.start_index - new_start
        other_end = other_start + len(other.samples)

        new_samples[other_start:other_end] += b * other.samples

        return Discrete_Signal(
            new_samples,
            self.sample_rate,
            new_start
        )

    def shift(self, k):
        return Discrete_Signal(
            self.samples.copy(),
            self.sample_rate,
            self.start_index + int(k)
        )

    def convolution(self, other):
        """
        Compute convolution using FFT (fast convolution).
        Much more efficient than naive O(n²) approach: O(n log n).
        """
        other = other.resample(self.sample_rate)

        output_size = len(self.samples) + len(other.samples) - 1
        self_padded = np.zeros(output_size, dtype=float)
        other_padded = np.zeros(output_size, dtype=float)
        self_padded[:len(self.samples)] = self.samples
        other_padded[:len(other.samples)] = other.samples

        self_fft = BluesteinFFT.fft(self_padded)
        other_fft = BluesteinFFT.fft(other_padded)
        result_samples = BluesteinFFT.ifft(self_fft * other_fft).real
        
        # The start index of the convolution result
        result_start_index = self.start_index + other.start_index
        
        return Discrete_Signal(result_samples, self.sample_rate, result_start_index)

    def fade_in(self, start_index=None, end_index=None):
        if start_index is None:
            start_index = self.start_index

        if end_index is None:
            end_index = self.end_index()

        if start_index < self.start_index or end_index > self.end_index():
            raise ValueError("Invalid fading range in fade_in/Discrete_Signal.")

        if start_index > end_index:
            raise ValueError(
                "start_index must not be greater than end_index in fade_in/Discrete_Signal."
            )

        new_samples = self.samples.copy()

        local_start = start_index - self.start_index
        local_end = end_index - self.start_index + 1

        length = local_end - local_start

        if length == 1:
            new_samples[local_start] = 0
        else:
            factors = np.linspace(0.0, 1.0, length)
            new_samples[local_start:local_end] *= factors

        return Discrete_Signal(
            new_samples,
            self.sample_rate,
            self.start_index
        )

    def remove_noise(self, noise_start_index=None, noise_end_index=None,
                      frame_size=1024, noise_reference=None):
        """
        Spectral-subtraction noise removal.

        The noise profile (average magnitude spectrum subtracted from
        every frame of the signal, original phase always kept) can come
        from either of two sources:

          - A range [noise_start_index, noise_end_index] within this
            same signal (the original behavior).
          - An external `noise_reference` Discrete_Signal — its entire
            length is used to build the profile. This lets a noise
            "fingerprint" be borrowed from a different signal/entity
            entirely, instead of a self-contained noisy stretch.

        Framing uses a Hann window at 50% overlap (hop = frame_size // 2),
        which satisfies the constant-overlap-add condition, so the
        overlap-add reconstruction needs no extra normalization.
        """
        if frame_size < 2:
            raise ValueError("Invalid frame_size in remove_noise/Discrete_Signal.")

        hop_size = frame_size // 2
        window = np.hanning(frame_size)

        samples = self.samples
        n = len(samples)

        n_frames = max(1, int(np.ceil(max(n - frame_size, 0) / hop_size)) + 1)
        padded_len = (n_frames - 1) * hop_size + frame_size
        padded = np.zeros(padded_len)
        padded[:n] = samples

        def framed(start):
            return padded[start:start + frame_size] * window

        if noise_reference is not None:
            # --- Build the noise profile from an external reference signal ---
            ref = noise_reference.resample(self.sample_rate).samples
            ref_n = len(ref)

            ref_n_frames = max(1, int(np.ceil(max(ref_n - frame_size, 0) / hop_size)) + 1)
            ref_padded_len = (ref_n_frames - 1) * hop_size + frame_size
            ref_padded = np.zeros(ref_padded_len)
            ref_padded[:ref_n] = ref

            def ref_framed(start):
                return ref_padded[start:start + frame_size] * window

            noise_spectra = []
            frame_start = 0
            while frame_start + frame_size <= ref_padded_len:
                spectrum = BluesteinFFT.fft(ref_framed(frame_start))
                noise_spectra.append(np.abs(spectrum))
                frame_start += hop_size

            if not noise_spectra:
                spectrum = BluesteinFFT.fft(ref_framed(0))
                noise_spectra.append(np.abs(spectrum))

            noise_profile = np.mean(noise_spectra, axis=0)
        else:
            if noise_start_index is None or noise_end_index is None:
                raise ValueError(
                    "remove_noise requires either a noise range or a noise_reference."
                )

            if noise_start_index < self.start_index or noise_end_index > self.end_index():
                raise ValueError("Invalid noise range in remove_noise/Discrete_Signal.")

            if noise_start_index > noise_end_index:
                raise ValueError("Invalid noise range in remove_noise/Discrete_Signal.")

            local_noise_start = noise_start_index - self.start_index
            local_noise_end = noise_end_index - self.start_index

            # --- Build the noise profile from frames overlapping the range ---
            noise_spectra = []
            frame_start = 0
            while frame_start + frame_size <= padded_len:
                frame_end = frame_start + frame_size
                if frame_end > local_noise_start and frame_start <= local_noise_end:
                    spectrum = BluesteinFFT.fft(framed(frame_start))
                    noise_spectra.append(np.abs(spectrum))
                frame_start += hop_size

            if not noise_spectra:
                # Range shorter than one frame — sample a single frame around it.
                anchor = max(0, min(padded_len - frame_size, local_noise_start))
                spectrum = BluesteinFFT.fft(framed(anchor))
                noise_spectra.append(np.abs(spectrum))

            noise_profile = np.mean(noise_spectra, axis=0)

        # --- Spectral subtraction with overlap-add reconstruction ---
        output = np.zeros(padded_len)
        frame_start = 0
        while frame_start + frame_size <= padded_len:
            frame = framed(frame_start)
            spectrum = BluesteinFFT.fft(frame)

            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)

            cleaned_magnitude = np.maximum(magnitude - noise_profile, 0.0)
            cleaned_spectrum = cleaned_magnitude * np.exp(1j * phase)

            cleaned_frame = BluesteinFFT.ifft(cleaned_spectrum).real
            output[frame_start:frame_start + frame_size] += cleaned_frame

            frame_start += hop_size

        output = output[:n]

        return Discrete_Signal(output, self.sample_rate, self.start_index)

    def fade_out(self, start_index=None, end_index=None):
        if start_index is None:
            start_index = self.start_index

        if end_index is None:
            end_index = self.end_index()

        if start_index < self.start_index or end_index > self.end_index():
            raise ValueError("Invalid fading range in fade_out/Discrete_Signal.")

        if start_index > end_index:
            raise ValueError(
                "start_index must not be greater than end_index in fade_out/Discrete_Signal."
            )

        new_samples = self.samples.copy()

        local_start = start_index - self.start_index
        local_end = end_index - self.start_index + 1

        length = local_end - local_start

        if length == 1:
            new_samples[local_start] = 0
        else:
            factors = np.linspace(1.0, 0.0, length)
            new_samples[local_start:local_end] *= factors

        return Discrete_Signal(
            new_samples,
            self.sample_rate,
            self.start_index
        )

    def echo(self, occurrence, delay, decay):
        """
        Apply echo effect using convolution with an impulse response.
        
        Parameters:
          - occurrence: Number of echo repetitions (int >= 1)
          - delay: Time between echoes in seconds (float > 0)
          - decay: Amplitude decay factor per echo (0.0 <= decay <= 1.0)
                   Represents the fraction lost; remaining amplitude = (1-decay)
        
        The impulse response h is constructed as:
          h[0] = 1.0
          h[d*sr] = (1-decay)
          h[2*d*sr] = (1-decay)^2
          ...
          h[(occurrence-1)*d*sr] = (1-decay)^(occurrence-1)
        
        where sr is the sample rate and d*sr is the delay in samples.
        
        Example: delay=5 samples, occurrence=3, decay=0.2:
          h[0] = 1.0
          h[5] = 0.8
          h[10] = 0.64
        """
        occurrence = int(occurrence)
        delay = float(delay)
        decay = float(decay)

        if occurrence < 1:
            raise ValueError("occurrence must be at least 1 in echo/Discrete_Signal.")
        if delay <= 0:
            raise ValueError("delay must be positive in echo/Discrete_Signal.")
        if decay < 0.0 or decay > 1.0:
            raise ValueError("decay must be between 0.0 and 1.0 in echo/Discrete_Signal.")

        # Convert delay from seconds to samples
        delay_samples = int(delay * self.sample_rate)
        if delay_samples < 1:
            delay_samples = 1

        # Total length of the impulse response (in samples)
        total_length = (occurrence - 1) * delay_samples + 1

        # Construct impulse response: h[0]=1, h[delay_samples]=(1-decay), etc.
        h_samples = np.zeros(total_length)
        h_samples[0] = 1.0
        remaining = 1.0 - decay  # Amplitude remaining per echo
        for i in range(1, occurrence):
            idx = i * delay_samples
            h_samples[idx] = remaining ** i

        # Create the impulse response signal and convolve
        h_signal = Discrete_Signal(h_samples, self.sample_rate, start_index=0)
        result = self.convolution(h_signal)

        return result

    def detect_echo(self, min_delay=0.08, max_delay=0.5, min_confidence=0.2):
        """Return (delay_seconds, decay, confidence) for the strongest echo-like lag.

        Echo detection is implemented in the cepstrum domain, which is much more
        robust than raw autocorrelation for single-tone or nearly periodic audio.
        A true reflected copy creates a distinct peak at the echo delay, while the
        signal's fundamental pitch usually appears at a much shorter quefrency and is
        ignored by the search window.
        """
        if self.sample_rate <= 0 or len(self.samples) < 8:
            return (0.0, 0.0, 0.0)

        samples = np.asarray(self.samples, dtype=float)
        samples = np.nan_to_num(samples)
        if np.allclose(samples, 0.0):
            return (0.0, 0.0, 0.0)

        centered = samples - np.mean(samples)
        if np.max(np.abs(centered)) < 1e-8:
            return (0.0, 0.0, 0.0)

        min_lag = max(1, int(round(min_delay * self.sample_rate)))
        max_lag = min(len(centered) // 2, int(round(max_delay * self.sample_rate)))
        if max_lag <= min_lag:
            return (0.0, 0.0, 0.0)

        window = np.hanning(len(centered))
        windowed = centered * window
        cepstrum = BluesteinFFT.cepstrum(windowed)

        # Restrict the search to positive quefrencies that correspond to the echo delay.
        cepstrum = np.maximum(cepstrum, 0.0)
        windowed_cepstrum = cepstrum[min_lag:max_lag + 1]
        if windowed_cepstrum.size == 0:
            return (0.0, 0.0, 0.0)

        best_index = int(np.argmax(windowed_cepstrum))
        best_lag = best_index + min_lag
        best_score = float(windowed_cepstrum[best_index])
        if best_score <= min_confidence:
            return (0.0, 0.0, 0.0)

        decay = float(np.clip(best_score / max(np.max(windowed_cepstrum), 1e-9), 0.0, 1.0))
        return (best_lag / self.sample_rate, decay, float(np.clip(best_score, 0.0, 1.0)))

    def extract_delayed_component(self, delay, amplitude=None):
        """Return the delayed signal component for an echo delay."""
        delay_samples = int(round(float(delay) * self.sample_rate))
        extracted = np.zeros(len(self.samples), dtype=float)
        if delay_samples <= 0 or delay_samples >= len(self.samples):
            return Discrete_Signal(extracted, self.sample_rate, self.start_index)

        source = np.asarray(self.samples, dtype=float)
        delayed = np.zeros_like(source)
        delayed[delay_samples:] = source[:-delay_samples]

        if amplitude is None:
            centered_source = source - np.mean(source)
            centered_delayed = delayed - np.mean(delayed)
            denominator = float(np.dot(centered_delayed, centered_delayed))
            amplitude = 0.0 if denominator <= 1e-12 else float(
                np.dot(centered_source, centered_delayed) / denominator
            )

        amplitude = float(np.clip(amplitude, 0.0, 1.0))
        extracted = delayed * amplitude
        return Discrete_Signal(extracted, self.sample_rate, self.start_index)
