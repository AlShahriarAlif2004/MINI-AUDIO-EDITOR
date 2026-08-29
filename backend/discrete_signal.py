import numpy as np

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
        other = other.resample(self.sample_rate)
    
        result = None
    
        for i, value in enumerate(other.samples):
            global_index = other.start_index + i
    
            term = self.shift(global_index).vertical_scale(value)
    
            if result is None:
                result = term
            else:
                result = result.add(term)
    
        return result

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

    def fft(self):
        """
        Placeholder FFT wrapper — delegates to NumPy for now. Kept as a
        separate method so it can later be swapped for a from-scratch
        implementation without touching any of its callers.
        """
        return np.fft.fft(self.samples)

    def remove_noise(self, noise_start_index, noise_end_index, frame_size=1024):
        """
        Spectral-subtraction noise removal.

        The samples in [noise_start_index, noise_end_index] are treated
        as a representative noise sample: the average magnitude
        spectrum over that range becomes the "noise profile" that gets
        subtracted from every frame of the signal (original phase is
        always kept).

        Framing uses a Hann window at 50% overlap (hop = frame_size // 2),
        which satisfies the constant-overlap-add condition, so the
        overlap-add reconstruction needs no extra normalization.
        """
        if noise_start_index < self.start_index or noise_end_index > self.end_index():
            raise ValueError("Invalid noise range in remove_noise/Discrete_Signal.")

        if noise_start_index > noise_end_index:
            raise ValueError("Invalid noise range in remove_noise/Discrete_Signal.")

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

        local_noise_start = noise_start_index - self.start_index
        local_noise_end = noise_end_index - self.start_index

        # --- Build the noise profile from frames overlapping the range ---
        noise_spectra = []
        frame_start = 0
        while frame_start + frame_size <= padded_len:
            frame_end = frame_start + frame_size
            if frame_end > local_noise_start and frame_start <= local_noise_end:
                spectrum = Discrete_Signal(framed(frame_start), self.sample_rate).fft()
                noise_spectra.append(np.abs(spectrum))
            frame_start += hop_size

        if not noise_spectra:
            # Range shorter than one frame — sample a single frame around it.
            anchor = max(0, min(padded_len - frame_size, local_noise_start))
            spectrum = Discrete_Signal(framed(anchor), self.sample_rate).fft()
            noise_spectra.append(np.abs(spectrum))

        noise_profile = np.mean(noise_spectra, axis=0)

        # --- Spectral subtraction with overlap-add reconstruction ---
        output = np.zeros(padded_len)
        frame_start = 0
        while frame_start + frame_size <= padded_len:
            frame = framed(frame_start)
            spectrum = Discrete_Signal(frame, self.sample_rate).fft()

            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)

            cleaned_magnitude = np.maximum(magnitude - noise_profile, 0.0)
            cleaned_spectrum = cleaned_magnitude * np.exp(1j * phase)

            cleaned_frame = np.fft.ifft(cleaned_spectrum).real
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