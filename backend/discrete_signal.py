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
