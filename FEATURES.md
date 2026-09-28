# Mini Audio Editor — Feature Documentation

> **Detailed signal & linear-systems theory behind every DSP feature, with the exact project code that implements it.**

---

## Table of Contents

1. [Project Architecture](#1-project-architecture)
2. [Core Data Structures](#2-core-data-structures)
   - 2.1 [Discrete_Signal](#21-discrete_signal)
   - 2.2 [AudioClip](#22-audioclip)
   - 2.3 [Workspace / Entity / Folder](#23-workspace--entity--folder)
3. [FFT Engine](#3-fft-engine)
   - 3.1 [Cooley–Tukey FFT (Radix-2 DIT)](#31-cooleytukey-fft-radix-2-dit)
   - 3.2 [Bluestein (Chirp-Z) FFT](#32-bluestein-chirp-z-fft)
4. [Audio I/O](#4-audio-io)
5. [Signal Operations](#5-signal-operations)
   - 5.1 [Resampling](#51-resampling)
   - 5.2 [Trim & Extract](#52-trim--extract)
   - 5.3 [Concatenate](#53-concatenate)
   - 5.4 [Time Scale](#54-time-scale)
   - 5.5 [Vertical Scale (Amplitude Scaling)](#55-vertical-scale-amplitude-scaling)
   - 5.6 [Reverse](#56-reverse)
   - 5.7 [Shift](#57-shift)
   - 5.8 [Signal Addition (Mixing)](#58-signal-addition-mixing)
   - 5.9 [Fade In & Fade Out](#59-fade-in--fade-out)
   - 5.10 [Convolution](#510-convolution)
   - 5.11 [Echo (Add Echo)](#511-echo-add-echo)
   - 5.12 [Echo Detection & Removal (Detect Echo)](#512-echo-detection--removal-detect-echo)
   - 5.13 [Noise Removal (Spectral Subtraction)](#513-noise-removal-spectral-subtraction)
   - 5.14 [Equalizer](#514-equalizer)
   - 5.15 [Frequency Domain View](#515-frequency-domain-view)
6. [Speaker Similarity / Voice Recognition](#6-speaker-similarity--voice-recognition)
   - 6.1 [Pre-emphasis](#61-pre-emphasis)
   - 6.2 [Framing & Windowing](#62-framing--windowing)
   - 6.3 [Mel Filter Bank](#63-mel-filter-bank)
   - 6.4 [MFCC & DCT](#64-mfcc--dct)
   - 6.5 [Delta & Delta-Delta Features](#65-delta--delta-delta-features)
   - 6.6 [CMVN Normalization](#66-cmvn-normalization)
   - 6.7 [Vector Quantization Codebook & Distortion](#67-vector-quantization-codebook--distortion)
7. [Recording](#7-recording)
8. [Workspace Persistence](#8-workspace-persistence)
9. [Dependency Summary](#9-dependency-summary)

---

## 1. Project Architecture

```
MINI-AUDIO-EDITOR/
├── app.py                          ← entry point (creates QApplication + MainWindow)
├── backend/
│   ├── discrete_signal.py          ← ALL DSP logic lives here
│   ├── fft_algorithms.py           ← from-scratch Cooley-Tukey + Bluestein FFT
│   ├── audio_clip.py               ← multi-channel wrapper over Discrete_Signal
│   ├── audio_io.py                 ← WAV / MP3 / OGG read-write
│   ├── workspace_model.py          ← Workspace / Entity / Folder data model
│   └── speaker_similarity.py       ← MFCC + VQ speaker-recognition engine
└── frontend/
    ├── main_window.py              ← top-level window, mode switching
    ├── views/
    │   ├── home_page.py
    │   ├── workspace_page.py       ← full editor shell (all edit-mode operations)
    │   └── voice_recognition_page.py
    ├── widgets/
    │   ├── entity_plot.py          ← per-channel waveform plots + operation wiring
    │   ├── waveform_player.py      ← interactive waveform + playback widget
    │   ├── sidebar_tree.py         ← workspace tree view
    │   └── voice_entity_view.py    ← speaker-recognition view per entity
    └── dialogs/
        ├── recording_dialog.py     ← live microphone recording
        ├── noise_source_dialog.py  ← noise profile source picker
        ├── concatenate_dialog.py
        ├── create_entity_dialog.py
        ├── recording_preview_dialog.py
        └── ...
```

**Design philosophy:** The backend is framework-agnostic pure-NumPy DSP; the frontend is PySide6 (Qt for Python). No `numpy.fft`, `scipy.fft`, `numpy.convolve`, or `scipy.signal` are used — all spectral transforms are the project's own FFT implementation.

---

## 2. Core Data Structures

### 2.1 `Discrete_Signal`

**File:** `backend/discrete_signal.py`

#### Theory

A **discrete-time signal** is a sequence of real numbers indexed by an integer variable n:

```
x[n],  n ∈ Z
```

In the project every signal carries three pieces of information:

| Attribute | Meaning |
|---|---|
| `samples` | NumPy array of shape `(N,)` — the sample values x[n₀], x[n₀+1], … |
| `sample_rate` | fs — samples per second (Hz) |
| `start_index` | n₀ — global index of `samples[0]` |

The **Nyquist–Shannon sampling theorem** guarantees that a band-limited signal with highest frequency component `f_max < fs/2` can be perfectly reconstructed from its samples. The maximum representable frequency is the **Nyquist frequency** `fN = fs / 2`.

#### Code

```python
class Discrete_Signal:
    def __init__(self, samples, sample_rate, start_index=0):
        self.samples     = np.asarray(samples).copy()
        self.sample_rate = int(sample_rate)
        self.start_index = int(start_index)

    def end_index(self):
        return self.start_index + len(self.samples) - 1

    def get_time(self, index):
        return index / self.sample_rate          # t = n / fs

    def get_index(self, seconds):
        return int(seconds * self.sample_rate)   # n = floor(t * fs)
```

The conversion `t = n / fs` (time from sample index) and `n = floor(t · fs)` (sample index from time) are the fundamental relations between the discrete and continuous time domains.

---

### 2.2 `AudioClip`

**File:** `backend/audio_clip.py`

#### Theory

Real-world audio is almost always **multi-channel**. A stereo recording has two channels — left (L) and right (R) — that share the same timeline and sample rate. The project wraps a list of `Discrete_Signal` objects (one per channel) and enforces invariants:

- All channels must share `sample_rate`, `start_index`, and length.

**Mono-mix formula:**

```
x_mono[n] = (1/C) * sum_{c=0}^{C-1} x_c[n]
```

The simple average is perceptually correct when channels have equal loudness.

#### Code

```python
class AudioClip:
    def to_mono(self):
        stacked = np.stack([ch.samples for ch in self.channels])
        mono_samples = stacked.mean(axis=0)
        mono_signal = self.channels[0].copy()
        mono_signal.samples = mono_samples
        return AudioClip([mono_signal], name=self.name)

    def average_channel_signal(self):
        """Used by the Voice Recognition view — returns the averaged Discrete_Signal."""
        stacked = np.stack([ch.samples for ch in self.channels])
        averaged_signal = self.channels[0].copy()
        averaged_signal.samples = stacked.mean(axis=0)
        return averaged_signal

    def apply(self, method_name, *args, channel=None, **kwargs):
        """Apply a Discrete_Signal method to all channels, or one specific channel."""
        if channel is not None:
            new_channels = list(self.channels)
            new_channels[channel] = getattr(self.channels[channel], method_name)(*args, **kwargs)
            return AudioClip(new_channels, name=self.name)
        new_channels = [getattr(ch, method_name)(*args, **kwargs) for ch in self.channels]
        return AudioClip(new_channels, name=self.name)
```

---

### 2.3 `Workspace / Entity / Folder`

**File:** `backend/workspace_model.py`

The workspace is a **tree** whose nodes are either `Folder` (internal) or `Entity` (leaf). An `Entity` wraps an `AudioClip` and also holds:

- **`divisions`** — sorted list of time-boundary positions on the *overall* plot (visual markers).
- **`channel_markers`** — per-channel marker dictionaries.
- **`id`** — a UUID hex string that maps to the asset WAV file on disk.

```python
class Entity(WorkspaceItem):
    def __init__(self, name, clip, divisions=None, channel_markers=None, ...):
        self.clip            = clip
        self.divisions       = sorted(divisions) if divisions else []
        self.channel_markers = {int(k): sorted(v) for k, v in (channel_markers or {}).items()}
        self.id              = id or uuid.uuid4().hex

    def get_segments(self):
        """Segments are time-intervals between consecutive division markers."""
        start_time = self.clip.get_time(self.clip.start_index)
        end_time   = self.clip.get_time(self.clip.end_index())
        bounds = [start_time, *self.divisions, end_time]
        return list(zip(bounds[:-1], bounds[1:]))
```

---

## 3. FFT Engine

**File:** `backend/fft_algorithms.py`

> **Constraint:** the project **never** calls `numpy.fft`, `scipy.fft`, `numpy.convolve`, or `scipy.signal`. All transforms are implemented from scratch.

### 3.1 Cooley–Tukey FFT (Radix-2 DIT)

#### Theory

The **Discrete Fourier Transform (DFT)** of length N is:

```
X[k] = sum_{n=0}^{N-1}  x[n] * exp(-j*2*pi*k*n/N),    k = 0, 1, ..., N-1
```

Naïve evaluation costs O(N²). The **Cooley–Tukey Radix-2 Decimation-In-Time (DIT)** algorithm exploits the periodicity and symmetry of the twiddle factor `W_N = exp(-j*2*pi/N)` to achieve **O(N log N)**.

**Divide-and-conquer decomposition:**

```
X[k] = E[k] + W_N^k * O[k]           (DFT of even-indexed samples)
X[k + N/2] = E[k] - W_N^k * O[k]    (DFT of odd-indexed samples)
```

This is the **butterfly** operation. The algorithm requires N to be a power of two. Before the butterfly stages, the input is **bit-reversed** so that sample n moves to position `reverse_bits(n)`.

**Inverse DFT** uses conjugate twiddle factors and normalizes by 1/N:

```
x[n] = (1/N) * sum_{k=0}^{N-1}  X[k] * exp(+j*2*pi*k*n/N)
```

#### Code

```python
def _bit_reversal_permutation(N):
    bits = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for i in range(bits):
        rev |= ((idx >> i) & 1) << (bits - 1 - i)
    return rev

class CooleyTukeyFFT:
    @staticmethod
    def fft(x):
        x = np.asarray(x, dtype=complex)
        N = len(x)
        spectrum = x[_bit_reversal_permutation(N)].copy()
        for s in range(1, int(np.log2(N)) + 1):
            M    = 1 << s           # butterfly group size
            half = M // 2
            twiddles = np.exp(-1j * 2 * np.pi * np.arange(half) / M)
            blocks = spectrum.reshape(-1, M)
            even = blocks[:, :half].copy()
            odd  = blocks[:, half:] * twiddles
            blocks[:, :half] = even + odd   # upper butterfly output
            blocks[:, half:] = even - odd   # lower butterfly output
        return spectrum

    @staticmethod
    def ifft(spectrum):
        # Conjugate twiddle sign (+j); normalize by 1/N at the end
        N = len(spectrum)
        x = spectrum[_bit_reversal_permutation(N)].copy()
        for s in range(1, int(np.log2(N)) + 1):
            M    = 1 << s
            half = M // 2
            twiddles = np.exp(+1j * 2 * np.pi * np.arange(half) / M)
            blocks = x.reshape(-1, M)
            even = blocks[:, :half].copy()
            odd  = blocks[:, half:] * twiddles
            blocks[:, :half] = even + odd
            blocks[:, half:] = even - odd
        return x / N
```

---

### 3.2 Bluestein (Chirp-Z) FFT

#### Theory

Cooley–Tukey requires N = 2^m. Real audio signals have arbitrary lengths. The **Bluestein / Chirp-Z Transform** computes the N-point DFT for **any** N in O(N log N) by rewriting the DFT as a convolution.

**Key identity** (index product decomposition):

```
k*n = -(k-n)²/2  +  k²/2  +  n²/2
```

Substituting into the DFT sum and defining the chirp `W_N = exp(-j*pi/N)`:

```
a[n] = x[n] * W_N^(n²)          (chirp-modulated input)
b[n] = conj(W_N^(n²))           (convolution kernel)
X[k] = W_N^(k²) * (a ⊛ b)[k]   (⊛ = linear convolution)
```

The convolution is computed by zero-padding both to `M = 2^ceil(log2(2N-1))` and using the Cooley–Tukey FFT internally.

The IDFT reuses the forward transform via:
```
IDFT(X) = conj(DFT(conj(X))) / N
```

#### Code

```python
class BluesteinFFT:
    @staticmethod
    def _chirp(N):
        n = np.arange(N)
        return np.exp(-1j * np.pi * (n ** 2) / N)   # W_N^{n²}

    @staticmethod
    def fft(x):
        x = np.asarray(x, dtype=complex)
        N = len(x)
        if N <= 1: return x.copy()
        if _is_power_of_two(N):
            return CooleyTukeyFFT.fft(x)   # fast path — no chirp needed

        w = BluesteinFFT._chirp(N)
        M = _next_power_of_two(2 * N - 1)  # avoid circular aliasing

        a = np.zeros(M, dtype=complex)
        a[:N] = x * w

        b = np.zeros(M, dtype=complex)
        b[:N]          = np.conj(w)
        b[M - N + 1:]  = np.conj(w[1:][::-1])  # circularly-symmetric tail

        convolved = CooleyTukeyFFT.ifft(CooleyTukeyFFT.fft(a) * CooleyTukeyFFT.fft(b))
        return convolved[:N] * w

    @staticmethod
    def ifft(spectrum):
        spectrum = np.asarray(spectrum, dtype=complex)
        N = len(spectrum)
        if N <= 1: return spectrum.copy()
        if _is_power_of_two(N):
            return CooleyTukeyFFT.ifft(spectrum)
        # IDFT(X) = conj(DFT(conj(X))) / N
        return np.conj(BluesteinFFT.fft(np.conj(spectrum))) / N
```

---

## 4. Audio I/O

**File:** `backend/audio_io.py`

| Format | Read | Write | Library |
|---|---|---|---|
| WAV | `soundfile.read` (float64) | `soundfile.write` (PCM_16/24/32 or 32-bit FLOAT) | `soundfile` |
| MP3 | `pydub.AudioSegment.from_mp3` + raw int16 → float64 | `pydub.AudioSegment.export` | `pydub` + `ffmpeg` |
| OGG | `soundfile.read` (float64) | `soundfile.write` (Vorbis) | `soundfile` |

**Internal workspace storage** uses 32-bit IEEE float WAV (no clamping), so sample values beyond ±1.0 (e.g., after vertical scaling > 1.0) are preserved exactly across save/reload cycles. User-facing export always clamps to ±1.0.

```python
class WavIO:
    @staticmethod
    def unload(clip, path, sample_width=2, clip_values=True):
        stacked = np.stack([ch.samples for ch in clip.channels], axis=1)
        if clip_values:
            stacked = np.clip(stacked, -1.0, 1.0)
            subtype = {1:"PCM_U8", 2:"PCM_16", 3:"PCM_24", 4:"PCM_32"}[sample_width]
        else:
            subtype = "FLOAT"   # 32-bit IEEE float — full range preserved
        sf.write(path, stacked, clip.sample_rate, subtype=subtype)
```

---

## 5. Signal Operations

All DSP operations are methods of `Discrete_Signal` in `backend/discrete_signal.py`. Each operation returns a **new** `Discrete_Signal` (immutable style); the original is not modified.

---

### 5.1 Resampling

#### Theory

**Sample-rate conversion** changes the effective temporal resolution of a signal. Given a signal `x[n]` at rate `fs`, we want the "same" signal at rate `fs'`.

**Linear interpolation** is used: the new sample at position `p_m` (in old-sample units) is:

```
x_new[m] = x[floor(p_m)] + (p_m - floor(p_m)) * (x[ceil(p_m)] - x[floor(p_m)])

p_m = m * (N_old - 1) / (N_new - 1)

N_new = round(N_old * fs' / fs)
```

> **Note:** Linear interpolation is a first-order polynomial reconstruction filter. It is faster than sinc-interpolation (ideal Whittaker–Shannon reconstruction) but introduces mild high-frequency roll-off. It is sufficient for intra-project resampling steps (e.g., ensuring two signals share the same rate before convolution).

#### Code

```python
def resample(self, new_sample_rate):
    old_length = len(self.samples)
    new_length = max(1, round(old_length * new_sample_rate / self.sample_rate))
    old_pos = np.arange(old_length)
    new_pos = np.linspace(0, old_length - 1, new_length)
    new_samples = np.interp(new_pos, old_pos, self.samples)   # piecewise-linear
    return Discrete_Signal(new_samples, new_sample_rate, self.start_index)
```

---

### 5.2 Trim & Extract

#### Theory

**Trimming** is windowing in the time domain to a finite interval `[n_a, n_b]`:

```
x_trim[n] = x[n],   n_a <= n <= n_b
```

All samples outside `[n_a, n_b]` are discarded. The `start_index` of the result is `n_a`.

**Extract** is the same operation but keeps the extracted portion as a *new* entity while preserving the original. In the editor both work on a user-drawn selection region on the waveform.

#### Code

```python
def trim(self, start_index, end_index):
    local_start = start_index - self.start_index
    local_end   = end_index   - self.start_index + 1
    return Discrete_Signal(
        self.samples[local_start:local_end],
        self.sample_rate,
        start_index          # new start_index = trimmed start
    )
```

---

### 5.3 Concatenate

#### Theory

**Concatenation** inserts signal `y[n]` into signal `x[n]` at position `i`:

```
z[n] = x[n]             if  n0 <= n < i
       y[n - i + n0y]   if  i <= n < i + Ny
       x[n - Ny]        if  i + Ny <= n <= n0 + Nx + Ny - 1
```

If `y` has a different sample rate, it is first resampled to match `x`'s rate.

#### Code

```python
def concatenate(self, other, index=None):
    other = other.resample(self.sample_rate)
    if index is None:
        index = self.end_index() + 1   # default: append at end
    local_index = index - self.start_index
    new_samples = np.concatenate((
        self.samples[:local_index],
        other.samples,
        self.samples[local_index:]
    ))
    return Discrete_Signal(new_samples, self.sample_rate, self.start_index)
```

---

### 5.4 Time Scale

#### Theory

**Time scaling** by factor α > 0 produces a signal whose duration changes by α while the sample rate stays fixed:

- α > 1 → signal is stretched (slower playback, longer in time)
- 0 < α < 1 → signal is compressed (faster playback, shorter in time)

```
N_new = round(N_old * α)
```

The implementation uses the same linear interpolation as resampling, but changes the number of samples at the **same** sample rate (so temporal length changes, pitch also changes — like playing a tape at a different speed).

The `start_index` is also scaled: `n0_new = round(n0 * α)`.

#### Code

```python
def time_scale(self, factor):
    old_length = len(self.samples)
    new_length = max(1, round(old_length * factor))
    old_pos = np.arange(old_length)
    new_pos = np.linspace(0, old_length - 1, new_length)
    new_samples = np.interp(new_pos, old_pos, self.samples)
    return Discrete_Signal(new_samples, self.sample_rate,
                           round(self.start_index * factor))
```

---

### 5.5 Vertical Scale (Amplitude Scaling)

#### Theory

Amplitude scaling multiplies every sample in a chosen range `[n_a, n_b]` by a real constant A:

```
y[n] = A * x[n]   if  n_a <= n <= n_b
y[n] = x[n]       otherwise
```

In the frequency domain this is equivalent to multiplying all DFT bins by A (DFT is linear):
`Y[k] = A * X[k]` within the scaled window.

- A > 1 → amplifies (louder)
- 0 < A < 1 → attenuates (quieter)
- A < 0 → inverts polarity
- A = 0 → silences

> Internal storage uses 32-bit IEEE float WAV (no clamping). Clamping to ±1.0 only happens on user-facing export.

#### Code

```python
def vertical_scale(self, factor, start_index=None, end_index=None):
    new_samples = self.samples.copy()
    local_start = start_index - self.start_index
    local_end   = end_index   - self.start_index + 1
    new_samples[local_start:local_end] *= factor
    return Discrete_Signal(new_samples, self.sample_rate, self.start_index)
```

---

### 5.6 Reverse

#### Theory

**Time-reversal** of a discrete signal maps `x[n] → x[-n]`. Within a window `[n_a, n_b]`:

```
y[n] = x[n_a + n_b - n],   n_a <= n <= n_b
```

In the z-transform: `x[-n] ↔ X(z⁻¹)`.  
In the DTFT: `x[-n] ↔ X(e^{-jω})` (complex conjugate of the spectrum — **magnitude spectrum unchanged**).

`reverse_time_domain()` performs global time-reversal and adjusts `start_index` to `-n_end` so the signal occupies the same global time range.

#### Code

```python
def reverse(self, start_index=None, end_index=None):
    new_samples = self.samples.copy()
    local_start = start_index - self.start_index
    local_end   = end_index   - self.start_index + 1
    new_samples[local_start:local_end] = new_samples[local_start:local_end][::-1]
    return Discrete_Signal(new_samples, self.sample_rate, self.start_index)

def reverse_time_domain(self):
    return Discrete_Signal(
        self.samples[::-1].copy(),
        self.sample_rate,
        -self.end_index()          # maps n → -n globally
    )
```

---

### 5.7 Shift

#### Theory

A **time shift** by k samples changes only the `start_index` without touching the sample values:

```
y[n] = x[n - k]   ↔   Y(z) = z^{-k} * X(z)
```

In the DTFT: `Y(e^{jω}) = e^{-jωk} * X(e^{jω})` — linear phase shift of `-ωk` radians. The magnitude spectrum `|Y| = |X|` is unchanged.

#### Code

```python
def shift(self, k):
    return Discrete_Signal(
        self.samples.copy(),
        self.sample_rate,
        self.start_index + int(k)
    )
```

---

### 5.8 Signal Addition (Mixing)

#### Theory

Mixing two signals is **superposition** — the defining property of linear systems:

```
z[n] = a*x[n] + b*y[n],   ∀ n ∈ [min(n0x, n0y), max(nex, ney)]
```

Samples outside a signal's range are treated as zero. If `y` has a different sample rate it is resampled to match `x` first.

#### Code

```python
def add(self, other, a=1, b=1):
    other = other.resample(self.sample_rate)
    new_start  = min(self.start_index,  other.start_index)
    new_end    = max(self.end_index(),  other.end_index())
    new_samples = np.zeros(new_end - new_start + 1, dtype=self.samples.dtype)

    s_off = self.start_index  - new_start
    new_samples[s_off : s_off + len(self.samples)]  += a * self.samples

    o_off = other.start_index - new_start
    new_samples[o_off : o_off + len(other.samples)] += b * other.samples

    return Discrete_Signal(new_samples, self.sample_rate, new_start)
```

---

### 5.9 Fade In & Fade Out

#### Theory

A **fade** multiplies the signal by a time-varying **envelope** `w[n]` over the range `[n_a, n_b]`.

**Fade In** — linearly increasing gain (ramp 0 → 1):

```
w_in[n] = (n - n_a) / (n_b - n_a),   n_a <= n <= n_b
```

**Fade Out** — linearly decreasing gain (ramp 1 → 0):

```
w_out[n] = (n_b - n) / (n_b - n_a),   n_a <= n <= n_b
```

This is **amplitude modulation** by a deterministic, non-periodic envelope. In the frequency domain the result is the convolution of X(ω) with the DTFT of the ramp window.

#### Code

```python
def fade_in(self, start_index=None, end_index=None):
    new_samples = self.samples.copy()
    local_start = start_index - self.start_index
    local_end   = end_index   - self.start_index + 1
    length = local_end - local_start
    if length == 1:
        new_samples[local_start] = 0
    else:
        factors = np.linspace(0.0, 1.0, length)   # ramp 0 → 1
        new_samples[local_start:local_end] *= factors
    return Discrete_Signal(new_samples, self.sample_rate, self.start_index)

def fade_out(self, start_index=None, end_index=None):
    # identical, but ramp is linspace(1.0, 0.0, length)
    factors = np.linspace(1.0, 0.0, length)
    new_samples[local_start:local_end] *= factors
```

---

### 5.10 Convolution

#### Theory

**Linear (aperiodic) convolution** is the central operation of linear time-invariant (LTI) system theory:

```
y[n] = (x ⊛ h)[n] = sum_{k=-∞}^{∞}  x[k] * h[n - k]
```

Given `x[n]` of length M and `h[n]` of length L, the output `y[n]` has length `M + L - 1` and its start index is `n0x + n0h`.

**Frequency-domain (convolution theorem):**

```
Y[k] = X[k] * H[k]   →   y = IFFT(FFT(x) · FFT(h))
```

To avoid **circular aliasing**, both sequences are zero-padded to length `M_pad = 2^ceil(log2(M+L-1))` before applying the Cooley–Tukey FFT. This converts circular convolution (natural for FFT) into linear convolution.

**Complexity:** O((M+L) log(M+L)) versus the naïve O(M·L).

#### Code

```python
def convolution(self, other):
    other    = other.resample(self.sample_rate)
    min_size = len(self.samples) + len(other.samples) - 1
    out_size = _next_power_of_two(min_size)

    x_pad = np.zeros(out_size, dtype=float)
    h_pad = np.zeros(out_size, dtype=float)
    x_pad[:len(self.samples)]  = self.samples
    h_pad[:len(other.samples)] = other.samples

    X = CooleyTukeyFFT.fft(x_pad)
    H = CooleyTukeyFFT.fft(h_pad)
    result = CooleyTukeyFFT.ifft(X * H).real[:min_size]

    return Discrete_Signal(result, self.sample_rate,
                           self.start_index + other.start_index)
```

---

### 5.11 Echo (Add Echo)

#### Theory

An **echo** is an LTI system whose impulse response `h[n]` is a sum of scaled, delayed Dirac deltas. For `occurrence` copies with delay D samples and amplitude-decay factor ρ = 1 − decay:

```
h[n] = sum_{i=0}^{k-1}  ρ^i * δ[n - i*D],    k = occurrence
```

This is a **comb filter** with geometrically decaying taps. Its z-transform:

```
H(z) = sum_{i=0}^{k-1}  ρ^i * z^{-i*D}  =  (1 - (ρ z^{-D})^k) / (1 - ρ z^{-D})
```

The output with echo is the convolution `y = x ⊛ h`, computed via `convolution()` in O(N log N).

**Parameters:**

| Parameter | Meaning |
|---|---|
| `occurrence` | Number of impulses in h[n] (≥ 1) |
| `delay` | Time between impulses in seconds → D = floor(delay · fs) samples |
| `decay` | Fraction of amplitude lost per copy → amplitude of i-th copy = (1−decay)^i |

#### Code

```python
def echo(self, occurrence, delay, decay):
    delay_samples  = max(1, int(delay * self.sample_rate))
    total_length   = (occurrence - 1) * delay_samples + 1
    h_samples      = np.zeros(total_length)
    h_samples[0]   = 1.0
    remaining      = 1.0 - decay   # amplitude factor ρ
    for i in range(1, occurrence):
        h_samples[i * delay_samples] = remaining ** i
    h_signal = Discrete_Signal(h_samples, self.sample_rate, start_index=0)
    return self.convolution(h_signal)
```

---

### 5.12 Echo Detection & Removal (Detect Echo)

#### Theory

Given `y[n] = (x ⊛ h)[n]` with `h[n] = sum ρ^i δ[n-iD]`, **blind deconvolution** recovers `x[n]` without knowing `h[n]` directly. The algorithm uses five DSP steps:

**Step 1 — Autocorrelation via Wiener–Khinchin Theorem:**

```
R_yy[m] = IFFT(|Y[k]|²)
```

Peaks in `R_yy[m]` for m > 0 reveal candidate echo spacings D. Multiple local maxima are retained.

**Step 2 — Real Cepstrum:**

```
c[n] = IFFT(log|Y[k]|)
```

The cepstrum of an echo-corrupted signal shows peaks at `n = D, 2D, 3D, ...` because:

```
log|Y(e^{jω})| = log|X(e^{jω})| + log|H(e^{jω})|
```

and `log|H|` is periodic with period D in the cepstral domain.

**Step 3 — Least-squares fit to estimate ρ:**

From cepstral theory the m-th cepstral peak satisfies:

```
m * c[m*D] ≈ m * log(ρ^m) = m² * log ρ
```

An OLS fit of `log(m * c[mD])` vs m gives slope `log ρ`, hence `ρ = e^slope`.

**Step 4 — Occurrence count:**

The number of cepstral peaks above 3× the median baseline of `c[n]` (excluding peak windows) gives `occurrence`.

**Step 5 — Recursive cancellation (the deconvolution):**

Using the geometric-series identity for H(z):

```
1/H(z) = (1 - ρ z^{-D}) / (1 - ρ^k z^{-k*D})
```

This yields the recursive filter:

```
x[n] = y[n] - ρ*y[n-D] + ρ^k * x[n-k*D]
```

which recovers `x[n]` sample-by-sample in O(N) — independent of `occurrence`.

The recovered signal is trimmed to length `Ny - (k-1)*D` to remove trailing echo artefacts.

#### Code

```python
def detect_echo(self, max_occurrence=12, min_delay_seconds=0.005):
    y = self.samples.astype(np.float64)
    n = len(y)

    # Step 1: linear autocorrelation (zero-padded FFT)
    pad_len      = _next_power_of_two(2 * n - 1)
    y_padded     = np.zeros(pad_len)
    y_padded[:n] = y
    power        = np.abs(BluesteinFFT.fft(y_padded)) ** 2
    autocorr     = BluesteinFFT.ifft(power).real

    # Step 2: real cepstrum
    spectrum      = BluesteinFFT.fft(y)
    log_magnitude = np.log(np.maximum(np.abs(spectrum), 1e-12))
    cepstrum      = BluesteinFFT.ifft(log_magnitude).real

    # Steps 3–4: for each candidate lag D, walk peaks → slope → ρ, occurrence
    best = None
    for delay_samples in candidate_lags:
        xs, ys = [], []
        for m in range(1, max_m + 1):
            value = cepstrum[m * delay_samples]
            if value <= threshold: break
            xs.append(float(m))
            ys.append(float(np.log(m * value)))
        if not xs: continue
        xs_arr = np.asarray(xs);  ys_arr = np.asarray(ys)
        slope  = float(np.sum(xs_arr * ys_arr) / np.sum(xs_arr ** 2))   # OLS through origin
        r      = min(max(float(np.exp(slope)), 0.0), 0.999)
        score  = len(xs) + fit_quality
        if best is None or score > best[0]:
            best = (score, delay_samples, len(xs), r)

    # Step 5: recursive cancellation
    x          = np.zeros(n, dtype=np.float64)
    r_pow_k    = r ** occurrence
    far_lag    = occurrence * delay_samples
    for i in range(n):
        val = y[i]
        if i - delay_samples >= 0:  val -= r * y[i - delay_samples]
        if i - far_lag       >= 0:  val += r_pow_k * x[i - far_lag]
        x[i] = val
    base_length = max(1, n - (occurrence - 1) * delay_samples)
    recovered = Discrete_Signal(x[:base_length], self.sample_rate, self.start_index)
    return {"occurrence": occurrence, "delay": delay_samples / self.sample_rate,
            "decay": round(1.0 - r, 6), "recovered": recovered}
```

---

### 5.13 Noise Removal (Spectral Subtraction)

#### Theory

**Spectral subtraction** is a classical single-channel noise reduction technique (Boll 1979). The fundamental additive model is:

```
Y[k] ≈ X[k] + N[k]
```

where Y is the noisy signal, X is the clean speech, and N is additive noise. The noise is assumed **stationary** (constant spectrum) during a known "noise-only" segment.

**Algorithm:**

1. **Estimate noise spectrum** `|N̂[k]|` by averaging magnitude spectra of frames from the noise-only segment (or from an external reference signal).

2. **Subtract and reconstruct** — for each frame of `y[n]`:
   ```
   |X̂[k]| = max(|Y[k]| - |N̂[k]|, 0)     (half-wave rectification)
   X̂[k]  = |X̂[k]| * exp(j * angle(Y[k]))  (keep original phase)
   ```

3. **Overlap-Add (OLA) reconstruction:** The signal is divided into frames of length L with 50% overlap (hop = L/2) using a **Hann window**:
   ```
   w[n] = 0.5 * (1 - cos(2*pi*n / (L-1)))
   ```
   The Hann window at 50% overlap satisfies the **Constant Overlap-Add (COLA)** condition:
   ```
   sum_m  w[n - m*H] = 1   ∀ n     (H = L/2)
   ```
   COLA guarantees perfect reconstruction — no extra normalization needed.

**Noise profile sources (via `NoiseSourceDialog`):**
- **Existing Signal** — user draws a selection on the entity's noisy region; frames overlapping the selection build the profile.
- **Entity Channel** — an entirely separate signal/channel is used as the noise fingerprint (entire length).

#### Code

```python
def remove_noise(self, noise_start_index=None, noise_end_index=None,
                 frame_size=1024, noise_reference=None):
    hop_size = frame_size // 2
    window   = np.hanning(frame_size)   # Hann window (COLA at 50% overlap)

    # Build noise profile
    if noise_reference is not None:
        ref = noise_reference.resample(self.sample_rate).samples
        noise_spectra = [np.abs(BluesteinFFT.fft(ref_frame * window)) for ref_frame in ...]
    else:
        noise_spectra = [np.abs(BluesteinFFT.fft(framed(start))) for start in noise_frames]
    noise_profile = np.mean(noise_spectra, axis=0)

    # Spectral subtraction + OLA
    output = np.zeros(padded_len)
    frame_start = 0
    while frame_start + frame_size <= padded_len:
        frame    = framed(frame_start)                  # windowed frame
        spectrum = BluesteinFFT.fft(frame)
        magnitude = np.abs(spectrum)
        phase     = np.angle(spectrum)
        cleaned_magnitude = np.maximum(magnitude - noise_profile, 0.0)
        cleaned_spectrum  = cleaned_magnitude * np.exp(1j * phase)
        cleaned_frame     = BluesteinFFT.ifft(cleaned_spectrum).real
        output[frame_start:frame_start + frame_size] += cleaned_frame
        frame_start += hop_size
    return Discrete_Signal(output[:n], self.sample_rate, self.start_index)
```

---

### 5.14 Equalizer

#### Theory

The **DFT-domain equalizer** directly edits individual frequency bins X[k]. Each DFT bin k corresponds to frequency:

```
f_k = k * fs / N
```

Editing X[k] sets the complex amplitude (magnitude and phase) of the sinusoidal component at f_k.

**Hermitian symmetry** must be maintained to ensure a real-valued reconstructed signal:

```
X[N - k] = conj(X[k])
```

The DC bin (k=0) and Nyquist bin (k=N/2, even N only) must be purely real.

**Reconstruction:**

```
x_new[n] = IFFT(X̂)[n]
```

#### Code

```python
def equalize(self, k: int, new_xk: complex) -> "Discrete_Signal":
    spectrum = BluesteinFFT.fft(self.samples.astype(np.float64))
    n = len(spectrum)

    # Force real at DC and Nyquist to keep output signal real
    if k == 0 or (n % 2 == 0 and k == n // 2):
        new_xk = complex(new_xk.real, 0.0)

    spectrum[k] = new_xk

    # Maintain Hermitian symmetry: X[N-k] = conj(X[k])
    mirror = (n - k) % n
    if mirror != k:
        spectrum[mirror] = np.conj(new_xk)

    reconstructed = BluesteinFFT.ifft(spectrum).real
    return Discrete_Signal(reconstructed, self.sample_rate, self.start_index)
```

---

### 5.15 Frequency Domain View

The editor can toggle any channel's waveform plot into a **frequency domain view** showing the magnitude spectrum `|X[k]|` vs frequency `f_k = k * fs / N`. The DFT is computed via:

```python
def fft(self):
    """Returns (frequencies, spectrum) via Bluestein FFT (works for any N)."""
    spectrum    = BluesteinFFT.fft(self.samples)
    n           = len(spectrum)
    frequencies = np.arange(n) * (self.sample_rate / n)
    return frequencies, spectrum
```

Only the first N/2 + 1 bins are meaningful (the positive-frequency half); the negative-frequency half is the complex conjugate mirror.

---

## 6. Speaker Similarity / Voice Recognition

**File:** `backend/speaker_similarity.py`

The voice-recognition mode compares a *target speaker* entity to one or more *sample speaker* entities using **MFCC feature extraction** + **Vector Quantization (VQ) distortion**.

### 6.1 Pre-emphasis

#### Theory

Speech signals have a roughly **+6 dB/octave** high-frequency roll-off from the lip radiation factor. Pre-emphasis applies a first-order high-pass filter to compensate:

```
y[n] = x[n] - α * x[n-1],    α = 0.97
```

z-transform: `H(z) = 1 - α z⁻¹` — a single zero near z = 1 (low frequency), boosting high frequencies.

#### Code

```python
# In extract_features():
samples = np.append(samples[0], samples[1:] - 0.97 * samples[:-1])
```

---

### 6.2 Framing & Windowing

#### Theory

Speech is approximately **stationary over short intervals** (~20–30 ms). The signal is divided into overlapping frames:

- **Frame length:** 25 ms → `L = round(fs * 0.025)` samples
- **Frame step:** 10 ms → `S = round(fs * 0.010)` samples (60% overlap)

Each frame is multiplied by a **Hamming window**:

```
w_Hamming[n] = 0.54 - 0.46 * cos(2*pi*n / (L-1))
```

The Hamming window reduces **spectral leakage** — the spreading of energy from a frequency bin into adjacent bins from computing the DFT of a finite-duration segment. Its first side lobe is ~43 dB below the main lobe (vs 13 dB for rectangular window).

#### Code

```python
frame_length = max(1, round(sample_rate * _FRAME_LENGTH_MS / 1000))   # 25 ms
frame_step   = max(1, round(sample_rate * _FRAME_STEP_MS  / 1000))    # 10 ms
frames       = _frame_signal(samples, frame_length, frame_step)
window       = np.hamming(frame_length)
n_fft        = 1 << (frame_length - 1).bit_length()   # next power-of-2 >= frame_length

# Per frame:
windowed = np.pad(frame * window, (0, n_fft - frame_length))
spectrum = BluesteinFFT.fft(windowed)
```

---

### 6.3 Mel Filter Bank

#### Theory

The **Mel scale** approximates human auditory pitch perception (logarithmic, not linear):

```
m = 2595 * log10(1 + f/700)
f = 700 * (10^(m/2595) - 1)
```

A **Mel filter bank** of M = 26 triangular band-pass filters, evenly spaced on the Mel scale between 0 Hz and fs/2, is applied to the power spectrum:

```
H_i[k] = (k - l_i)/(c_i - l_i)    if  l_i <= k <= c_i
          (r_i - k)/(r_i - c_i)    if  c_i <= k <= r_i
          0                         otherwise

E_i = log( sum_k  H_i[k] * |X[k]|² / N )
```

#### Code

```python
def _mel_filter_bank(sample_rate, n_fft):
    low_mel  = 2595 * np.log10(1 + 0             / 700)
    high_mel = 2595 * np.log10(1 + (sample_rate/2) / 700)
    points   = np.linspace(low_mel, high_mel, _NUM_MEL_FILTERS + 2)  # 28 Mel points
    frequencies = 700 * (10 ** (points / 2595) - 1)                  # convert back to Hz
    bins = np.floor((n_fft + 1) * frequencies / sample_rate).astype(int)
    filters = np.zeros((_NUM_MEL_FILTERS, n_fft // 2 + 1))
    for i in range(_NUM_MEL_FILTERS):
        left, center, right = bins[i], bins[i+1], bins[i+2]
        for k in range(left, center):
            filters[i, k] = (k - left)  / (center - left)
        for k in range(center, right):
            filters[i, k] = (right - k) / (right - center)
    return filters

# Usage:
power      = np.abs(spectrum[: n_fft // 2 + 1]) ** 2 / n_fft
mel_energy = np.maximum(filter_bank @ power, np.finfo(float).eps)
```

---

### 6.4 MFCC & DCT

#### Theory

**Mel-Frequency Cepstral Coefficients (MFCCs)** are computed by the **DCT-II** applied to the log Mel energies. The DCT decorrelates the overlapping filter outputs and concentrates information into the first few coefficients:

```
c[m] = sum_{i=0}^{M-1}  log(E_i) * cos(pi * m * (2i+1) / (2M)),    m = 0, ..., M-1
```

Only the first P = 13 coefficients are kept. c[0] captures overall log energy; higher coefficients capture spectral shape.

#### Code

```python
def _dct(values):
    length  = len(values)
    indices = np.arange(length)
    return np.asarray([
        np.sum(values * np.cos(np.pi * m * (2 * indices + 1) / (2 * length)))
        for m in range(length)
    ])

# Per frame:
mfcc.append(_dct(np.log(mel_energy))[:_NUM_MFCC])   # first 13 coefficients
```

---

### 6.5 Delta & Delta-Delta Features

#### Theory

Static MFCCs only capture the **instantaneous spectral shape**. Temporal dynamics are captured by finite-difference approximations:

**Delta (Δ) — first-order temporal derivative:**

```
Δc[m, t] = (c[m, t+1] - c[m, t-1]) / 2
```

**Delta-delta (ΔΔ) — second-order temporal derivative:**

```
ΔΔc[m, t] = (Δc[m, t+1] - Δc[m, t-1]) / 2
```

Final feature vector: concatenation of static, Δ, ΔΔ → shape (T, 39).

#### Code

```python
def _delta(features):
    padded = np.pad(features, ((1, 1), (0, 0)), mode="edge")
    return (padded[2:] - padded[:-2]) / 2   # central difference

static      = np.asarray(mfcc, dtype=np.float64)        # shape (T, 13)
delta       = _delta(static)                             # shape (T, 13)
delta_delta = _delta(delta)                              # shape (T, 13)
features    = np.concatenate((static, delta, delta_delta), axis=1)  # (T, 39)
```

---

### 6.6 CMVN Normalization

#### Theory

**Cepstral Mean and Variance Normalization (CMVN)** removes channel-level biases (microphone frequency response differences) by standardizing each feature dimension to zero mean and unit variance:

```
f̂[m, t] = (f[m, t] - μ_m) / σ_m

μ_m = (1/T) * sum_t f[m,t]
σ_m = sqrt((1/T) * sum_t (f[m,t] - μ_m)²)
```

This is essential for comparing speakers recorded with different microphones.

#### Code

```python
mean     = features.mean(axis=0)
std      = features.std(axis=0)
features = (features - mean) / np.maximum(std, 1e-8)   # CMVN
```

---

### 6.7 Vector Quantization Codebook & Distortion

#### Theory

**Vector Quantization (VQ)** represents the target speaker's feature distribution with a **codebook** of K = 64 prototype vectors (cluster centers), built by the **k-means algorithm** (max 20 iterations):

1. Initialize K centers by uniform sampling of the feature matrix.
2. **Assignment:** each frame → nearest center (Euclidean distance).
3. **Update:** recompute each center as the mean of its assigned frames.
4. Repeat until convergence.

To compare a *sample speaker* to the target, compute the **VQ distortion** — average minimum squared Euclidean distance from each sample feature vector to the nearest code vector:

```
D = (1/T') * sum_{t=1}^{T'}  min_k  ||f'_t - c_k||²
```

Lower distortion → more similar to target speaker. Speakers are ranked in ascending order of distortion.

#### Code

```python
def _build_codebook(features):
    codebook_size = min(_CODEBOOK_SIZE, len(features))
    centers = features[np.linspace(0, len(features)-1, codebook_size).astype(int)].copy()
    for _ in range(20):
        distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels    = distances.argmin(axis=1)
        updated   = centers.copy()
        for i in range(codebook_size):
            members = features[labels == i]
            if len(members):
                updated[i] = members.mean(axis=0)
        if np.allclose(updated, centers): break
        centers = updated
    return centers

def _quantization_distortion(features, codebook):
    distances = ((features[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=2)
    return float(distances.min(axis=1).mean())   # average nearest-centroid distance

def rank_sample_speakers(target_clip, sample_speakers):
    target_features = extract_features(target_clip)
    codebook = _build_codebook(target_features)
    ranked = [(speaker, _quantization_distortion(
                    extract_features(speaker.clip, target_rate=target_clip.sample_rate),
                    codebook))
              for speaker in sample_speakers]
    ranked.sort(key=lambda r: r[1])   # ascending distortion = most similar first
    return ranked
```

---

## 7. Recording

**File:** `frontend/dialogs/recording_dialog.py`

The recording dialog captures mono audio from the default microphone using **PortAudio** (via `sounddevice`).

### Architecture

| Component | Role |
|---|---|
| PortAudio callback thread | Captures `float32` chunks → upcasts to `float64` → appends to `_pending_chunks` under a `threading.Lock` |
| Qt timer (33 ms ≈ 30 fps) | Drains `_pending_chunks` into `_recorded_chunks`; scrolls the display buffer |

### Display — Scrolling Strip-Chart

A fixed-length rolling window of 12 seconds. New samples enter at the right; old samples scroll off the left:

```python
self._display_buffer = np.zeros(self._display_len, dtype=np.float64)

def _scroll_display(self, new_samples):
    n = len(new_samples)
    if n >= self._display_len:
        self._display_buffer = new_samples[-self._display_len:].copy()
    else:
        self._display_buffer = np.concatenate(
            [self._display_buffer[n:], new_samples]   # shift left
        )
    self._curve.setData(self._display_times, self._display_buffer)
```

### State Machine

```
IDLE  ──Start──▶  RECORDING  ──Pause──▶  PAUSED
  ▲                   │                    │
  │                   ▼ Stop               ▼ Reset
  └────────────────ACCEPTED/CLOSED───────IDLE
```

Final audio:

```python
def recorded_samples(self):
    self._drain_pending()
    return np.concatenate(self._recorded_chunks)   # all float64 samples
```

---

## 8. Workspace Persistence

**File:** `backend/workspace_model.py`

A workspace is persisted as a **directory**:

```
<workspace_name>/
├── workspace.json          ← tree structure (folders + entity metadata)
└── assets/
    ├── <entity-id-1>.wav   ← audio as 32-bit IEEE float WAV (no clipping)
    ├── <entity-id-2>.wav
    └── ...
```

### Save

```python
def save(self):
    assets_dir = os.path.join(self.path, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    for entity in self._collect_entities():
        # 32-bit float WAV — preserves values beyond ±1.0 exactly
        WavIO.unload_internal(entity.clip, os.path.join(assets_dir, f"{entity.id}.wav"))

    # Remove orphan assets (entities deleted since last save)
    valid = {f"{e.id}.wav" for e in entities}
    for fname in os.listdir(assets_dir):
        if fname not in valid:
            os.remove(os.path.join(assets_dir, fname))

    tree = {"name": self.name, "root": self.root.to_dict()}
    json.dump(tree, open(json_path, "w"), indent=2)
```

### Load

```python
@classmethod
def load(cls, path):
    tree = json.load(open(os.path.join(path, "workspace.json")))
    workspace = cls(name=tree["name"], path=path)
    assets_dir = os.path.join(path, "assets")
    workspace.root = Folder.from_dict(tree["root"], assets_dir)
    return workspace
```

---

## 9. Dependency Summary

| Package | Version | Role |
|---|---|---|
| `numpy` | 2.5.1 | All numerical computation (arrays, math) |
| `PySide6` | 6.11.1 | Qt GUI framework |
| `pyqtgraph` | 0.14.0 | Interactive waveform plots |
| `sounddevice` | 0.5.5 | PortAudio microphone recording + playback |
| `soundfile` | — | WAV and OGG file I/O (libsndfile) |
| `pydub` | 0.25.1 | MP3 I/O (requires `ffmpeg` installed separately) |
| `audioop-lts` | 0.2.2 | Internal pydub dependency |

> The project does **not** use `scipy`, `librosa`, or any audio-specific DSP library. All FFTs, spectral operations, and feature extractors are hand-implemented using only NumPy.
