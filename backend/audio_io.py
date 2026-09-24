import numpy as np
import wave
from abc import ABC, abstractmethod
from backend.discrete_signal import Discrete_Signal
from backend.audio_clip import AudioClip
import soundfile as sf

class AudioClipIO(ABC):
    @staticmethod
    @abstractmethod
    def load(path) -> "AudioClip":
        ...

    @staticmethod
    @abstractmethod
    def unload(clip: "AudioClip", path):
        ...

class WavIO(AudioClipIO):

    _SUBTYPE_BY_WIDTH = {
        1: "PCM_U8",
        2: "PCM_16",
        3: "PCM_24",
        4: "PCM_32",
    }

    @staticmethod
    def load(path) -> AudioClip:
        # always_2d=True gives shape (frames, channels) even for mono
        data, sample_rate = sf.read(path, dtype="float64", always_2d=True)

        channels = []
        for c in range(data.shape[1]):
            channels.append(
                Discrete_Signal(data[:, c], sample_rate, start_index=0)
            )

        return AudioClip(channels, name=None)

    @staticmethod
    def unload(clip: AudioClip, path, sample_width=2, clip_values=True):
        """Write clip to *path* as WAV.

        clip_values=True  (default) clamps samples to ±1.0 and writes with
                          the PCM subtype matching *sample_width* — suitable
                          for user-facing export where hard clipping is the
                          expected behaviour of audio formats.

        clip_values=False  skips clamping and writes with the 32-bit IEEE
                           float subtype, which fully preserves sample values
                           outside ±1.0 — used for lossless workspace-internal
                           storage so that vertically-scaled audio survives the
                           save/reload round-trip without data loss.
        """
        stacked = np.stack([ch.samples for ch in clip.channels], axis=1)

        if clip_values:
            if sample_width not in WavIO._SUBTYPE_BY_WIDTH:
                raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")
            subtype = WavIO._SUBTYPE_BY_WIDTH[sample_width]
            stacked = np.clip(stacked, -1.0, 1.0)
        else:
            # 32-bit float WAV: no clipping, full numeric range preserved.
            subtype = "FLOAT"

        sf.write(path, stacked, clip.sample_rate, subtype=subtype)

    @staticmethod
    def unload_internal(clip: AudioClip, path):
        """Workspace-internal save: 32-bit float WAV, no clipping.
        Preserves sample values beyond ±1.0 exactly so that editing
        operations (e.g. vertical scale) survive workspace close/reopen."""
        WavIO.unload(clip, path, clip_values=False)

class MP3IO(AudioClipIO):
    """
    MP3 is lossy and not natively supported by the stdlib.
    This implementation shells out via pydub (requires ffmpeg installed).
    """

    @staticmethod
    def load(path) -> AudioClip:
        from pydub import AudioSegment

        audio = AudioSegment.from_mp3(path)
        sample_rate = audio.frame_rate
        n_channels = audio.channels

        raw = np.array(audio.get_array_of_samples())
        raw = raw.reshape(-1, n_channels)

        max_val = float(2 ** (8 * audio.sample_width - 1))
        float_data = raw.astype(np.float64) / max_val

        channels = []
        for c in range(n_channels):
            channels.append(
                Discrete_Signal(float_data[:, c], sample_rate, start_index=0)
            )

        return AudioClip(channels, name=None)

    @staticmethod
    def unload(clip: AudioClip, path, bitrate="192k"):
        from pydub import AudioSegment

        stacked = np.stack([ch.samples for ch in clip.channels], axis=1)
        stacked = np.clip(stacked, -1.0, 1.0)

        int16_data = (stacked * 32767).astype(np.int16)
        interleaved = int16_data.reshape(-1)

        audio = AudioSegment(
            interleaved.tobytes(),
            frame_rate=clip.sample_rate,
            sample_width=2,
            channels=clip.num_channels,
        )

        audio.export(path, format="mp3", bitrate=bitrate)

class OggIO(AudioClipIO):
    """
    OGG Vorbis support via soundfile (libsndfile), which handles
    OGG natively — no extra tools required.
    """

    @staticmethod
    def load(path) -> AudioClip:
        data, sample_rate = sf.read(path, dtype="float64", always_2d=True)

        channels = []
        for c in range(data.shape[1]):
            channels.append(
                Discrete_Signal(data[:, c], sample_rate, start_index=0)
            )

        return AudioClip(channels, name=None)

    @staticmethod
    def unload(clip: AudioClip, path, quality: float = 0.7):
        """quality: 0.0 (worst) to 1.0 (best), default 0.7 ≈ good balance."""
        stacked = np.stack([ch.samples for ch in clip.channels], axis=1)
        stacked = np.clip(stacked, -1.0, 1.0)

        sf.write(path, stacked, clip.sample_rate, format="ogg", subtype="vorbis")
