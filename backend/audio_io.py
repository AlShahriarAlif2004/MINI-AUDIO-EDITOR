import numpy as np
import wave
from abc import ABC, abstractmethod
from backend.discrete_signal import Discrete_Signal
from backend.audio_clip import AudioClip

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

    _DTYPE_BY_WIDTH = {
        1: np.uint8,   # WAV 8-bit is unsigned
        2: np.int16,
        4: np.int32,
    }

    @staticmethod
    def load(path) -> AudioClip:
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()

            raw = wf.readframes(n_frames)

        if sample_width not in WavIO._DTYPE_BY_WIDTH:
            raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

        dtype = WavIO._DTYPE_BY_WIDTH[sample_width]
        data = np.frombuffer(raw, dtype=dtype)

        # de-interleave: [L0,R0,L1,R1,...] -> per-channel arrays
        data = data.reshape(-1, n_channels)

        # normalize to float64 in [-1, 1]
        if dtype == np.uint8:
            float_data = (data.astype(np.float64) - 128) / 128.0
        else:
            max_val = float(np.iinfo(dtype).max)
            float_data = data.astype(np.float64) / max_val

        channels = []
        for c in range(n_channels):
            channels.append(
                Discrete_Signal(float_data[:, c], sample_rate, start_index=0)
            )

        return AudioClip(channels, name=None)

    @staticmethod
    def unload(clip: AudioClip, path, sample_width=2):
        if sample_width not in WavIO._DTYPE_BY_WIDTH:
            raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

        dtype = WavIO._DTYPE_BY_WIDTH[sample_width]

        stacked = np.stack([ch.samples for ch in clip.channels], axis=1)
        stacked = np.clip(stacked, -1.0, 1.0)

        if dtype == np.uint8:
            int_data = (stacked * 128 + 128).astype(dtype)
        else:
            max_val = float(np.iinfo(dtype).max)
            int_data = (stacked * max_val).astype(dtype)

        interleaved = int_data.reshape(-1)

        with wave.open(path, "wb") as wf:
            wf.setnchannels(clip.num_channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(clip.sample_rate)
            wf.writeframes(interleaved.tobytes())

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
