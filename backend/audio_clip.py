import os


class AudioClip:
    """
    Represents a multi-channel audio clip as a list of Discrete_Signal channels.
    channels[0] = left/mono, channels[1] = right, etc.
    """

    def __init__(self, channels, name):
        channels = list(channels)

        if len(channels) == 0:
            raise ValueError("AudioClip must have at least one channel.")

        rate = channels[0].sample_rate
        start = channels[0].start_index
        length = len(channels[0].samples)

        for ch in channels:
            if ch.sample_rate != rate:
                raise ValueError("All channels must share the same sample_rate.")
            if ch.start_index != start:
                raise ValueError("All channels must share the same start_index.")
            if len(ch.samples) != length:
                raise ValueError("All channels must have the same length.")

        self.channels = channels
        self.name = name

    @property
    def num_channels(self):
        return len(self.channels)

    @property
    def sample_rate(self):
        return self.channels[0].sample_rate

    @property
    def start_index(self):
        return self.channels[0].start_index

    def end_index(self):
        return self.channels[0].end_index()

    def get_time(self, index):
        """Delegates to channel 0 — all channels share timing by construction."""
        return self.channels[0].get_time(index)

    def get_index(self, seconds):
        return self.channels[0].get_index(seconds)

    def __getitem__(self, i):
        return self.channels[i]

    def __setitem__(self, i, value):
        if not isinstance(value, type(self.channels[0])):
            raise ValueError("Channel must be a Discrete_Signal.")
        self.channels[i] = value

    def __len__(self):
        return self.num_channels

    def copy(self):
        return AudioClip([ch.copy() for ch in self.channels], name=self.name)

    def apply(self, method_name, *args, channel=None, **kwargs):
        if channel is not None:
            new_channels = list(self.channels)
            new_channels[channel] = getattr(
                self.channels[channel], method_name
            )(*args, **kwargs)
            return AudioClip(new_channels, name=self.name)

        new_channels = [
            getattr(ch, method_name)(*args, **kwargs) for ch in self.channels
        ]
        return AudioClip(new_channels, name=self.name)

    def to_mono(self):
        import numpy as np
        stacked = np.stack([ch.samples for ch in self.channels])
        mono_samples = stacked.mean(axis=0)

        mono_signal = self.channels[0].copy()
        mono_signal.samples = mono_samples

        return AudioClip([mono_signal], name=self.name)

    def average_channel_signal(self):
        """
        A single Discrete_Signal that is the sample-by-sample average of
        every channel — used by Voice Recognition mode's single-plot
        view, which shows one averaged waveform instead of per-channel
        plots. Same math as to_mono(), but returns the raw signal instead
        of wrapping it back in a new AudioClip, since callers here only
        need samples/sample_rate/start_index for plotting and playback.
        """
        import numpy as np
        stacked = np.stack([ch.samples for ch in self.channels])
        averaged_samples = stacked.mean(axis=0)

        averaged_signal = self.channels[0].copy()
        averaged_signal.samples = averaged_samples
        return averaged_signal

    def to_stereo(self):
        if self.num_channels != 1:
            raise ValueError("to_stereo requires a mono AudioClip.")

        return AudioClip(
            [self.channels[0].copy(), self.channels[0].copy()],
            name=self.name
        )

    @staticmethod
    def load(path):
        from backend.audio_io import WavIO, MP3IO, OggIO  # deferred import breaks the cycle

        ext = path.rsplit(".", 1)[-1].lower()

        if ext == "wav":
            clip = WavIO.load(path)
        elif ext == "mp3":
            clip = MP3IO.load(path)
        elif ext == "ogg":
            clip = OggIO.load(path)
        else:
            raise ValueError(f"Unsupported audio format: .{ext}")

        clip.name = os.path.basename(path).rsplit(".", 1)[0]
        return clip

    def unload(self, file_type, directory=""):
        from backend.audio_io import WavIO, MP3IO, OggIO  # deferred import breaks the cycle

        if self.name is None:
            raise ValueError("AudioClip has no name; set clip.name before unloading.")

        ext = file_type.lower()
        filename = f"{self.name}.{ext}"
        path = os.path.join(directory, filename) if directory else filename

        if ext == "wav":
            WavIO.unload(self, path)
        elif ext == "mp3":
            MP3IO.unload(self, path)
        elif ext == "ogg":
            OggIO.unload(self, path)
        else:
            raise ValueError(f"Unsupported audio format: .{ext}")