import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea

from backend.workspace_model import Entity
from frontend.widgets.waveform_player import PlaybackGroup, WaveformPlayer


_CHANNEL_COLORS = ["#4fc3f7", "#ff8a65", "#81c784", "#ba68c8", "#ffd54f", "#a1887f"]


class EntityPlotView(QWidget):

    entity_modified = Signal()
    """
    Shown when an entity tab is active: a 'Channels' section (one scrolling
    waveform per channel, skipped for mono clips) and an 'Overall' section
    (all channels overlaid). Every plot has independent play/pause/reset,
    but only one plot across the whole entity can play at a time, and the
    channel plots visually follow the Overall plot while it's driving.
    """

    def __init__(self, entity: Entity, parent=None):
        super().__init__(parent)
        self._entity = entity
        self._group = PlaybackGroup.get_instance()
        self._players = []
        self._build_ui()

    def _build_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer_layout.addWidget(scroll)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        clip = self._entity.clip
        num_channels = clip.num_channels
        times = self._time_axis(clip.channels[0])

        if num_channels > 1:
            channels_label = QLabel("Channels")
            channels_label.setStyleSheet("font-size: 16px; font-weight: 600;")
            layout.addWidget(channels_label)

            for i, channel in enumerate(clip.channels):
                color = _CHANNEL_COLORS[i % len(_CHANNEL_COLORS)]
                player = WaveformPlayer(
                    times=times,
                    series=[(channel.samples, color, f"Ch {i + 1}")],
                    sample_rate=channel.sample_rate,
                    audio_data=channel.samples.astype(np.float32),
                    group=self._group,
                    entity=self._entity,
                    channel_index=i,
                    is_driver=False,
                )
                layout.addWidget(player)
                player.marker_added.connect(self.entity_modified.emit)
                self._players.append(player)

        overall_label = QLabel("Overall")
        overall_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(overall_label)

        overall_series = [
            (channel.samples, _CHANNEL_COLORS[i % len(_CHANNEL_COLORS)], f"Ch {i + 1}")
            for i, channel in enumerate(clip.channels)
        ]
        overall_player = WaveformPlayer(
            times=times,
            series=overall_series,
            sample_rate=clip.sample_rate,
            audio_data=self._playback_mix(clip),
            group=self._group,
            entity=self._entity,
            channel_index=None,
            is_driver=True,
        )
        layout.addWidget(overall_player)
        overall_player.marker_added.connect(self.entity_modified.emit)
        self._players.append(overall_player)

        layout.addStretch()
        scroll.setWidget(container)

    def shutdown(self):
        """Force every plot in this tab back to idle/released state.
        Must be called before the tab is closed, otherwise a lingering
        active/clip-mode lock in PlaybackGroup can block every other
        player until the app restarts."""
        for player in self._players:
            player.force_idle()

    @staticmethod
    def _time_axis(channel):
        n = len(channel.samples)
        start_time = channel.get_time(channel.start_index)
        return start_time + np.arange(n) / channel.sample_rate

    @staticmethod
    def _playback_mix(clip):
        """
        Audio actually sent to the output device for the 'Overall' plot.
        Most sound cards only expose stereo output, so clips with more than
        2 channels are downmixed to mono for playback — the plot still
        shows every channel individually regardless.
        """
        stacked = np.stack([ch.samples for ch in clip.channels], axis=1).astype(np.float32)

        if stacked.shape[1] <= 2:
            return stacked
        return stacked.mean(axis=1).astype(np.float32)