import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QToolButton,
    QMenu,
)
from PySide6.QtCore import Signal

from backend.workspace_model import Entity
from frontend.widgets.waveform_player import PlaybackGroup, WaveformPlayer


class VoiceEntityView(QWidget):
    """
    Shown when an entity tab is open in Voice Recognition mode.

    Unlike EntityPlotView (per-channel plots + the full edit-op
    toolbox), this is a read-only, playback-only view: a single plot of
    the sample-by-sample average across every channel, driven by the
    same WaveformPlayer used elsewhere. It gets Play/Pause/Reset and the
    Clip marker control "for free" in their default (File-mode) state,
    since Voice Recognition mode has no Edit/Tool menu and never calls
    set_edit_mode(True) / set_tool_mode(True) on it — those controls
    stay hidden automatically.

    Above the plot: three buttons for the speaker-recognition workflow.
    Only "Target Speaker" — this averaged waveform — is functional
    today; "Sample Speakers" is a dropdown with no entries yet, and
    "Similarity" is a disabled placeholder for a later feature.
    """

    entity_modified = Signal()

    _ACTIVE_BTN_STYLE = (
        "QPushButton { background-color: #2f81f7; color: white; font-weight: 600; }"
    )

    def __init__(self, entity: Entity, parent=None):
        super().__init__(parent)
        self._entity = entity
        self._group = PlaybackGroup.get_instance()
        self._player = None
        self._build_ui()

    @property
    def entity(self) -> Entity:
        return self._entity

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # ---- Target Speaker / Sample Speakers / Similarity ----
        button_row = QHBoxLayout()
        button_row.setSpacing(6)

        self._target_speaker_btn = QPushButton("Target Speaker")
        self._target_speaker_btn.setStyleSheet(self._ACTIVE_BTN_STYLE)
        self._target_speaker_btn.setEnabled(False)  # already the (only) active view

        self._sample_speakers_btn = QToolButton()
        self._sample_speakers_btn.setText("Sample Speakers")
        self._sample_speakers_btn.setPopupMode(QToolButton.InstantPopup)
        self._sample_speakers_menu = QMenu(self._sample_speakers_btn)
        self._sample_speakers_btn.setMenu(self._sample_speakers_menu)  # empty for now

        self._similarity_btn = QPushButton("Similarity")
        self._similarity_btn.setEnabled(False)  # inactive placeholder

        button_row.addWidget(self._target_speaker_btn)
        button_row.addWidget(self._sample_speakers_btn)
        button_row.addWidget(self._similarity_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        # ---- single averaged-channel plot ----
        averaged_signal = self._entity.clip.average_channel_signal()
        times = self._time_axis(averaged_signal)

        self._player = WaveformPlayer(
            times=times,
            series=[(averaged_signal.samples, "#4fc3f7", "Average")],
            sample_rate=averaged_signal.sample_rate,
            audio_data=averaged_signal.samples.astype(np.float32),
            group=self._group,
            entity=self._entity,
            channel_index=None,
            is_driver=True,
        )
        self._player.marker_added.connect(self.entity_modified.emit)
        layout.addWidget(self._player)
        layout.addStretch()

    def shutdown(self):
        """Called when the tab is closed — stop playback, mirroring
        EntityPlotView.shutdown()'s force_idle on every player."""
        if self._player is not None:
            self._player.force_idle()

    def clear_all_selection(self):
        """No-op hook so VoiceRecognitionPage can treat this the same as
        EntityPlotView when clearing selections on tab/mode changes —
        there is no selectable segment in this read-only view."""
        pass

    @staticmethod
    def _time_axis(signal):
        n = len(signal.samples)
        start_time = signal.get_time(signal.start_index)
        return start_time + np.arange(n) / signal.sample_rate