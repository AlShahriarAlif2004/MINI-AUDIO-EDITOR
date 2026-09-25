import numpy as np
import sounddevice as sd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QFrame,
    QGridLayout,
    QVBoxLayout,
    QWidget,
)

from backend.workspace_model import Entity, Workspace
from backend.speaker_similarity import rank_sample_speakers
from frontend.dialogs.add_sample_speaker_dialog import AddSampleSpeakerDialog
from frontend.dialogs.delete_sample_speaker_dialog import DeleteSampleSpeakerDialog
from frontend.widgets.waveform_player import PlaybackGroup, WaveformPlayer


class VoiceEntityView(QWidget):
    """
    Shown when an entity tab is open in Voice Recognition mode.

    Unlike EntityPlotView (per-channel plots + the full edit-op
    toolbox), this is a read-only, playback-only view driven by the same
    WaveformPlayer used elsewhere, with no Clip control and no white
    boundary/division lines (this is a listen-only speaker-recognition
    view, not an editing surface).

    Above the plot sit three buttons for the speaker-recognition workflow:

      - "Target Speaker" — the entity's own sample-by-sample averaged
        waveform. Selected by default.
      - "Sample Speakers" — a per-tab list of other entities picked as
        comparison samples. The first click on this button (while Target
        Speaker is active) switches to it; a second click, while it's
        already active, expands/collapses a dropdown list of the
        speakers added so far (shown even when empty, so it's clear
        there's nothing there yet). Selecting an entry shows that
        speaker's averaged waveform. "Add Sample Speaker" / "Delete
        Sample Speaker" live at the bottom-right of the view and are
        only shown while this view is active.
            - "Similarity" — the ranked result view, enabled after a ranking has
                been calculated for the current sample-speaker list.

    A title label above the plot names whichever waveform is currently
    shown: the target entity's name, the selected sample speaker's name,
    or "No selected sample speaker" when the Sample Speakers view is
    active but nothing (yet) is selected.
    """

    entity_modified = Signal()

    _ACTIVE_BTN_STYLE = (
        "QPushButton { background-color: #2f81f7; color: white; font-weight: 600; }"
    )
    _BTN_MIN_SIZE = (150, 34)   # (width, height) — a bit wider than the Qt default
    _SAMPLE_ACTION_MIN_SIZE = (190, 38)

    VIEW_TARGET = "target"
    VIEW_SAMPLE = "sample"
    VIEW_SIMILARITY = "similarity"

    def __init__(self, entity: Entity, workspace: Workspace | None = None, parent=None):
        super().__init__(parent)
        self._entity = entity
        self._workspace = workspace
        self._group = PlaybackGroup.get_instance()

        self._player: WaveformPlayer | None = None   # whichever waveform is currently shown
        self._active_view = self.VIEW_TARGET
        self._sample_speakers: list[Entity] = []      # per-tab, not persisted
        self._selected_sample_speaker: Entity | None = None
        self._dropdown_visible = False
        self._similarity_results = []
        self._similarity_ready = False

        self._build_ui()
        self._refresh_button_styles()
        self._refresh_title()
        self._refresh_plot()

    @property
    def entity(self) -> Entity:
        return self._entity

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # ---- Target Speaker / Sample Speakers / Similarity ----
        button_row = QHBoxLayout()
        button_row.setSpacing(6)

        self._target_speaker_btn = QPushButton("Target Speaker")
        self._sample_speakers_btn = QPushButton("Sample Speakers")
        self._similarity_btn = QPushButton("Similarity")
        self._similarity_btn.setEnabled(False)

        for btn in (self._target_speaker_btn, self._sample_speakers_btn, self._similarity_btn):
            btn.setMinimumSize(*self._BTN_MIN_SIZE)

        self._target_speaker_btn.clicked.connect(self._on_target_speaker_clicked)
        self._sample_speakers_btn.clicked.connect(self._on_sample_speakers_clicked)
        self._similarity_btn.clicked.connect(self._on_similarity_clicked)

        button_row.addWidget(self._target_speaker_btn)
        button_row.addWidget(self._sample_speakers_btn)
        button_row.addWidget(self._similarity_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        # ---- Sample Speakers panel: dropdown (only when active) ----
        self._sample_panel = QWidget()
        panel_layout = QVBoxLayout(self._sample_panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(4)

        self._dropdown_list = QListWidget()
        self._dropdown_list.setMaximumHeight(120)
        self._dropdown_list.setStyleSheet(
            "QListWidget::item { padding-left: 30px; }"
            "QListWidget::item:selected { background-color: #2f81f7; color: white; }"
        )
        self._dropdown_list.itemClicked.connect(self._on_dropdown_item_clicked)
        self._dropdown_list.setVisible(False)
        panel_layout.addWidget(self._dropdown_list)

        self._sample_panel.setVisible(False)
        layout.addWidget(self._sample_panel)

        self._similarity_panel = QWidget()
        self._similarity_layout = QVBoxLayout(self._similarity_panel)
        self._similarity_layout.setContentsMargins(0, 0, 0, 0)
        self._similarity_layout.setSpacing(6)
        self._similarity_panel.setVisible(False)
        layout.addWidget(self._similarity_panel)

        # ---- title above the plot ----
        self._title_label = QLabel("")
        self._title_label.setStyleSheet("font-size: 15px; font-weight: 600;")
        layout.addWidget(self._title_label)

        # ---- plot area (swapped between target / sample speaker waveforms) ----
        self._plot_container = QWidget()
        self._plot_layout = QVBoxLayout(self._plot_container)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot_container)
        layout.addStretch()

        # ---- Sample speaker actions: bottom-right of the whole view ----
        action_row = QHBoxLayout()
        action_row.addStretch()
        self._add_sample_btn = QPushButton("Add Sample Speaker")
        self._delete_sample_btn = QPushButton("Delete Sample Speaker")
        self._find_similarity_btn = QPushButton("Find Similarity")
        self._find_similarity_btn.setEnabled(False)
        for button in (self._add_sample_btn, self._delete_sample_btn):
            button.setMinimumSize(*self._SAMPLE_ACTION_MIN_SIZE)
        self._find_similarity_btn.setMinimumSize(*self._SAMPLE_ACTION_MIN_SIZE)
        self._add_sample_btn.clicked.connect(self._on_add_sample_speaker_clicked)
        self._delete_sample_btn.clicked.connect(self._on_delete_sample_speaker_clicked)
        self._find_similarity_btn.clicked.connect(self._on_find_similarity_clicked)
        action_row.addWidget(self._add_sample_btn)
        action_row.addWidget(self._delete_sample_btn)
        action_row.addWidget(self._find_similarity_btn)
        self._sample_actions = QWidget()
        self._sample_actions.setLayout(action_row)
        self._sample_actions.setVisible(True)
        layout.addWidget(self._sample_actions)

    # ------------------------------------------------------------------
    # Target Speaker / Sample Speakers toggle
    # ------------------------------------------------------------------

    def _on_target_speaker_clicked(self):
        if self._active_view == self.VIEW_TARGET:
            return
        self._active_view = self.VIEW_TARGET
        self._set_dropdown_visible(False)
        self._sample_panel.setVisible(False)
        self._similarity_panel.setVisible(False)
        self._sample_actions.setVisible(False)
        self._title_label.setVisible(True)
        self._plot_container.setVisible(True)
        self._refresh_button_styles()
        self._refresh_title()
        self._refresh_plot()

    def _on_sample_speakers_clicked(self):
        if self._active_view != self.VIEW_SAMPLE:
            self._active_view = self.VIEW_SAMPLE
            self._sample_panel.setVisible(True)
            self._similarity_panel.setVisible(False)
            self._sample_actions.setVisible(True)
            self._set_dropdown_visible(False)
            self._title_label.setVisible(True)
            self._plot_container.setVisible(True)
            self._refresh_button_styles()
            self._refresh_title()
            self._refresh_plot()
        else:
            # Already active — toggle the dropdown open/closed instead.
            self._set_dropdown_visible(not self._dropdown_visible)

    def _on_similarity_clicked(self):
        if not self._similarity_ready:
            return
        self._active_view = self.VIEW_SIMILARITY
        self._set_dropdown_visible(False)
        self._sample_panel.setVisible(False)
        self._sample_actions.setVisible(False)
        self._similarity_panel.setVisible(True)
        self._title_label.setVisible(False)
        self._plot_container.setVisible(False)
        self._refresh_button_styles()
        self._refresh_plot()
        self._refresh_similarity_panel()

    def _on_find_similarity_clicked(self):
        if not self._sample_speakers:
            return
        self._similarity_results = rank_sample_speakers(
            self._entity.clip, self._sample_speakers
        )
        self._similarity_ready = True
        self._similarity_btn.setEnabled(True)
        self._on_similarity_clicked()

    def _refresh_button_styles(self):
        is_target = self._active_view == self.VIEW_TARGET
        is_sample = self._active_view == self.VIEW_SAMPLE
        self._target_speaker_btn.setStyleSheet(self._ACTIVE_BTN_STYLE if is_target else "")
        self._sample_speakers_btn.setStyleSheet(self._ACTIVE_BTN_STYLE if is_sample else "")
        self._similarity_btn.setStyleSheet(
            self._ACTIVE_BTN_STYLE if self._active_view == self.VIEW_SIMILARITY else ""
        )

    # ------------------------------------------------------------------
    # Dropdown (list of this tab's added sample speakers)
    # ------------------------------------------------------------------

    def _set_dropdown_visible(self, visible: bool):
        self._dropdown_visible = visible
        if visible:
            self._populate_dropdown()
        self._dropdown_list.setVisible(visible)

    def _populate_dropdown(self):
        self._dropdown_list.clear()
        if not self._sample_speakers:
            placeholder = QListWidgetItem("No sample speakers added yet")
            placeholder.setFlags(Qt.NoItemFlags)
            self._dropdown_list.addItem(placeholder)
            return
        for speaker in self._sample_speakers:
            item = QListWidgetItem(speaker.name)
            item.setData(Qt.UserRole, speaker)
            self._dropdown_list.addItem(item)
            if speaker is self._selected_sample_speaker:
                item.setSelected(True)
                self._dropdown_list.setCurrentItem(item)

    def _on_dropdown_item_clicked(self, item: QListWidgetItem):
        speaker = item.data(Qt.UserRole)
        if speaker is None:
            return
        self._selected_sample_speaker = speaker
        self._set_dropdown_visible(False)
        self._refresh_title()
        self._refresh_plot()

    # ------------------------------------------------------------------
    # Add / Delete sample speaker
    # ------------------------------------------------------------------

    def _on_add_sample_speaker_clicked(self):
        dialog = AddSampleSpeakerDialog(
            self._workspace,
            parent=self,
            excluded_entities=self._sample_speakers,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        entity = dialog.selected_entity()
        if entity is None:
            return

        if entity in self._sample_speakers:
            return
        self._sample_speakers.append(entity)
        self._selected_sample_speaker = entity
        self._reset_similarity()

        if self._dropdown_visible:
            self._populate_dropdown()

        self._refresh_title()
        self._refresh_plot()

    def _on_delete_sample_speaker_clicked(self):
        dialog = DeleteSampleSpeakerDialog(self._sample_speakers, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        to_delete = dialog.selected_entities()
        if not to_delete:
            return

        removed_current = self._selected_sample_speaker in to_delete
        self._sample_speakers = [
            speaker for speaker in self._sample_speakers if speaker not in to_delete
        ]
        self._reset_similarity()

        if removed_current:
            self._selected_sample_speaker = None

        if self._dropdown_visible:
            self._populate_dropdown()

        self._refresh_title()
        if removed_current:
            self._refresh_plot()

    # ------------------------------------------------------------------
    # Title + plot
    # ------------------------------------------------------------------

    def _reset_similarity(self):
        self._similarity_results = []
        self._similarity_ready = False
        self._similarity_btn.setEnabled(False)
        self._find_similarity_btn.setEnabled(bool(self._sample_speakers))
        if self._active_view == self.VIEW_SIMILARITY:
            self._active_view = self.VIEW_TARGET
            self._similarity_panel.setVisible(False)
            self._sample_actions.setVisible(True)
            self._refresh_button_styles()
            self._refresh_title()
            self._refresh_plot()

    def _refresh_similarity_panel(self):
        while self._similarity_layout.count():
            item = self._similarity_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._add_similarity_section("Target Speaker", [(self._entity, None)])
        if not self._similarity_results:
            return

        best_score = self._similarity_results[0][1]
        nearest = [
            result for result in self._similarity_results
            if np.isclose(result[1], best_score, rtol=1e-5, atol=1e-8)
        ]
        self._add_similarity_section("Nearest Sample Speaker", nearest)

    def _add_similarity_section(self, title, rows):
        section = QLabel(title)
        section.setStyleSheet("font-size: 17px; font-weight: 700;")
        self._similarity_layout.addWidget(section)

        for entity, score in rows:
            row = QWidget()
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(4, 4, 4, 4)
            row_layout.setSpacing(6)

            name = QLabel(entity.name)
            name.setStyleSheet("font-size: 15px; font-weight: 600;")
            if score is not None:
                name.setToolTip(f"Average quantization distortion: {score:.6g}")
            row_layout.addWidget(name)

            player = self._build_similarity_player(entity)
            row_layout.addWidget(player)
            self._similarity_layout.addWidget(row)

    def _build_similarity_player(self, entity):
        signal = entity.clip.average_channel_signal()
        times = self._time_axis(signal)
        player = WaveformPlayer(
            times=times,
            series=[(signal.samples, "#4fc3f7", "Average")],
            sample_rate=signal.sample_rate,
            audio_data=signal.samples.astype(np.float32),
            group=self._group,
            entity=entity,
            channel_index=None,
            is_driver=False,
            show_markers=False,
            show_clip_button=False,
            parent=self._similarity_panel,
        )
        player.marker_added.connect(self.entity_modified.emit)
        return player

    @staticmethod
    def _play_entity(entity):
        signal = entity.clip.average_channel_signal()
        sd.stop()
        sd.play(signal.samples.astype(np.float32), signal.sample_rate)

    def _refresh_title(self):
        if self._active_view == self.VIEW_TARGET:
            self._title_label.setText(self._entity.name)
        elif self._active_view == self.VIEW_SIMILARITY:
            self._title_label.setText("Speaker Similarity")
        else:
            if self._selected_sample_speaker is not None:
                self._title_label.setText(self._selected_sample_speaker.name)
            else:
                self._title_label.setText("No selected sample speaker")

    def _refresh_plot(self):
        if self._player is not None:
            self._player.force_idle()
            self._plot_layout.removeWidget(self._player)
            self._player.deleteLater()
            self._player = None

        shown_entity = None
        if self._active_view == self.VIEW_TARGET:
            shown_entity = self._entity
        elif self._selected_sample_speaker is not None:
            shown_entity = self._selected_sample_speaker

        if shown_entity is None:
            return

        averaged_signal = shown_entity.clip.average_channel_signal()
        times = self._time_axis(averaged_signal)

        self._player = WaveformPlayer(
            times=times,
            series=[(averaged_signal.samples, "#4fc3f7", "Average")],
            sample_rate=averaged_signal.sample_rate,
            audio_data=averaged_signal.samples.astype(np.float32),
            group=self._group,
            entity=shown_entity,
            channel_index=None,
            is_driver=True,
            show_markers=False,
            show_clip_button=False,
            parent=self._plot_container,
        )
        self._player.marker_added.connect(self.entity_modified.emit)
        self._plot_layout.addWidget(self._player)

    # ------------------------------------------------------------------
    # Lifecycle hooks (called by VoiceRecognitionPage)
    # ------------------------------------------------------------------

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