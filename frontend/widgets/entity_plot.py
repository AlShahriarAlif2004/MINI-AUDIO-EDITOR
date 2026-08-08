import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea

from backend.workspace_model import Entity
from backend.audio_clip import AudioClip
from backend.discrete_signal import Discrete_Signal

from frontend.widgets.waveform_player import PlaybackGroup, WaveformPlayer


_CHANNEL_COLORS = ["#4fc3f7", "#ff8a65", "#81c784", "#ba68c8", "#ffd54f", "#a1887f"]

class _SegmentCanvas(QWidget):
    """Container that holds the per-channel/overall plots. A click that
    lands on empty space (not on a plot itself) clears the current
    segment selection — the plots consume their own clicks, so this
    only fires for gaps, labels, and margins."""

    clicked_empty = Signal()

    def mousePressEvent(self, event):
        self.clicked_empty.emit()
        super().mousePressEvent(event)

class EntityPlotView(QWidget):

    entity_modified = Signal()
    entity_trimmed = Signal()
    entity_timescaled = Signal()
    entity_vscaled = Signal()
    entity_reversed = Signal()
    entity_faded_in = Signal()
    entity_faded_out = Signal()
    trim_cancelled = Signal()
    segment_selected = Signal(object, object, float, float)
    segment_deselected = Signal()
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
        self._selected_player = None
        self._trim_player = None
        self._trim_range = None
        self._timescale_preview_active = False
        self._vscale_player = None
        self._vscale_range = None
        self._vscale_mirror_players = []     # channel players mirrored when Overall is selected
        self._vscale_overall_target = None   # Overall player mirrored when a channel is selected
        self._effect_op = None               # 'reverse' | 'fade_in' | 'fade_out'
        self._effect_player = None
        self._effect_range = None
        self._effect_mirror_players = []     # channel players mirrored when Overall is selected
        self._effect_overall_target = None   # Overall player mirrored when a channel is selected
        self._build_ui()

    @property
    def entity(self) -> Entity:
        return self._entity

    def _lock_selection(self, locked: bool):
        for player in self._players:
            player.set_segment_locked(locked)

    def _find_overall_player(self):
        for player in self._players:
            if player.channel_index is None:
                return player
        return None

    def begin_trim(self) -> bool:
        if self._selected_player is None or self._selected_player.channel_index is not None:
            return False
        segment = self._selected_player.selected_segment
        if segment is None:
            return False

        self._trim_player = self._selected_player
        self._trim_range = segment
        for player in self._players:
            player.enter_trim_preview(*segment)
        self._lock_selection(True)
        return True

    def cancel_trim(self):
        """Exit trim preview, restore original graph, keep selection."""
        if self._trim_player is not None:
            for player in self._players:
                player.exit_trim_preview()
        self._trim_player = None
        self._trim_range = None
        self._lock_selection(False)
        # Restore selection highlight (don't clear the selection)
        if self._selected_player is not None:
            self._selected_player.restore_selection_visual()

    def apply_trim(self):
        if self._trim_player is None or self._trim_range is None:
            return
        start, end = self._trim_range
        self._perform_trim(start, end)
        self._trim_player = None
        self._trim_range = None
        self._lock_selection(False)
        self.clear_all_selection()   # apply clears selection
        self.entity_modified.emit()
        self.entity_trimmed.emit()

    # ------------------------------------------------------------------
    # Time-scale preview
    # ------------------------------------------------------------------

    def begin_timescale(self) -> bool:
        """Enter timescale preview. Only valid when the overall plot is selected."""
        if self._selected_player is None or self._selected_player.channel_index is not None:
            return False
        self._timescale_preview_active = True
        for player in self._players:
            player.enter_timescale_preview(1.0)
        self._lock_selection(True)
        return True

    def update_timescale_preview(self, factor: float):
        """Update the live preview as the factor changes."""
        if not self._timescale_preview_active:
            return
        for player in self._players:
            player.update_timescale_preview(factor)

    def cancel_timescale(self):
        """Exit timescale preview, restore graph, keep selection."""
        if not self._timescale_preview_active:
            return
        self._timescale_preview_active = False
        for player in self._players:
            player.exit_timescale_preview()
        self._lock_selection(False)
        # Restore selection highlight
        if self._selected_player is not None:
            self._selected_player.restore_selection_visual()

    def apply_timescale(self, factor: float):
        """Apply timescale permanently, exit preview, clear selection, rebuild."""
        if not self._timescale_preview_active:
            return
        self._timescale_preview_active = False
        for player in self._players:
            player.exit_timescale_preview()
        self._perform_timescale(factor)
        self._lock_selection(False)
        self.clear_all_selection()   # apply clears selection
        self.entity_modified.emit()
        self.entity_timescaled.emit()

    def _perform_timescale(self, factor: float):
        clip = self._entity.clip
        new_channels = [ch.time_scale(factor) for ch in clip.channels]
        self._entity.clip = AudioClip(new_channels, name=clip.name)
        # Rescale marker/division timestamps proportionally
        self._entity.divisions = [t * factor for t in self._entity.divisions]
        for ch_idx in list(self._entity.channel_markers.keys()):
            self._entity.channel_markers[ch_idx] = [
                t * factor for t in self._entity.channel_markers[ch_idx]
            ]

    # ------------------------------------------------------------------
    # Vertical-scale preview
    # ------------------------------------------------------------------

    def begin_vertical_scale(self) -> bool:
        if self._selected_player is None:
            return False
        segment = self._selected_player.selected_segment
        if segment is None:
            return False

        self._vscale_player = self._selected_player
        self._vscale_range = segment
        self._vscale_player.enter_vertical_scale_preview(1.0, *segment)

        self._vscale_overall_target = None
        self._vscale_mirror_players = []

        if self._vscale_player.channel_index is not None:
            # A single channel was selected — mirror the preview onto that
            # channel's curve inside the Overall plot.
            overall_player = self._find_overall_player()
            if overall_player is not None and overall_player is not self._vscale_player:
                overall_player.preview_channel_curve(self._vscale_player.channel_index, 1.0, *segment)
                self._vscale_overall_target = overall_player
        else:
            # The Overall plot was selected — it scales every channel
            # uniformly, so mirror the same preview onto each channel plot.
            self._vscale_mirror_players = [
                p for p in self._players
                if p is not self._vscale_player and p.channel_index is not None
            ]
            for player in self._vscale_mirror_players:
                player.enter_vertical_scale_preview(1.0, *segment)

        self._lock_selection(True)
        return True

    def update_vertical_scale_preview(self, factor: float):
        if self._vscale_player is not None:
            self._vscale_player.update_vertical_scale_preview(factor)
        if self._vscale_overall_target is not None:
            self._vscale_overall_target.update_channel_curve_preview(factor)
        for player in self._vscale_mirror_players:
            player.update_vertical_scale_preview(factor)

    def cancel_vertical_scale(self):
        if self._vscale_player is None:
            return
        self._vscale_player.exit_vertical_scale_preview()
        if self._vscale_overall_target is not None:
            self._vscale_overall_target.exit_channel_curve_preview()
        for player in self._vscale_mirror_players:
            player.exit_vertical_scale_preview()
        self._vscale_player = None
        self._vscale_overall_target = None
        self._vscale_mirror_players = []
        self._vscale_range = None
        self._lock_selection(False)
        if self._selected_player is not None:
            self._selected_player.restore_selection_visual()

    def apply_vertical_scale(self, factor: float):
        if self._vscale_player is None:
            return
        channel_index = self._vscale_player.channel_index
        start, end = self._vscale_range
        self._vscale_player.exit_vertical_scale_preview()
        if self._vscale_overall_target is not None:
            self._vscale_overall_target.exit_channel_curve_preview()
        for player in self._vscale_mirror_players:
            player.exit_vertical_scale_preview()
        self._perform_vertical_scale(factor, channel_index, start, end)
        self._vscale_player = None
        self._vscale_overall_target = None
        self._vscale_mirror_players = []
        self._vscale_range = None
        self._lock_selection(False)
        self.clear_all_selection()
        self.entity_modified.emit()
        self.entity_vscaled.emit()

    def _perform_vertical_scale(self, factor, channel_index, start, end):
        clip = self._entity.clip
        start_idx = clip.get_index(start)
        end_idx = clip.get_index(end)

        if channel_index is None:
            new_clip = clip.apply("vertical_scale", factor, start_idx, end_idx)
        else:
            new_clip = clip.apply("vertical_scale", factor, start_idx, end_idx, channel=channel_index)

        self._entity.clip = new_clip

    # ------------------------------------------------------------------
    # Reverse / Fade In / Fade Out preview
    # ------------------------------------------------------------------
    # Same architecture as vertical scale: a preview lives on the selected
    # player and, when a channel is selected, mirrors onto that channel's
    # curve in the Overall plot; when Overall is selected, mirrors onto
    # every channel plot. The only difference from vertical scale is that
    # none of these operations take a factor, so there's no live-update
    # step — the preview is computed once when the op begins.

    def begin_effect(self, op) -> bool:
        if self._selected_player is None:
            return False
        segment = self._selected_player.selected_segment
        if segment is None:
            return False

        self._effect_op = op
        self._effect_player = self._selected_player
        self._effect_range = segment
        self._effect_player.enter_effect_preview(op, *segment)

        self._effect_overall_target = None
        self._effect_mirror_players = []

        if self._effect_player.channel_index is not None:
            # A single channel was selected — mirror the preview onto that
            # channel's curve inside the Overall plot.
            overall_player = self._find_overall_player()
            if overall_player is not None and overall_player is not self._effect_player:
                overall_player.preview_channel_curve_effect(self._effect_player.channel_index, op, *segment)
                self._effect_overall_target = overall_player
        else:
            # The Overall plot was selected — it applies the effect to
            # every channel uniformly, so mirror the same preview onto
            # each channel plot.
            self._effect_mirror_players = [
                p for p in self._players
                if p is not self._effect_player and p.channel_index is not None
            ]
            for player in self._effect_mirror_players:
                player.enter_effect_preview(op, *segment)

        self._lock_selection(True)
        return True

    def begin_reverse(self) -> bool:
        return self.begin_effect("reverse")

    def begin_fade_in(self) -> bool:
        return self.begin_effect("fade_in")

    def begin_fade_out(self) -> bool:
        return self.begin_effect("fade_out")

    def cancel_effect(self):
        if self._effect_player is None:
            return
        self._effect_player.exit_effect_preview()
        if self._effect_overall_target is not None:
            self._effect_overall_target.exit_channel_curve_preview()
        for player in self._effect_mirror_players:
            player.exit_effect_preview()
        self._effect_op = None
        self._effect_player = None
        self._effect_overall_target = None
        self._effect_mirror_players = []
        self._effect_range = None
        self._lock_selection(False)
        if self._selected_player is not None:
            self._selected_player.restore_selection_visual()

    def apply_effect(self):
        if self._effect_player is None:
            return
        op = self._effect_op
        channel_index = self._effect_player.channel_index
        start, end = self._effect_range
        self._effect_player.exit_effect_preview()
        if self._effect_overall_target is not None:
            self._effect_overall_target.exit_channel_curve_preview()
        for player in self._effect_mirror_players:
            player.exit_effect_preview()
        self._perform_effect(op, channel_index, start, end)
        self._effect_op = None
        self._effect_player = None
        self._effect_overall_target = None
        self._effect_mirror_players = []
        self._effect_range = None
        self._lock_selection(False)
        self.clear_all_selection()
        self.entity_modified.emit()
        if op == "reverse":
            self.entity_reversed.emit()
        elif op == "fade_in":
            self.entity_faded_in.emit()
        elif op == "fade_out":
            self.entity_faded_out.emit()

    def _perform_effect(self, op, channel_index, start, end):
        clip = self._entity.clip
        start_idx = clip.get_index(start)
        end_idx = clip.get_index(end)

        if channel_index is None:
            new_clip = clip.apply(op, start_idx, end_idx)
        else:
            new_clip = clip.apply(op, start_idx, end_idx, channel=channel_index)

        self._entity.clip = new_clip

    def _perform_trim(self, start, end):
        clip = self._entity.clip
        remove_start_idx = clip.get_index(start)
        remove_end_idx = clip.get_index(end)

        new_channels = []
        for ch in clip.channels:
            before = None
            after = None

            if remove_start_idx > ch.start_index:
                before = ch.trim(ch.start_index, remove_start_idx - 1)
            if remove_end_idx < ch.end_index():
                after = ch.trim(remove_end_idx + 1, ch.end_index())

            if before is not None and after is not None:
                new_channels.append(before.concatenate(after))
            elif before is not None:
                new_channels.append(before)
            elif after is not None:
                new_channels.append(after)
            else:
                # the whole channel was inside the selection — keep a single
                # silent sample so the clip never collapses to zero length
                new_channels.append(Discrete_Signal(
                    np.zeros(1, dtype=ch.samples.dtype), ch.sample_rate, ch.start_index
                ))

        self._entity.clip = AudioClip(new_channels, name=clip.name)

        removed_span = end - start

        def _shift_time(t):
            if t <= start:
                return t
            if t >= end:
                return t - removed_span
            return None  # fell inside the removed span — drop it

        self._entity.divisions = [
            t for t in (_shift_time(d) for d in self._entity.divisions) if t is not None
        ]
        for ch_idx in list(self._entity.channel_markers.keys()):
            self._entity.channel_markers[ch_idx] = [
                t for t in (_shift_time(m) for m in self._entity.channel_markers[ch_idx])
                if t is not None
            ]

    def _on_canvas_clicked(self):
        if (self._trim_player is not None or self._timescale_preview_active
                or self._vscale_player is not None or self._effect_player is not None):
            return
        self.clear_all_selection()
        if self._trim_player is not None:
            self.cancel_trim()
            self.trim_cancelled.emit()

    def _build_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer_layout.addWidget(scroll)

        container = _SegmentCanvas()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)
        container.clicked_empty.connect(self._on_canvas_clicked)

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
                player.segment_selected.connect(self._on_segment_selected)
                player.segment_deselected.connect(self._on_segment_deselected)
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
        overall_player.segment_selected.connect(self._on_segment_selected)
        overall_player.segment_deselected.connect(self._on_segment_deselected)
        self._players.append(overall_player)

        layout.addStretch()
        scroll.setWidget(container)

    def shutdown(self):
        self.cancel_trim()
        self.cancel_timescale()
        self.cancel_vertical_scale()
        self.cancel_effect()
        for player in self._players:
            player.force_idle()
        self.clear_all_selection()

    def _on_segment_selected(self, player, start, end):
        if self._selected_player is not None and self._selected_player is not player:
            self._selected_player.clear_selection()
        self._selected_player = player
        self.segment_selected.emit(self._entity, player.channel_index, start, end)

    def _on_segment_deselected(self, player):
        if self._selected_player is player:
            self._selected_player = None
            self.segment_deselected.emit()

    def clear_all_selection(self):
        if self._selected_player is not None:
            self._selected_player.clear_selection()

    def set_edit_mode(self, enabled: bool):
        """Hide/show each plot's Play and Clip controls when the workspace
        switches between File and Edit top-level modes."""
        for player in self._players:
            player.set_edit_mode(enabled)

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