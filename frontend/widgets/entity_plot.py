import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QPushButton, QStackedWidget,
)

from backend.workspace_model import Entity
from backend.audio_clip import AudioClip
from backend.discrete_signal import Discrete_Signal

from frontend.widgets.waveform_player import PlaybackGroup, WaveformPlayer


_CHANNEL_COLORS = ["#4fc3f7", "#ff8a65", "#81c784", "#ba68c8", "#ffd54f", "#a1887f"]

# Spectral subtraction leaves residual noise after a single pass; running
# the same subtraction again on its own output measurably cleans it up
# further. This is how many passes are run automatically instead of the
# user re-triggering Filter by hand.
_NOISE_REMOVAL_ITERATIONS = 10

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
    entity_concatenated = Signal()
    entity_extracted = Signal()
    entity_divisor_deleted = Signal()
    entity_noise_removed = Signal()
    entity_echo_added = Signal()
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
        self._extract_player = None
        self._extract_range = None
        self._timescale_preview_active = False
        self._timescale_range = None
        self._vscale_player = None
        self._vscale_range = None
        self._vscale_mirror_players = []     # channel players mirrored when Overall is selected
        self._vscale_overall_target = None   # Overall player mirrored when a channel is selected
        self._reverse_player = None
        self._reverse_range = None
        self._reverse_mirror_players = []    # channel players mirrored when Overall is selected
        self._reverse_overall_target = None  # Overall player mirrored when a channel is selected
        self._fadein_player = None
        self._fadein_range = None
        self._fadein_mirror_players = []     # channel players mirrored when Overall is selected
        self._fadein_overall_target = None   # Overall player mirrored when a channel is selected
        self._fadeout_player = None
        self._fadeout_range = None
        self._fadeout_mirror_players = []    # channel players mirrored when Overall is selected
        self._fadeout_overall_target = None  # Overall player mirrored when a channel is selected
        self._concat_player = None
        self._concat_range = None
        self._concat_sources = []
        self._concat_extra_channel_samples = None
        self._delete_divisor_active = False
        
        self._echo_original_clip = None      # clip snapshot before echo is applied
        self._echo_parameters = None         # (occurrence, delay, decay) tuple

        self._noise_removal_active = False   # True for either phase (selection or filtered)
        self._noise_filtered = False         # True once Filter has been clicked
        self._noise_from_entity = False      # True when the noise source is another entity's channel
        self._noise_reference_signal = None  # the other entity's channel, when _noise_from_entity
        self._noise_overall_player = None
        self._noise_range = None             # (start_time, end_time) chosen by the user
        self._noise_original_clip = None     # clip snapshot taken at Filter time
        self._noise_filtered_clip = None     # AudioClip produced by Filter
        self._noise_tab_widget = None        # container inserted above the normal players
        self._noise_tab_stack = None
        self._noise_tab_buttons = {}
        self._noise_tab_pages = {}
        self._noise_preview_players = []     # every WaveformPlayer built for the tab strip

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

    # ------------------------------------------------------------------
    # Delete divisor
    # ------------------------------------------------------------------
    # Unlike Trim/Extract/etc., this needs no pre-selected segment: it
    # simply arms every plot (channel + Overall) for divisor-line
    # selection. Each plot stages its own deletions locally (hidden but
    # not yet removed); Apply here collects whatever was staged across
    # every plot and strips those times out of the entity's real
    # division/marker lists in one go.

    def begin_delete_divisor(self) -> bool:
        self._delete_divisor_active = True
        for player in self._players:
            player.enter_delete_divisor_mode()
        return True

    def cancel_delete_divisor(self):
        if not self._delete_divisor_active:
            return
        self._delete_divisor_active = False
        for player in self._players:
            player.exit_delete_divisor_mode()

    def _deselect_all_markers(self):
        for player in self._players:
            player.deselect_marker()

    def clear_divisor_selection(self):
        """Public entry point so WorkspacePage can deselect any
        highlighted divisor line on mode/menu switch, independent of
        whether the Delete Divisor op panel is currently bookkept as
        active."""
        self._deselect_all_markers()

    def apply_delete_divisor(self):
        if not self._delete_divisor_active:
            return
        self._delete_divisor_active = False
        for player in self._players:
            deleted_times = player.commit_delete_divisor()
            if not deleted_times:
                continue
            if player.channel_index is None:
                self._entity.divisions = [
                    t for t in self._entity.divisions if t not in deleted_times
                ]
            else:
                ch = player.channel_index
                if ch in self._entity.channel_markers:
                    self._entity.channel_markers[ch] = [
                        t for t in self._entity.channel_markers[ch] if t not in deleted_times
                    ]
        self.entity_modified.emit()
        self.entity_divisor_deleted.emit()

    # ------------------------------------------------------------------
    # Trim preview
    # ------------------------------------------------------------------
    # Removes the selected segment and closes the gap, keeping everything
    # before and after it. Only valid when the Overall plot is selected
    # (enforced by the caller via _OVERALL_ONLY_OPS), since the operation
    # applies uniformly across every channel.

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

    # ------------------------------------------------------------------
    # Extract preview
    # ------------------------------------------------------------------
    # The inverse of Trim: instead of removing the selected segment and
    # keeping everything else, Extract keeps *only* the selected segment
    # and discards everything before and after it. Only valid when the
    # Overall plot is selected (enforced by the caller via
    # _OVERALL_ONLY_OPS), same restriction as Trim/Time Scale/Concatenate,
    # since the operation applies uniformly across every channel.

    def begin_extract(self) -> bool:
        if self._selected_player is None or self._selected_player.channel_index is not None:
            return False
        segment = self._selected_player.selected_segment
        if segment is None:
            return False

        self._extract_player = self._selected_player
        self._extract_range = segment
        for player in self._players:
            player.enter_extract_preview(*segment)
        self._lock_selection(True)
        return True

    def cancel_extract(self):
        """Exit extract preview, restore original graph, keep selection."""
        if self._extract_player is not None:
            for player in self._players:
                player.exit_extract_preview()
        self._extract_player = None
        self._extract_range = None
        self._lock_selection(False)
        # Restore selection highlight (don't clear the selection)
        if self._selected_player is not None:
            self._selected_player.restore_selection_visual()

    def apply_extract(self):
        if self._extract_player is None or self._extract_range is None:
            return
        start, end = self._extract_range
        self._perform_extract(start, end)
        self._extract_player = None
        self._extract_range = None
        self._lock_selection(False)
        self.clear_all_selection()   # apply clears selection
        self.entity_modified.emit()
        self.entity_extracted.emit()

    def _perform_extract(self, start, end):
        clip = self._entity.clip
        keep_start_idx = clip.get_index(start)
        keep_end_idx = clip.get_index(end)

        new_channels = []
        for ch in clip.channels:
            local_start = max(keep_start_idx, ch.start_index)
            local_end = min(keep_end_idx, ch.end_index())

            if local_end < local_start:
                # selection fell entirely outside this channel's range —
                # keep a single silent sample so the clip never collapses
                # to zero length
                new_channels.append(Discrete_Signal(
                    np.zeros(1, dtype=ch.samples.dtype), ch.sample_rate, ch.start_index
                ))
            else:
                new_channels.append(ch.trim(local_start, local_end))

        self._entity.clip = AudioClip(new_channels, name=clip.name)

        # Sample indices (and therefore times) inside the kept segment are
        # unchanged — only content outside [start, end] is discarded — so
        # divisions/markers just need filtering to what still falls
        # strictly inside the kept range, with no time-shift required.
        self._entity.divisions = [
            t for t in self._entity.divisions if start < t < end
        ]
        for ch_idx in list(self._entity.channel_markers.keys()):
            self._entity.channel_markers[ch_idx] = [
                t for t in self._entity.channel_markers[ch_idx] if start < t < end
            ]

    # ------------------------------------------------------------------
    # Noise removal
    # ------------------------------------------------------------------
    # Two phases:
    #   1. Selection  — only the Overall plot is shown, with no play
    #      button and no white lines, just two draggable green handles
    #      marking the noise window. Ends in Filter (-> phase 2) or
    #      Cancel (closes the panel entirely).
    #   2. Filtered   — every normal player is hidden and replaced by a
    #      tab strip (Channel 1/Channel 2/Overall/Noise, or just
    #      Overall/Noise for mono) where each tab shows a Previous/
    #      Filtered pair of read-only players. Ends in Apply (commits
    #      the filtered clip) or Cancel (returns to phase 1, keeping the
    #      previously chosen range).

    def begin_noise_removal(self) -> bool:
        if self._noise_removal_active:
            return False
        overall = self._find_overall_player()
        if overall is None:
            return False

        self._noise_removal_active = True
        self._noise_filtered = False
        self._noise_from_entity = False
        self._noise_overall_player = overall
        self._noise_range = None
        self._noise_original_clip = None
        self._noise_filtered_clip = None

        for player in self._players:
            player.setVisible(player is overall)

        clip = self._entity.clip
        start_time = clip.get_time(clip.start_index)
        end_time = clip.get_time(clip.end_index())
        span = end_time - start_time
        default_start = start_time + span * 0.25
        default_end = start_time + span * 0.75

        overall.enter_noise_range_mode(default_start, default_end)
        self._lock_selection(True)
        return True

    def begin_noise_removal_from_entity(self, source_entity, source_channel_index) -> bool:
        """Noise-source == another entity's channel: no range-selection
        stage — the whole reference channel is used as the noise
        profile, and the view opens straight into the filtered/Apply
        phase (see _build_noise_tabs / noise_removal_is_from_entity)."""
        if self._noise_removal_active:
            return False
        overall = self._find_overall_player()
        if overall is None:
            return False
        if source_entity is None:
            return False

        source_channels = source_entity.clip.channels
        if source_channel_index is None:
            noise_signal = source_channels[0]
        elif 0 <= source_channel_index < len(source_channels):
            noise_signal = source_channels[source_channel_index]
        else:
            return False

        self._noise_removal_active = True
        self._noise_filtered = True
        self._noise_from_entity = True
        self._noise_reference_signal = noise_signal
        self._noise_overall_player = overall
        self._noise_range = None

        clip = self._entity.clip
        self._noise_original_clip = clip

        filtered = clip
        for _ in range(_NOISE_REMOVAL_ITERATIONS):
            for ch_idx in range(clip.num_channels):
                filtered = filtered.apply(
                    "remove_noise", channel=ch_idx, noise_reference=noise_signal
                )
            QApplication.processEvents()
        self._noise_filtered_clip = filtered

        for player in self._players:
            player.setVisible(False)

        self._build_noise_tabs()
        self._lock_selection(True)
        return True

    @property
    def noise_removal_is_from_entity(self) -> bool:
        return self._noise_from_entity

    def cancel_noise_removal(self):
        """Fully closes the panel, from either phase."""
        if not self._noise_removal_active:
            return
        if self._noise_filtered:
            self._teardown_noise_tabs()
        elif self._noise_overall_player is not None:
            self._noise_overall_player.exit_noise_range_mode()
        self._restore_normal_players()
        self._reset_noise_state()

    def back_to_noise_selection(self):
        """Cancel pressed during the Filter/Apply phase: returns to the
        range-selection phase (with the previous range restored) instead
        of closing the whole panel."""
        if not self._noise_removal_active or not self._noise_filtered:
            return

        saved_range = self._noise_range
        overall = self._noise_overall_player

        self._teardown_noise_tabs()
        for player in self._players:
            player.setVisible(player is overall)

        self._noise_filtered = False
        self._noise_original_clip = None
        self._noise_filtered_clip = None

        overall.enter_noise_range_mode(*saved_range)

    def filter_noise(self):
        if not self._noise_removal_active or self._noise_filtered:
            return

        overall = self._noise_overall_player
        start, end = overall.get_noise_range()
        self._noise_range = (start, end)

        clip = self._entity.clip
        self._noise_original_clip = clip

        start_idx = clip.get_index(start)
        end_idx = clip.get_index(end)

        filtered = clip
        for _ in range(_NOISE_REMOVAL_ITERATIONS):
            for ch_idx in range(clip.num_channels):
                filtered = filtered.apply("remove_noise", start_idx, end_idx, channel=ch_idx)
            QApplication.processEvents()
        self._noise_filtered_clip = filtered

        overall.exit_noise_range_mode()
        for player in self._players:
            player.setVisible(False)

        self._noise_filtered = True
        self._build_noise_tabs()

    def apply_noise_removal(self):
        if not self._noise_removal_active or not self._noise_filtered:
            return
        self._entity.clip = self._noise_filtered_clip
        self._teardown_noise_tabs()
        self._restore_normal_players()
        self._reset_noise_state()
        self.entity_modified.emit()
        self.entity_noise_removed.emit()

    def _restore_normal_players(self):
        for player in self._players:
            player.setVisible(True)
        self._lock_selection(False)

    def _reset_noise_state(self):
        self._noise_removal_active = False
        self._noise_filtered = False
        self._noise_from_entity = False
        self._noise_reference_signal = None
        self._noise_overall_player = None
        self._noise_range = None
        self._noise_original_clip = None
        self._noise_filtered_clip = None

    @staticmethod
    def _channels_to_series_and_audio(channels, only_index=None):
        """Mirrors _build_ui's per-channel / Overall series+audio setup,
        for the read-only preview players in the Noise tab strip."""
        if only_index is not None:
            ch = channels[only_index]
            color = _CHANNEL_COLORS[only_index % len(_CHANNEL_COLORS)]
            series = [(ch.samples, color, f"Ch {only_index + 1}")]
            audio = ch.samples.astype(np.float32)
            return series, audio

        series = [
            (ch.samples, _CHANNEL_COLORS[i % len(_CHANNEL_COLORS)], f"Ch {i + 1}")
            for i, ch in enumerate(channels)
        ]
        stacked = np.stack([ch.samples for ch in channels], axis=1).astype(np.float32)
        audio = stacked if stacked.shape[1] <= 2 else stacked.mean(axis=1).astype(np.float32)
        return series, audio

    def _build_noise_tabs(self):
        clip = self._entity.clip
        num_channels = clip.num_channels

        tabs = []
        if num_channels > 1:
            for i in range(num_channels):
                tabs.append((f"Channel {i + 1}", i))
        tabs.append(("Overall", None))
        tabs.append(("Noise", "noise"))

        if self._noise_range is not None:
            noise_start_idx = clip.get_index(self._noise_range[0])
            noise_end_idx = clip.get_index(self._noise_range[1])
        else:
            noise_start_idx = None
            noise_end_idx = None

        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        tab_row = QHBoxLayout()
        tab_row.setSpacing(6)
        stack = QStackedWidget()

        self._noise_tab_buttons = {}
        self._noise_tab_pages = {}
        self._noise_preview_players = []

        for label, key in tabs:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked=False, k=key: self._select_noise_tab(k))
            tab_row.addWidget(btn)
            self._noise_tab_buttons[key] = btn

            page = self._build_noise_tab_page(key, noise_start_idx, noise_end_idx)
            stack.addWidget(page)
            self._noise_tab_pages[key] = page

        tab_row.addStretch()
        outer.addLayout(tab_row)
        outer.addWidget(stack)

        self._noise_tab_stack = stack
        self._noise_tab_widget = container
        self._content_layout.insertWidget(0, container)

        self._select_noise_tab(tabs[0][1])

    def _select_noise_tab(self, key):
        for k, btn in self._noise_tab_buttons.items():
            btn.setChecked(k == key)
        self._noise_tab_stack.setCurrentWidget(self._noise_tab_pages[key])

    def _build_noise_tab_page(self, key, noise_start_idx, noise_end_idx):
        orig_clip = self._noise_original_clip
        filt_clip = self._noise_filtered_clip

        if key == "noise" and self._noise_reference_signal is not None:
            return self._build_noise_reference_page()

        if key == "noise":
            prev_channels = [ch.trim(noise_start_idx, noise_end_idx) for ch in orig_clip.channels]
            only_index = None
        else:
            prev_channels = orig_clip.channels
            only_index = key if isinstance(key, int) else None

        times = self._time_axis(prev_channels[0] if only_index is None else prev_channels[only_index])

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        if key == "noise":
            # For noise window: show only Noise Reference
            label = QLabel("Noise Reference")
            label.setStyleSheet("font-size: 13px; font-weight: 600;")
            layout.addWidget(label)

            series, audio = self._channels_to_series_and_audio(prev_channels, only_index)

            player = WaveformPlayer(
                times=times,
                series=series,
                sample_rate=prev_channels[0].sample_rate,
                audio_data=audio,
                group=self._group,
                entity=self._entity,
                channel_index=None,
                is_driver=False,
                show_markers=False,
            )
            player.set_tool_mode(True)
            layout.addWidget(player)
            self._noise_preview_players.append(player)
        else:
            # For channel/overall: show Previous and Filtered
            filt_channels = filt_clip.channels

            for title, channels in (("Previous Signal", prev_channels), ("Filtered Signal", filt_channels)):
                series, audio = self._channels_to_series_and_audio(channels, only_index)

                label = QLabel(title)
                label.setStyleSheet("font-size: 13px; font-weight: 600;")
                layout.addWidget(label)

                player = WaveformPlayer(
                    times=times,
                    series=series,
                    sample_rate=channels[0].sample_rate,
                    audio_data=audio,
                    group=self._group,
                    entity=self._entity,
                    channel_index=None,
                    is_driver=False,
                    show_markers=False,
                )
                player.set_tool_mode(True)
                layout.addWidget(player)
                self._noise_preview_players.append(player)

        return page

    def _build_noise_reference_page(self):
        """Noise tab page for the entity-channel source: shows only the 
        reference noise entity (Noise Reference)."""
        ref = self._noise_reference_signal
        times = self._time_axis(ref)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        label = QLabel("Noise Reference")
        label.setStyleSheet("font-size: 13px; font-weight: 600;")
        layout.addWidget(label)

        player = WaveformPlayer(
            times=times,
            series=[(ref.samples, _CHANNEL_COLORS[0], "Noise Profile")],
            sample_rate=ref.sample_rate,
            audio_data=ref.samples.astype(np.float32),
            group=self._group,
            entity=self._entity,
            channel_index=None,
            is_driver=False,
            show_markers=False,
        )
        player.set_tool_mode(True)
        layout.addWidget(player)
        self._noise_preview_players.append(player)

        return page

    def _teardown_noise_tabs(self):
        if self._noise_tab_widget is None:
            return
        for player in self._noise_preview_players:
            player.force_idle()
        self._content_layout.removeWidget(self._noise_tab_widget)
        self._noise_tab_widget.deleteLater()
        self._noise_tab_widget = None
        self._noise_tab_stack = None
        self._noise_tab_buttons = {}
        self._noise_tab_pages = {}
        self._noise_preview_players = []

    # ------------------------------------------------------------------
    # Time-scale preview
    # ------------------------------------------------------------------

    def begin_timescale(self) -> bool:
        """Enter timescale preview. Only valid when the overall plot is
        selected with a segment chosen — the factor applies to that
        segment only, not the whole clip."""
        if self._selected_player is None or self._selected_player.channel_index is not None:
            return False
        segment = self._selected_player.selected_segment
        if segment is None:
            return False
        self._timescale_preview_active = True
        self._timescale_range = segment
        for player in self._players:
            player.enter_timescale_preview(1.0, *segment)
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
        self._timescale_range = None
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
        start, end = self._timescale_range
        self._timescale_range = None
        self._perform_timescale(factor, start, end)
        self._lock_selection(False)
        self.clear_all_selection()   # apply clears selection
        self.entity_modified.emit()
        self.entity_timescaled.emit()

    def _perform_timescale(self, factor: float, start, end):
        clip = self._entity.clip
        start_idx = clip.get_index(start)
        end_idx = clip.get_index(end)

        new_channels = []
        added_len = 0
        for ch in clip.channels:
            pieces = []
            if start_idx > ch.start_index:
                pieces.append(ch.trim(ch.start_index, start_idx - 1))

            seg_end_idx = min(end_idx, ch.end_index())
            if seg_end_idx >= start_idx:
                segment = ch.trim(start_idx, seg_end_idx).time_scale(factor)
                added_len = len(segment.samples)
                pieces.append(segment)

            if end_idx < ch.end_index():
                pieces.append(ch.trim(end_idx + 1, ch.end_index()))

            if not pieces:
                pieces = [Discrete_Signal(
                    np.zeros(1, dtype=ch.samples.dtype), ch.sample_rate, ch.start_index
                )]

            merged = pieces[0]
            for piece in pieces[1:]:
                merged = merged.concatenate(piece)
            new_channels.append(merged)

        self._entity.clip = AudioClip(new_channels, name=clip.name)

        removed_span = end - start
        new_seg_duration = added_len / clip.sample_rate
        delta = new_seg_duration - removed_span

        def _shift_time(t):
            if t <= start:
                return t
            if t >= end:
                return t + delta
            return start + (t - start) * factor   # marker fell inside the scaled segment

        self._entity.divisions = [_shift_time(t) for t in self._entity.divisions]
        for ch_idx in list(self._entity.channel_markers.keys()):
            self._entity.channel_markers[ch_idx] = [
                _shift_time(t) for t in self._entity.channel_markers[ch_idx]
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
    # Reverse preview
    # ------------------------------------------------------------------
    # Same architecture as vertical scale: a preview lives on the selected
    # player and, when a channel is selected, mirrors onto that channel's
    # curve in the Overall plot; when Overall is selected, mirrors onto
    # every channel plot. Unlike vertical scale, reverse takes no factor,
    # so there's no live-update step — the preview is computed once when
    # the op begins.

    def begin_reverse(self) -> bool:
        if self._selected_player is None:
            return False
        segment = self._selected_player.selected_segment
        if segment is None:
            return False

        self._reverse_player = self._selected_player
        self._reverse_range = segment
        self._reverse_player.enter_effect_preview("reverse", *segment)

        self._reverse_overall_target = None
        self._reverse_mirror_players = []

        if self._reverse_player.channel_index is not None:
            # A single channel was selected — mirror the preview onto that
            # channel's curve inside the Overall plot.
            overall_player = self._find_overall_player()
            if overall_player is not None and overall_player is not self._reverse_player:
                overall_player.preview_channel_curve_effect(
                    self._reverse_player.channel_index, "reverse", *segment
                )
                self._reverse_overall_target = overall_player
        else:
            # The Overall plot was selected — it applies the effect to
            # every channel uniformly, so mirror the same preview onto
            # each channel plot.
            self._reverse_mirror_players = [
                p for p in self._players
                if p is not self._reverse_player and p.channel_index is not None
            ]
            for player in self._reverse_mirror_players:
                player.enter_effect_preview("reverse", *segment)

        self._lock_selection(True)
        return True

    def cancel_reverse(self):
        if self._reverse_player is None:
            return
        self._reverse_player.exit_effect_preview()
        if self._reverse_overall_target is not None:
            self._reverse_overall_target.exit_channel_curve_preview()
        for player in self._reverse_mirror_players:
            player.exit_effect_preview()
        self._reverse_player = None
        self._reverse_overall_target = None
        self._reverse_mirror_players = []
        self._reverse_range = None
        self._lock_selection(False)
        if self._selected_player is not None:
            self._selected_player.restore_selection_visual()

    def apply_reverse(self):
        if self._reverse_player is None:
            return
        channel_index = self._reverse_player.channel_index
        start, end = self._reverse_range
        self._reverse_player.exit_effect_preview()
        if self._reverse_overall_target is not None:
            self._reverse_overall_target.exit_channel_curve_preview()
        for player in self._reverse_mirror_players:
            player.exit_effect_preview()
        self._perform_reverse(channel_index, start, end)
        self._reverse_player = None
        self._reverse_overall_target = None
        self._reverse_mirror_players = []
        self._reverse_range = None
        self._lock_selection(False)
        self.clear_all_selection()
        self.entity_modified.emit()
        self.entity_reversed.emit()

    def _perform_reverse(self, channel_index, start, end):
        clip = self._entity.clip
        start_idx = clip.get_index(start)
        end_idx = clip.get_index(end)

        if channel_index is None:
            new_clip = clip.apply("reverse", start_idx, end_idx)
        else:
            new_clip = clip.apply("reverse", start_idx, end_idx, channel=channel_index)

        self._entity.clip = new_clip

    # ------------------------------------------------------------------
    # Fade In preview
    # ------------------------------------------------------------------
    # Same architecture as Reverse: a preview lives on the selected player
    # and mirrors onto the Overall plot's matching channel curve (channel
    # selected) or onto every channel plot (Overall selected). No factor,
    # so the preview is computed once when the op begins.

    def begin_fade_in(self) -> bool:
        if self._selected_player is None:
            return False
        segment = self._selected_player.selected_segment
        if segment is None:
            return False

        self._fadein_player = self._selected_player
        self._fadein_range = segment
        self._fadein_player.enter_effect_preview("fade_in", *segment)

        self._fadein_overall_target = None
        self._fadein_mirror_players = []

        if self._fadein_player.channel_index is not None:
            overall_player = self._find_overall_player()
            if overall_player is not None and overall_player is not self._fadein_player:
                overall_player.preview_channel_curve_effect(
                    self._fadein_player.channel_index, "fade_in", *segment
                )
                self._fadein_overall_target = overall_player
        else:
            self._fadein_mirror_players = [
                p for p in self._players
                if p is not self._fadein_player and p.channel_index is not None
            ]
            for player in self._fadein_mirror_players:
                player.enter_effect_preview("fade_in", *segment)

        self._lock_selection(True)
        return True

    def cancel_fade_in(self):
        if self._fadein_player is None:
            return
        self._fadein_player.exit_effect_preview()
        if self._fadein_overall_target is not None:
            self._fadein_overall_target.exit_channel_curve_preview()
        for player in self._fadein_mirror_players:
            player.exit_effect_preview()
        self._fadein_player = None
        self._fadein_overall_target = None
        self._fadein_mirror_players = []
        self._fadein_range = None
        self._lock_selection(False)
        if self._selected_player is not None:
            self._selected_player.restore_selection_visual()

    def apply_fade_in(self):
        if self._fadein_player is None:
            return
        channel_index = self._fadein_player.channel_index
        start, end = self._fadein_range
        self._fadein_player.exit_effect_preview()
        if self._fadein_overall_target is not None:
            self._fadein_overall_target.exit_channel_curve_preview()
        for player in self._fadein_mirror_players:
            player.exit_effect_preview()
        self._perform_fade_in(channel_index, start, end)
        self._fadein_player = None
        self._fadein_overall_target = None
        self._fadein_mirror_players = []
        self._fadein_range = None
        self._lock_selection(False)
        self.clear_all_selection()
        self.entity_modified.emit()
        self.entity_faded_in.emit()

    def _perform_fade_in(self, channel_index, start, end):
        clip = self._entity.clip
        start_idx = clip.get_index(start)
        end_idx = clip.get_index(end)

        if channel_index is None:
            new_clip = clip.apply("fade_in", start_idx, end_idx)
        else:
            new_clip = clip.apply("fade_in", start_idx, end_idx, channel=channel_index)

        self._entity.clip = new_clip

    # ------------------------------------------------------------------
    # Fade Out preview
    # ------------------------------------------------------------------
    # Same architecture as Reverse: a preview lives on the selected player
    # and mirrors onto the Overall plot's matching channel curve (channel
    # selected) or onto every channel plot (Overall selected). No factor,
    # so the preview is computed once when the op begins.

    def begin_fade_out(self) -> bool:
        if self._selected_player is None:
            return False
        segment = self._selected_player.selected_segment
        if segment is None:
            return False

        self._fadeout_player = self._selected_player
        self._fadeout_range = segment
        self._fadeout_player.enter_effect_preview("fade_out", *segment)

        self._fadeout_overall_target = None
        self._fadeout_mirror_players = []

        if self._fadeout_player.channel_index is not None:
            overall_player = self._find_overall_player()
            if overall_player is not None and overall_player is not self._fadeout_player:
                overall_player.preview_channel_curve_effect(
                    self._fadeout_player.channel_index, "fade_out", *segment
                )
                self._fadeout_overall_target = overall_player
        else:
            self._fadeout_mirror_players = [
                p for p in self._players
                if p is not self._fadeout_player and p.channel_index is not None
            ]
            for player in self._fadeout_mirror_players:
                player.enter_effect_preview("fade_out", *segment)

        self._lock_selection(True)
        return True

    def cancel_fade_out(self):
        if self._fadeout_player is None:
            return
        self._fadeout_player.exit_effect_preview()
        if self._fadeout_overall_target is not None:
            self._fadeout_overall_target.exit_channel_curve_preview()
        for player in self._fadeout_mirror_players:
            player.exit_effect_preview()
        self._fadeout_player = None
        self._fadeout_overall_target = None
        self._fadeout_mirror_players = []
        self._fadeout_range = None
        self._lock_selection(False)
        if self._selected_player is not None:
            self._selected_player.restore_selection_visual()

    def apply_fade_out(self):
        if self._fadeout_player is None:
            return
        channel_index = self._fadeout_player.channel_index
        start, end = self._fadeout_range
        self._fadeout_player.exit_effect_preview()
        if self._fadeout_overall_target is not None:
            self._fadeout_overall_target.exit_channel_curve_preview()
        for player in self._fadeout_mirror_players:
            player.exit_effect_preview()
        self._perform_fade_out(channel_index, start, end)
        self._fadeout_player = None
        self._fadeout_overall_target = None
        self._fadeout_mirror_players = []
        self._fadeout_range = None
        self._lock_selection(False)
        self.clear_all_selection()
        self.entity_modified.emit()
        self.entity_faded_out.emit()

    def _perform_fade_out(self, channel_index, start, end):
        clip = self._entity.clip
        start_idx = clip.get_index(start)
        end_idx = clip.get_index(end)

        if channel_index is None:
            new_clip = clip.apply("fade_out", start_idx, end_idx)
        else:
            new_clip = clip.apply("fade_out", start_idx, end_idx, channel=channel_index)

        self._entity.clip = new_clip

    # ------------------------------------------------------------------
    # Echo
    # ------------------------------------------------------------------
    # Tool operation: applies echo effect to the entire entity clip with
    # live preview. As parameters (occurrence, delay, decay) change in the
    # sidebar, the preview updates instantly showing both original and
    # echoed versions in the waveform players.

    def begin_echo(self) -> bool:
        """Start echo preview mode."""
        if self._echo_original_clip is not None:
            return False

        clip = self._entity.clip
        self._echo_original_clip = clip.copy()
        self._echo_parameters = (2, 1.0, 0.0)  # default: occurrence=2, delay=1s, decay=0
        self._lock_selection(True)
        return True

    def update_echo_preview(self, occurrence: int, delay: float, decay: float):
        """Update the preview as echo parameters change. Shows both original
        and echoed versions overlaid in the waveform players."""
        if self._echo_original_clip is None:
            return

        self._echo_parameters = (occurrence, delay, decay)
        
        try:
            # Apply the same operation to all channels at once so their
            # resulting lengths remain identical.
            preview_clip = self._echo_original_clip.apply(
                "echo", occurrence, delay, decay
            )
            
            # Update entity with preview clip
            self._entity.clip = preview_clip
            # Emit signal to trigger UI refresh
            self.entity_echo_added.emit()
        except Exception as e:
            # If parameters are invalid, keep original
            print(f"Echo preview error: {e}")
            if self._entity.clip != self._echo_original_clip:
                self._entity.clip = self._echo_original_clip.copy()
                self.entity_echo_added.emit()

    def apply_echo(self):
        """Apply the echo effect to the entity."""
        if self._echo_original_clip is None or self._echo_parameters is None:
            return

        # Clip already has preview, just finalize it
        self._reset_echo_state()
        self._lock_selection(False)
        self.clear_all_selection()
        self.entity_modified.emit()
        self.entity_echo_added.emit()

    def cancel_echo(self):
        """Cancel the echo effect and restore original."""
        if self._echo_original_clip is None:
            return
        
        # Restore original clip
        self._entity.clip = self._echo_original_clip.copy()
        self._reset_echo_state()
        self._lock_selection(False)
        self.clear_all_selection()
        self.entity_echo_added.emit()

    def _reset_echo_state(self):
        """Reset echo-related state variables."""
        self._echo_original_clip = None
        self._echo_parameters = None
    # Only valid when the Overall plot is selected (enforced by the caller
    # via _OVERALL_ONLY_OPS). Each chosen source is a division-bounded
    # *segment* of some entity's Overall plot (not necessarily the whole
    # clip) — all of which must share this entity's channel count. They're
    # appended, in order, right after the end of the current selection on
    # the Overall plot. Like vertical scale / effects, the Overall preview
    # mirrors onto every channel plot since every channel is affected
    # uniformly.

    def begin_concatenate(self, source_portions) -> bool:
        """source_portions: ordered list of (Entity, segment_start, segment_end)."""
        if self._selected_player is None or self._selected_player.channel_index is not None:
            return False
        segment = self._selected_player.selected_segment
        if segment is None:
            return False
        if not source_portions:
            return False

        clip = self._entity.clip
        target_channels = clip.num_channels
        for src_entity, _seg_start, _seg_end in source_portions:
            if src_entity.clip.num_channels != target_channels:
                return False

        insert_time = segment[1]
        target_rate = clip.sample_rate

        extra_channel_samples = []
        for ch_idx in range(target_channels):
            pieces = []
            for src_entity, seg_start, seg_end in source_portions:
                src_ch = src_entity.clip.channels[ch_idx]
                start_idx = max(src_ch.get_index(seg_start), src_ch.start_index)
                end_idx = min(src_ch.get_index(seg_end), src_ch.end_index())
                if end_idx < start_idx:
                    end_idx = start_idx
                segment_signal = src_ch.trim(start_idx, end_idx)
                resampled = (
                    segment_signal if segment_signal.sample_rate == target_rate
                    else segment_signal.resample(target_rate)
                )
                pieces.append(resampled.samples)
            extra_channel_samples.append(
                np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float64)
            )

        self._concat_player = self._selected_player
        self._concat_range = segment
        self._concat_sources = list(source_portions)
        self._concat_extra_channel_samples = extra_channel_samples

        for player in self._players:
            if player.channel_index is None:
                player.enter_concat_preview(insert_time, extra_channel_samples)
            else:
                player.enter_concat_preview(insert_time, [extra_channel_samples[player.channel_index]])

        self._lock_selection(True)
        return True

    def cancel_concatenate(self):
        if self._concat_player is None:
            return
        for player in self._players:
            player.exit_concat_preview()
        self._concat_player = None
        self._concat_range = None
        self._concat_sources = []
        self._concat_extra_channel_samples = None
        self._lock_selection(False)
        if self._selected_player is not None:
            self._selected_player.restore_selection_visual()

    def apply_concatenate(self):
        if self._concat_player is None or self._concat_range is None:
            return
        insert_time = self._concat_range[1]
        for player in self._players:
            player.exit_concat_preview()
        self._perform_concatenate(insert_time)
        self._concat_player = None
        self._concat_range = None
        self._concat_sources = []
        self._concat_extra_channel_samples = None
        self._lock_selection(False)
        self.clear_all_selection()   # apply clears selection
        self.entity_modified.emit()
        self.entity_concatenated.emit()

    def _perform_concatenate(self, insert_time):
        clip = self._entity.clip
        insert_idx = clip.get_index(insert_time)
        extra = self._concat_extra_channel_samples

        new_channels = []
        added_len = 0
        for ch_idx, ch in enumerate(clip.channels):
            extra_samples = extra[ch_idx]
            added_len = len(extra_samples)
            extra_signal = Discrete_Signal(extra_samples, ch.sample_rate, ch.start_index)
            new_channels.append(ch.concatenate(extra_signal, index=insert_idx))

        self._entity.clip = AudioClip(new_channels, name=clip.name)

        added_duration = added_len / clip.sample_rate

        def _shift_time(t):
            return t + added_duration if t >= insert_time else t

        self._entity.divisions = [_shift_time(t) for t in self._entity.divisions]
        for ch_idx in list(self._entity.channel_markers.keys()):
            self._entity.channel_markers[ch_idx] = [
                _shift_time(t) for t in self._entity.channel_markers[ch_idx]
            ]

    def _on_canvas_clicked(self):
        # A click on empty canvas is a no-op while any preview is active —
        # every operation (Trim included) is cancelled explicitly via its
        # own Cancel button, not implicitly by clicking away.
        if (self._trim_player is not None
                or self._extract_player is not None
                or self._timescale_preview_active
                or self._vscale_player is not None
                or self._reverse_player is not None
                or self._fadein_player is not None
                or self._fadeout_player is not None
                or self._concat_player is not None
                or self._noise_removal_active):
            return
        if self._delete_divisor_active:
            self._deselect_all_markers()
            return
        self.clear_all_selection()

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
        self._content_layout = layout

    def shutdown(self):
        self.cancel_trim()
        self.cancel_extract()
        self.cancel_delete_divisor()
        self.cancel_noise_removal()
        self.cancel_timescale()
        self.cancel_vertical_scale()
        self.cancel_reverse()
        self.cancel_fade_in()
        self.cancel_fade_out()
        self.cancel_echo()
        self.cancel_concatenate()
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

    def set_tool_mode(self, enabled: bool):
        """Same idea, for the Tool top-level mode."""
        for player in self._players:
            player.set_tool_mode(enabled)

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