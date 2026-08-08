import time

import numpy as np
import pyqtgraph as pg
import sounddevice as sd
from PySide6.QtCore import QObject, QTimer, Signal, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy


_HALF_WINDOW_SECONDS = 3.0   # visible seconds on each side of the playhead while playing
_TICK_MS = 33                # UI refresh while playing (~30 fps)


class PlaybackGroup(QObject):
    """
    Single, app-wide coordinator (singleton — use PlaybackGroup.get_instance()).

    - Enforces "only one plot plays at a time" across every entity tab, not
      just within one tab: once a player is active, every other player's
      Play/Resume button — in this tab and every other open tab — is
      disabled until that player is reset or finishes.
    - Lets the 'Overall' plot (the driver) broadcast its playhead position
      so the per-channel plots in its own tab scroll in sync with it.
    """

    _instance = None

    active_changed = Signal(object)     # WaveformPlayer that became active, or None
    position_broadcast = Signal(float)  # seconds; emitted only by the driver while playing
    clip_active_changed = Signal(object)   # WaveformPlayer in divide-mode, or None

    def __init__(self):
        super().__init__()
        self.active_player = None
        self.clip_active_player = None

    @classmethod
    def get_instance(cls) -> "PlaybackGroup":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def request_play(self, player) -> bool:
        if self.active_player is not None and self.active_player is not player:
            return False
        self.active_player = player
        self.active_changed.emit(player)
        return True

    def release(self, player):
        if self.active_player is player:
            self.active_player = None
            self.active_changed.emit(None)

    def request_clip_mode(self, player) -> bool:
        if self.clip_active_player is not None:
            return False
        self.clip_active_player = player
        self.clip_active_changed.emit(player)
        return True

    def release_clip_mode(self, player):
        if self.clip_active_player is player:
            self.clip_active_player = None
            self.clip_active_changed.emit(None)

    def broadcast_position(self, seconds):
        self.position_broadcast.emit(seconds)

class _PlaybackEngine:
    """Play / pause / resume / stop one audio buffer via sounddevice."""

    def __init__(self, audio_data: np.ndarray, sample_rate: int):
        self._audio_data = audio_data              # 1-D (mono) or 2-D (frames, ch) float32
        self._sample_rate = sample_rate
        self._played_frames = 0
        self._segment_start = None                 # time.monotonic() at last (re)start

    @property
    def total_frames(self):
        return self._audio_data.shape[0]

    @property
    def audio_data(self):
        return self._audio_data

    def start_or_resume(self):
        remaining = self._audio_data[self._played_frames:]
        if len(remaining) == 0:
            return
        sd.play(remaining, samplerate=self._sample_rate)
        self._segment_start = time.monotonic()

    def pause(self):
        self._played_frames = self.current_frame()
        sd.stop()
        self._segment_start = None

    def stop(self):
        sd.stop()
        self._segment_start = None
        self._played_frames = 0

    def current_frame(self):
        if self._segment_start is None:
            return self._played_frames
        elapsed = time.monotonic() - self._segment_start
        frame = self._played_frames + int(elapsed * self._sample_rate)
        return min(frame, self.total_frames)

    def is_finished(self):
        return self.current_frame() >= self.total_frames

    def seek(self, frame):
        frame = max(0, min(frame, self.total_frames))
        was_playing = self._segment_start is not None
        sd.stop()
        self._played_frames = frame
        if was_playing:
            self.start_or_resume()
        else:
            self._segment_start = None

class _DraggablePlotWidget(pg.PlotWidget):
    """Supports both: (a) relative image-style panning during playback/pause,
    and (b) divide-mode, where a green divisor line can be dragged (with
    magnetic hit-tolerance) and, if allowed, the graph can still be panned."""

    _DIVISOR_HIT_TOLERANCE_PX = 12

    def __init__(self, parent=None):
        super().__init__(parent)
        self.drag_enabled = False
        self.on_drag_start = None
        self.on_seek = None
        self.on_drag_end = None
        self._dragging = False
        self._last_x = None
        self._pixels_per_second = None

        self._clip_mode = False
        self._clip_allow_graph_drag = False
        self._divisor_getter = None
        self._divisor_setter = None
        self._time_to_pixel = None
        self._dragging_divisor = False
        self._clip_mode = False
        self._clip_allow_graph_drag = False
        self._divisor_getter = None
        self._divisor_setter = None
        self._time_to_pixel = None
        self._dragging_divisor = False

        self._segment_mode = False
        self._segment_boundaries_getter = None
        self._on_segment_hover = None
        self._on_segment_click = None
        self.setMouseTracking(True)

    def set_clip_mode(self, enabled, allow_graph_drag=False,
                       divisor_getter=None, divisor_setter=None, time_to_pixel=None):
        self._clip_mode = enabled
        self._clip_allow_graph_drag = allow_graph_drag
        self._divisor_getter = divisor_getter
        self._divisor_setter = divisor_setter
        self._time_to_pixel = time_to_pixel
        self._dragging_divisor = False
        self._dragging = False

    def set_segment_mode(self, enabled, boundaries_getter=None, on_hover=None, on_click=None):
        self._segment_mode = enabled
        self._segment_boundaries_getter = boundaries_getter
        self._on_segment_hover = on_hover
        self._on_segment_click = on_click

    def _segment_at_pixel(self, event_pos):
        if not self._segment_boundaries_getter:
            return None
        boundaries = self._segment_boundaries_getter()
        if len(boundaries) < 2:
            return None
        view_box = self.getPlotItem().vb
        scene_pos = self.mapToScene(event_pos)
        t = view_box.mapSceneToView(scene_pos).x()
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= t <= boundaries[i + 1]:
                return i, boundaries[i], boundaries[i + 1]
        return None

    def _compute_pixels_per_second(self):
        view_range = self.getPlotItem().vb.viewRange()[0]
        span = view_range[1] - view_range[0]
        width = max(1, self.width())
        return width / span if span > 0 else 1.0

    def _near_divisor(self, event_x):
        if not self._divisor_getter or not self._time_to_pixel:
            return False
        divisor_px = self._time_to_pixel(self._divisor_getter())
        return abs(event_x - divisor_px) <= self._DIVISOR_HIT_TOLERANCE_PX

    def mousePressEvent(self, event):
        if self._clip_mode and event.button() == Qt.LeftButton:
            if self._near_divisor(event.pos().x()):
                self._dragging_divisor = True
                event.accept()
                return
            if self._clip_allow_graph_drag and self.drag_enabled:
                self._dragging = True
                self._last_x = event.pos().x()
                self._pixels_per_second = self._compute_pixels_per_second()
                if self.on_drag_start:
                    self.on_drag_start()
            event.accept()
            return

        if self.drag_enabled and event.button() == Qt.LeftButton:
            self._dragging = True
            self._last_x = event.pos().x()
            self._pixels_per_second = self._compute_pixels_per_second()
            if self.on_drag_start:
                self.on_drag_start()
            event.accept()
            return

        if self._segment_mode and event.button() == Qt.LeftButton:
            result = self._segment_at_pixel(event.pos())
            if result and self._on_segment_click:
                self._on_segment_click(result[0], result[1], result[2])
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._clip_mode:
            if self._dragging_divisor and self._divisor_setter:
                view_box = self.getPlotItem().vb
                scene_pos = self.mapToScene(event.pos())
                t = view_box.mapSceneToView(scene_pos).x()
                self._divisor_setter(t)
                event.accept()
                return
            if self._dragging and self._clip_allow_graph_drag:
                dx = event.pos().x() - self._last_x
                self._last_x = event.pos().x()
                if self.on_seek and self._pixels_per_second:
                    self.on_seek(-dx / self._pixels_per_second)
                event.accept()
                return
            event.accept()
            return

        if self.drag_enabled and self._dragging:
            dx = event.pos().x() - self._last_x
            self._last_x = event.pos().x()
            if self.on_seek and self._pixels_per_second:
                self.on_seek(dx / self._pixels_per_second)
            event.accept()
            return

        if self._segment_mode:
            result = self._segment_at_pixel(event.pos())
            if self._on_segment_hover:
                self._on_segment_hover(result[1] if result else None,
                                        result[2] if result else None)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._clip_mode and event.button() == Qt.LeftButton:
            was_graph_drag = self._dragging and self._clip_allow_graph_drag and not self._dragging_divisor
            self._dragging_divisor = False
            self._dragging = False
            self._last_x = None
            self._pixels_per_second = None
            if was_graph_drag and self.on_drag_end:
                self.on_drag_end()
            event.accept()
            return

        if self.drag_enabled and event.button() == Qt.LeftButton:
            self._dragging = False
            self._last_x = None
            self._pixels_per_second = None
            if self.on_drag_end:
                self.on_drag_end()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        if self._segment_mode and self._on_segment_hover:
            self._on_segment_hover(None, None)
        super().leaveEvent(event)


class WaveformPlayer(QWidget):
    STATE_STOPPED = "stopped"
    STATE_PLAYING = "playing"
    STATE_PAUSED = "paused"

    marker_added = Signal()
    segment_selected = Signal(object, float, float)   # self, start_time, end_time
    segment_deselected = Signal(object)

    def __init__(
        self,
        times: np.ndarray,
        series: list,
        sample_rate: int,
        audio_data: np.ndarray,
        group: PlaybackGroup,
        entity,
        channel_index=None,       # None = Overall plot
        is_driver: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._times = times
        self._series = series
        self._sample_rate = sample_rate
        self._group = group
        self._is_driver = is_driver
        self._entity = entity
        self._channel_index = channel_index

        self._start_time = float(times[0]) if len(times) else 0.0
        self._end_time = float(times[-1]) if len(times) else 0.0

        self._engine = _PlaybackEngine(audio_data, sample_rate)
        self._state = self.STATE_STOPPED

        self._clip_mode = False
        self._divisor_time = None
        self._entered_clip_from_pause = False
        self._edit_mode = False 
        self._selected_segment = None
        self._segment_locked = False

        self._trim_mode = False
        self._trim_engine = None
        self._trim_start = None
        self._trim_end = None

        self._timescale_mode = False
        self._vscale_mode = False
        self._vscale_range = None
        self._preview_curve_index = None
        self._preview_curve_range = None

        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._on_tick)

        self._build_ui()
        self._show_overview()

        self._group.active_changed.connect(self._on_group_active_changed)
        self._group.position_broadcast.connect(self._on_driver_position)
        self._group.clip_active_changed.connect(self._on_group_clip_active_changed)

        self._on_group_active_changed(self._group.active_player)
        self._on_group_clip_active_changed(self._group.clip_active_player)
        self._refresh_clip_enabled()
        self._refresh_segment_mode()

    # ---------- UI ----------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._plot = _DraggablePlotWidget()
        self._plot.on_drag_start = self._on_drag_start
        self._plot.on_seek = self._on_seek
        self._plot.on_drag_end = self._on_drag_end
        self._plot.setBackground(None)
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setYRange(-1.05, 1.05)
        self._plot.setLabel("bottom", "Time", "s")
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._plot.setMinimumHeight(150)

        self._curve_items = []
        for samples, color, _label in self._series:
            curve = self._plot.plot(self._times, samples, pen=pg.mkPen(color=color, width=1))
            self._curve_items.append(curve)

        self._playhead = pg.InfiniteLine(angle=90, pen=pg.mkPen(color="#ff4d4d", width=1.5))
        self._playhead.addMarker('^', position=0.0, size=14)
        self._playhead.addMarker('v', position=1.0, size=14)
        self._playhead.setVisible(False)
        self._plot.addItem(self._playhead)

        self._start_boundary = pg.InfiniteLine(pos=self._start_time, angle=90,
                                                 pen=pg.mkPen(color="#ffffff", width=1))
        self._end_boundary = pg.InfiniteLine(pos=self._end_time, angle=90,
                                              pen=pg.mkPen(color="#ffffff", width=1))
        self._plot.addItem(self._start_boundary)
        self._plot.addItem(self._end_boundary)

        self._marker_lines = []
        for t in self._entity.markers_for(self._channel_index):
            self._add_marker_line(t)

        self._divisor_line = pg.InfiniteLine(angle=90, movable=False,
                                      pen=pg.mkPen(color="#4caf50", width=2))
        self._divisor_line.addMarker('^', position=0.0, size=14)  # triangle at bottom, pointing up (inward)
        self._divisor_line.addMarker('v', position=1.0, size=14)  # inverse triangle at top, pointing down (inward)
        self._divisor_line.setVisible(False)
        self._plot.addItem(self._divisor_line)

        self._hover_region = pg.LinearRegionItem(
            values=(0, 0), brush=pg.mkBrush(255, 255, 255, 40),
            pen=pg.mkPen(None), movable=False,
        )
        self._hover_region.setZValue(-10)
        self._hover_region.setVisible(False)
        self._plot.addItem(self._hover_region)

        self._selection_region = pg.LinearRegionItem(
            values=(0, 0), brush=pg.mkBrush(79, 195, 247, 70),
            pen=pg.mkPen(None), movable=False,
        )
        self._selection_region.setZValue(-9)
        self._selection_region.setVisible(False)
        self._plot.addItem(self._selection_region)

        self._trim_playhead = pg.InfiniteLine(angle=90, movable=False,
                                               pen=pg.mkPen(color="#ff4d4d", width=2))
        self._trim_playhead.addMarker('^', position=0.0, size=14)
        self._trim_playhead.addMarker('v', position=1.0, size=14)
        self._trim_playhead.setVisible(False)
        self._plot.addItem(self._trim_playhead)

        layout.addWidget(self._plot)

        controls = QHBoxLayout()
        controls.setSpacing(6)

        self._play_btn = QPushButton("▶ Play")
        self._pause_btn = QPushButton("⏸ Pause")
        self._reset_btn = QPushButton("⏹ Reset")
        self._clip_btn = QPushButton("✂ Clip")

        self._play_btn.clicked.connect(self._on_play_clicked)
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        self._reset_btn.clicked.connect(self._on_reset_clicked)
        self._clip_btn.clicked.connect(self._on_clip_clicked)

        self._divide_btn = QPushButton("✔ Divide")
        self._cancel_btn = QPushButton("✕ Cancel")
        self._divide_btn.setVisible(False)
        self._cancel_btn.setVisible(False)
        self._divide_btn.clicked.connect(self._on_divide_clicked)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)

        for btn in (self._play_btn, self._pause_btn, self._reset_btn, self._clip_btn,
                    self._divide_btn, self._cancel_btn):
            btn.setFixedWidth(90)
            controls.addWidget(btn)

        controls.addStretch()

        layout.addLayout(controls)
        self._update_button_visibility()

    def _update_button_visibility(self):
        playing = self._state == self.STATE_PLAYING
        paused = self._state == self.STATE_PAUSED
        self._plot.drag_enabled = playing or paused

        show_playback_controls = True          # ← was: self._trim_mode or not self._edit_mode
        self._play_btn.setVisible(not playing and show_playback_controls)
        self._play_btn.setText("▶ Resume" if paused else "▶ Play")
        self._pause_btn.setVisible(playing and show_playback_controls)
        self._reset_btn.setVisible((playing or paused) and show_playback_controls)

    def _add_marker_line(self, t):
        line = pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen(color="#ffffff", width=1))
        self._plot.addItem(line)
        self._marker_lines.append(line)

    @property
    def channel_index(self):
        return self._channel_index

    @property
    def selected_segment(self):
        return self._selected_segment

    def _boundary_times(self):
        return sorted({self._start_time, self._end_time,
                        *self._entity.markers_for(self._channel_index)})

    def _on_segment_hover(self, start, end):
        if start is None or self._selected_segment == (start, end):
            self._hover_region.setVisible(False)
            return
        self._hover_region.setRegion((start, end))
        self._hover_region.setVisible(True)

    def _on_segment_click(self, index, start, end):
        self.select_segment(start, end)

    def select_segment(self, start, end):
        self._selected_segment = (start, end)
        self._selection_region.setRegion((start, end))
        self._selection_region.setVisible(True)
        self._hover_region.setVisible(False)
        self.segment_selected.emit(self, start, end)

    def clear_selection(self):
        if self._selected_segment is None:
            return
        self._selected_segment = None
        self._selection_region.setVisible(False)
        self.segment_deselected.emit(self)

    def _refresh_segment_mode(self):
        enabled = (
            self._edit_mode
            and self._state == self.STATE_STOPPED
            and not self._clip_mode
            and not self._trim_mode
            and not self._timescale_mode
            and not self._vscale_mode
            and not self._segment_locked
        )
        self._plot.set_segment_mode(
            enabled,
            boundaries_getter=self._boundary_times,
            on_hover=self._on_segment_hover,
            on_click=self._on_segment_click,
        )
        if not enabled:
            self._hover_region.setVisible(False)

    def set_segment_locked(self, locked: bool):
        self._segment_locked = locked
        self._refresh_segment_mode()

    def _preview_slice_indices(self, start, end):
        start_idx = int(np.searchsorted(self._times, start, side="left"))
        end_idx = int(np.searchsorted(self._times, end, side="right"))
        if end_idx <= start_idx:
            end_idx = start_idx + 1
        return start_idx, end_idx

    def enter_trim_preview(self, start, end):
        start_idx, end_idx = self._preview_slice_indices(start, end)

        self._trim_mode = True

        keep_mask = np.ones(len(self._times), dtype=bool)
        keep_mask[start_idx:end_idx] = False

        dt = 1.0 / self._sample_rate
        kept_count = int(keep_mask.sum())
        preview_times = self._start_time + np.arange(max(kept_count, 1)) * dt

        self._trim_start = self._start_time
        self._trim_end = float(preview_times[-1])

        for curve, (samples, _color, _label) in zip(self._curve_items, self._series):
            kept_samples = samples[keep_mask]
            if len(kept_samples) == 0:
                kept_samples = np.zeros(1, dtype=samples.dtype)
            curve.setData(preview_times, kept_samples)

        removed_span = end - start

        def _shift(t):
            if t <= start:
                return t
            if t >= end:
                return t - removed_span
            return None  # was inside the removed span

        self._start_boundary.setPos(self._trim_start)
        self._start_boundary.setVisible(True)
        self._end_boundary.setPos(self._trim_end)
        self._end_boundary.setVisible(True)

        for line, original_t in zip(self._marker_lines, self._entity.markers_for(self._channel_index)):
            shifted = _shift(original_t)
            if shifted is None:
                line.setVisible(False)
            else:
                line.setPos(shifted)
                line.setVisible(True)

        self._hover_region.setVisible(False)
        self._selection_region.setVisible(False)

        preview_audio = self._engine.audio_data[keep_mask]
        if len(preview_audio) == 0:
            preview_audio = np.zeros(
                (1,) + self._engine.audio_data.shape[1:], dtype=self._engine.audio_data.dtype
            )
        self._trim_engine = _PlaybackEngine(preview_audio, self._sample_rate)

        self._trim_playhead.setPos(self._trim_start)
        self._trim_playhead.setVisible(False)
        self._plot.setXRange(self._trim_start, max(self._trim_end, self._trim_start + 0.001), padding=0.02)

        self._refresh_segment_mode()
        self._update_button_visibility()

    def exit_trim_preview(self):
        if not self._trim_mode:
            return
        if self._state != self.STATE_STOPPED:
            self._do_reset()

        self._trim_mode = False
        self._trim_engine = None
        self._trim_start = None
        self._trim_end = None
        self._trim_playhead.setVisible(False)

        for curve, (samples, _color, _label) in zip(self._curve_items, self._series):
            curve.setData(self._times, samples)

        self._start_boundary.setPos(self._start_time)
        self._start_boundary.setVisible(True)
        self._end_boundary.setPos(self._end_time)
        self._end_boundary.setVisible(True)

        for line, original_t in zip(self._marker_lines, self._entity.markers_for(self._channel_index)):
            line.setPos(original_t)
            line.setVisible(True)

        self._show_overview()
        self._refresh_segment_mode()
        self._update_button_visibility()

    # ------------------------------------------------------------------
    # Timescale preview
    # ------------------------------------------------------------------

    def enter_timescale_preview(self, factor: float):
        """Show a scaled version of the waveform (time-axis only) as a preview."""
        self._timescale_mode = True
        self._hover_region.setVisible(False)
        self._update_timescale_display(factor)
        self._refresh_segment_mode()

    def update_timescale_preview(self, factor: float):
        """Refresh the preview whenever the factor spinbox changes."""
        if self._timescale_mode:
            self._update_timescale_display(factor)

    def _update_timescale_display(self, factor: float):
        factor = max(0.1, float(factor))
        start = self._start_time
        preview_times = start + (self._times - start) * factor
        for curve, (samples, _color, _label) in zip(self._curve_items, self._series):
            curve.setData(preview_times, samples)
        preview_end = float(preview_times[-1]) if len(preview_times) > 0 else start + 0.001
        self._end_boundary.setPos(preview_end)

        # Keep division/marker lines aligned with the stretched waveform.
        for line, original_t in zip(self._marker_lines, self._entity.markers_for(self._channel_index)):
            line.setPos(start + (original_t - start) * factor)

        self._plot.setXRange(start, max(preview_end, start + 0.001), padding=0.02)

    def exit_timescale_preview(self):
        """Restore the original waveform display."""
        if not self._timescale_mode:
            return
        self._timescale_mode = False
        for curve, (samples, _color, _label) in zip(self._curve_items, self._series):
            curve.setData(self._times, samples)
        self._start_boundary.setPos(self._start_time)
        self._end_boundary.setPos(self._end_time)
    
        for line, original_t in zip(self._marker_lines, self._entity.markers_for(self._channel_index)):
            line.setPos(original_t)                     # ← add
    
        self._show_overview()
        self._refresh_segment_mode()

    def enter_vertical_scale_preview(self, factor: float, start, end):
        """Live amplitude-only preview for the selected range. No time-axis
        remapping happens here, so markers and the selection box don't move."""
        self._vscale_mode = True
        self._vscale_range = (start, end)
        self._hover_region.setVisible(False)
        self._update_vertical_scale_display(factor)
        self._refresh_segment_mode()

    def update_vertical_scale_preview(self, factor: float):
        if self._vscale_mode:
            self._update_vertical_scale_display(factor)

    def _update_vertical_scale_display(self, factor: float):
        start, end = self._vscale_range
        start_idx, end_idx = self._preview_slice_indices(start, end)
        for curve, (samples, _color, _label) in zip(self._curve_items, self._series):
            preview = samples.copy()
            preview[start_idx:end_idx] = preview[start_idx:end_idx] * factor
            curve.setData(self._times, preview)
            self._autoscale_y_for_preview() 

    def _autoscale_y_for_preview(self):
        """Grow (or shrink back) the Y range so every curve currently drawn
        on this plot — including a live vertical-scale preview — fits fully
        inside the view, instead of clipping at the fixed ±1.05 default."""
        peak = 1.0
        for curve in self._curve_items:
            y_data = curve.yData
            if y_data is not None and len(y_data):
                curve_peak = float(np.max(np.abs(y_data)))
                if curve_peak > peak:
                    peak = curve_peak
        self._plot.setYRange(-peak * 1.05, peak * 1.05)

    def exit_vertical_scale_preview(self):
        if not self._vscale_mode:
            return
        self._vscale_mode = False
        self._vscale_range = None
        for curve, (samples, _color, _label) in zip(self._curve_items, self._series):
            curve.setData(self._times, samples)
        self._autoscale_y_for_preview()
        self._refresh_segment_mode()

    def preview_channel_curve(self, curve_index, factor, start, end):
        """Live-preview another plot's vertical-scale edit by scaling just
        that one overlaid curve here too. Used on the Overall plot so a
        channel being edited stays visible even if it's currently drawn
        underneath an identical/overlapping channel."""
        self._preview_curve_index = curve_index
        self._preview_curve_range = (start, end)
        self._update_channel_curve_preview(factor)

    def update_channel_curve_preview(self, factor: float):
        if self._preview_curve_index is not None:
            self._update_channel_curve_preview(factor)

    def _update_channel_curve_preview(self, factor: float):
        curve_index = self._preview_curve_index
        start, end = self._preview_curve_range
        start_idx, end_idx = self._preview_slice_indices(start, end)
        samples = self._series[curve_index][0]
        preview = samples.copy()
        preview[start_idx:end_idx] = preview[start_idx:end_idx] * factor
        self._curve_items[curve_index].setData(self._times, preview)
        self._autoscale_y_for_preview() 

    def exit_channel_curve_preview(self):
        if self._preview_curve_index is None:
            return
        curve_index = self._preview_curve_index
        samples = self._series[curve_index][0]
        self._curve_items[curve_index].setData(self._times, samples)
        self._preview_curve_index = None
        self._preview_curve_range = None
        self._autoscale_y_for_preview()

    def restore_selection_visual(self):
        """Re-show the selection highlight after exiting a preview mode."""
        if self._selected_segment is not None:
            self._selection_region.setRegion(self._selected_segment)
            self._selection_region.setVisible(True)

    def _update_clip_related_visibility(self):
        self._clip_btn.setVisible(not self._clip_mode and not self._edit_mode)
        self._divide_btn.setVisible(self._clip_mode)
        self._cancel_btn.setVisible(self._clip_mode)

    def _clip_button_allowed(self):
        active = self._group.active_player
        if active is None:
            return True
        if active is self and self._state == self.STATE_PAUSED:
            return True
        return False

    def _refresh_clip_enabled(self):
        if self._clip_mode:
            return
        if self._group.clip_active_player is not None and self._group.clip_active_player is not self:
            self._clip_btn.setEnabled(False)
            return
        self._clip_btn.setEnabled(self._clip_button_allowed())

    def _refresh_play_enabled(self):
        if self._clip_mode:
            self._play_btn.setEnabled(False)
            return
        active = self._group.active_player
        self._play_btn.setEnabled(active is None or active is self)

    def _on_group_clip_active_changed(self, clip_player):
        if clip_player is None:
            self._refresh_clip_enabled()
            self._refresh_play_enabled()
            return
        if clip_player is self:
            return
        self._clip_btn.setEnabled(False)
        self._play_btn.setEnabled(False)

    def _plot_time_to_pixel(self, t):
        view_box = self._plot.getPlotItem().vb
        scene_pt = view_box.mapViewToScene(pg.Point(t, 0))
        return self._plot.mapFromScene(scene_pt).x()

    def _set_divisor_time(self, t):
        self._divisor_time = self._clamp_time(t)
        self._divisor_line.setPos(self._divisor_time)

    def _on_clip_clicked(self):
        if not self._group.request_clip_mode(self):
            return

        self._entered_clip_from_pause = (self._state == self.STATE_PAUSED)
        self._clip_mode = True

        if self._entered_clip_from_pause:
            self._divisor_time = self._clamp_time(self._current_center_time())
        else:
            self._divisor_time = self._clamp_time((self._start_time + self._end_time) / 2.0)

        self._divisor_line.setPos(self._divisor_time)
        self._divisor_line.setVisible(True)

        self._plot.set_clip_mode(
            enabled=True,
            allow_graph_drag=self._entered_clip_from_pause,
            divisor_getter=lambda: self._divisor_time,
            divisor_setter=self._set_divisor_time,
            time_to_pixel=self._plot_time_to_pixel,
        )

        self._update_clip_related_visibility()
        self._play_btn.setEnabled(False)
        self._reset_btn.setEnabled(False)

    def _exit_clip_mode(self):
        self._clip_mode = False
        self._divisor_line.setVisible(False)
        self._plot.set_clip_mode(enabled=False)
        self._update_clip_related_visibility()
        self._reset_btn.setEnabled(True)
        self._group.release_clip_mode(self)
        self._refresh_play_enabled()
        self._refresh_clip_enabled()

    def _on_divide_clicked(self):
        t = self._divisor_time
        existing = set(self._entity.markers_for(self._channel_index))
        self._entity.add_marker(self._channel_index, t)
        if t not in existing:
            self._add_marker_line(t)
        self._exit_clip_mode()
        self.marker_added.emit()

    def _on_cancel_clicked(self):
        self._exit_clip_mode()

    def force_idle(self):
        if self._clip_mode:
            self._exit_clip_mode()
        if self._trim_mode:
            self.exit_trim_preview()
        if self._timescale_mode:
            self.exit_timescale_preview()
        if self._vscale_mode:
            self.exit_vertical_scale_preview()
        if self._preview_curve_index is not None:     # ← add
            self.exit_channel_curve_preview()
        if self._state != self.STATE_STOPPED:
            self._do_reset()
        self.clear_selection()

    def set_edit_mode(self, enabled: bool):
        self._edit_mode = enabled
        self._update_button_visibility()
        self._update_clip_related_visibility()
        self._refresh_segment_mode()
        if not enabled:
            self.clear_selection()

    # ---------- view helpers ----------

    def _show_overview(self):
        self._plot.setXRange(self._start_time, max(self._end_time, self._start_time + 0.001))
        self._playhead.setVisible(False)

    def _show_window_centered_at(self, t):
        self._plot.setXRange(t - _HALF_WINDOW_SECONDS, t + _HALF_WINDOW_SECONDS, padding=0)
        self._playhead.setPos(t)
        self._playhead.setVisible(True)

    # ---------- own playback ----------

    def _on_play_clicked(self):
        if not self._group.request_play(self):
            return
        self._state = self.STATE_PLAYING
        (self._trim_engine if self._trim_mode else self._engine).start_or_resume()
        self._timer.start()
        self._update_button_visibility()
        self._refresh_clip_enabled()
        self._refresh_segment_mode()

    def _on_pause_clicked(self):
        (self._trim_engine if self._trim_mode else self._engine).pause()
        self._timer.stop()
        self._state = self.STATE_PAUSED
        if not self._trim_mode:
            if self._edit_mode:
                self._playhead.setPos(self._current_center_time())
                self._playhead.setVisible(True)
            else:
                self._show_window_centered_at(self._current_center_time())
        self._update_button_visibility()
        self._refresh_clip_enabled()
        self._refresh_segment_mode()  

    def _on_reset_clicked(self):
        self._do_reset()

    def _do_reset(self):
        (self._trim_engine if self._trim_mode else self._engine).stop()
        self._timer.stop()
        self._state = self.STATE_STOPPED
        if self._trim_mode:
            self._trim_playhead.setPos(self._trim_start)
            self._trim_playhead.setVisible(False)
        else:
            self._show_overview()
        self._update_button_visibility()
        self._group.release(self)
        self._refresh_clip_enabled()
        self._refresh_segment_mode()  

    def _on_tick(self):
        engine = self._trim_engine if self._trim_mode else self._engine
        if engine.is_finished():
            self._do_reset()
            return

        if self._trim_mode:
            current_time = self._trim_start + engine.current_frame() / self._sample_rate
            self._trim_playhead.setPos(current_time)
            self._trim_playhead.setVisible(True)
            return

        current_time = self._start_time + engine.current_frame() / self._sample_rate

        if self._edit_mode:
            self._playhead.setPos(current_time)
            self._playhead.setVisible(True)
        else:
            self._show_window_centered_at(current_time)

        if self._is_driver:
            self._group.broadcast_position(current_time)

    def _current_center_time(self):
        return self._start_time + self._engine.current_frame() / self._sample_rate

    def _clamp_time(self, t):
        return max(self._start_time, min(t, self._end_time))

    def _on_drag_start(self):
        if self._state not in (self.STATE_PLAYING, self.STATE_PAUSED):
            return
        self._drag_was_playing = (self._state == self.STATE_PLAYING)
        if self._drag_was_playing:
            self._timer.stop()
            (self._trim_engine if self._trim_mode else self._engine).pause()

    def _on_seek(self, dt):
        if self._state not in (self.STATE_PLAYING, self.STATE_PAUSED):
            return

        if self._trim_mode:
            current = self._trim_start + self._trim_engine.current_frame() / self._sample_rate
            new_time = max(self._trim_start, min(current + dt, self._trim_end))
            frame = int(round((new_time - self._trim_start) * self._sample_rate))
            self._trim_engine.seek(frame)
            self._trim_playhead.setPos(new_time)
            self._trim_playhead.setVisible(True)
            return

        old_center = self._current_center_time()
        # Edit mode: graph is fixed, so dragging right should move the line
        # right (direct mapping). Non-edit mode keeps the old "pan the image"
        # feel, where dragging right reveals earlier content.
        new_time = self._clamp_time(old_center + dt) if self._edit_mode else self._clamp_time(old_center - dt)
        actual_dt = new_time - old_center

        frame = int(round((new_time - self._start_time) * self._sample_rate))
        self._engine.seek(frame)

        if self._edit_mode:
            self._playhead.setPos(new_time)
            self._playhead.setVisible(True)
        else:
            self._show_window_centered_at(new_time)

        if self._clip_mode and self._divisor_time is not None:
            self._divisor_time = self._clamp_time(self._divisor_time + actual_dt)
            self._divisor_line.setPos(self._divisor_time)

        if self._is_driver:
            self._group.broadcast_position(new_time)

    def _on_drag_end(self):
        if self._state not in (self.STATE_PLAYING, self.STATE_PAUSED):
            return
        if getattr(self, "_drag_was_playing", False):
            (self._trim_engine if self._trim_mode else self._engine).start_or_resume()
            self._timer.start()
        self._drag_was_playing = False
    def _on_drag_end(self):
        if self._state not in (self.STATE_PLAYING, self.STATE_PAUSED):
            return

        if getattr(self, "_drag_was_playing", False):
            self._engine.start_or_resume()   # single, clean restart
            self._timer.start()
        self._drag_was_playing = False

    # ---------- following another player's broadcast ----------

    def _on_driver_position(self, t):
        if self._group.active_player is self:
            return
        if self._edit_mode:
            self._playhead.setPos(t)
            self._playhead.setVisible(True)
        else:
            self._show_window_centered_at(t)

    def _on_group_active_changed(self, active_player):
        if active_player is None:
            if self._state != self.STATE_STOPPED:
                self._do_reset()
            else:
                self._show_overview()
            self._play_btn.setEnabled(True)
            return

        if active_player is self:
            return

        self._play_btn.setEnabled(False)