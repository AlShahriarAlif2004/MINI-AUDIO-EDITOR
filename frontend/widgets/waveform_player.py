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

        self._marker_mode = False
        self._marker_positions_getter = None
        self._on_marker_click = None

        self._range_mode = False
        self._range_left_getter = None
        self._range_left_setter = None
        self._range_right_getter = None
        self._range_right_setter = None
        self._range_time_to_pixel = None
        self._dragging_range = None   # "left" | "right" | None

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

    def set_range_mode(self, enabled, left_getter=None, left_setter=None,
                        right_getter=None, right_setter=None, time_to_pixel=None):
        """Two independently draggable vertical lines (used by noise-range
        selection): left_getter/right_getter report each line's current
        time, left_setter/right_setter are called with the proposed new
        time while dragging. Clamping so left never crosses right is the
        setters' responsibility (see WaveformPlayer._set_noise_start/_end)."""
        self._range_mode = enabled
        self._range_left_getter = left_getter
        self._range_left_setter = left_setter
        self._range_right_getter = right_getter
        self._range_right_setter = right_setter
        self._range_time_to_pixel = time_to_pixel
        self._dragging_range = None

    def set_segment_mode(self, enabled, boundaries_getter=None, on_hover=None, on_click=None):
        self._segment_mode = enabled
        self._segment_boundaries_getter = boundaries_getter
        self._on_segment_hover = on_hover
        self._on_segment_click = on_click

    def set_marker_mode(self, enabled, positions_getter=None, on_click=None):
        """positions_getter() -> list[(index, time)] of *selectable*
        markers only — boundary lines are never included, so they can
        never be picked."""
        self._marker_mode = enabled
        self._marker_positions_getter = positions_getter
        self._on_marker_click = on_click

    def _marker_at_pixel(self, event_pos):
        if not self._marker_positions_getter:
            return None
        positions = self._marker_positions_getter()
        if not positions:
            return None
        view_box = self.getPlotItem().vb
        scene_pos = self.mapToScene(event_pos)
        t = view_box.mapSceneToView(scene_pos).x()
        px_per_sec = self._compute_pixels_per_second()
        best_index = None
        best_dist = self._DIVISOR_HIT_TOLERANCE_PX
        for index, pos_t in positions:
            dist_px = abs(pos_t - t) * px_per_sec
            if dist_px <= best_dist:
                best_index = index
                best_dist = dist_px
        return best_index

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
        if self._marker_mode and event.button() == Qt.LeftButton:
            index = self._marker_at_pixel(event.pos())
            if self._on_marker_click:
                self._on_marker_click(index)
            event.accept()
            return

        if self._range_mode and event.button() == Qt.LeftButton:
            x = event.pos().x()
            left_px = self._range_time_to_pixel(self._range_left_getter())
            right_px = self._range_time_to_pixel(self._range_right_getter())
            if abs(x - left_px) <= self._DIVISOR_HIT_TOLERANCE_PX:
                self._dragging_range = "left"
            elif abs(x - right_px) <= self._DIVISOR_HIT_TOLERANCE_PX:
                self._dragging_range = "right"
            else:
                self._dragging_range = None
            event.accept()
            return

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
        if self._range_mode:
            if self._dragging_range:
                view_box = self.getPlotItem().vb
                scene_pos = self.mapToScene(event.pos())
                t = view_box.mapSceneToView(scene_pos).x()
                if self._dragging_range == "left":
                    self._range_left_setter(t)
                else:
                    self._range_right_setter(t)
            event.accept()
            return

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
        if self._range_mode and event.button() == Qt.LeftButton:
            self._dragging_range = None
            event.accept()
            return

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
        show_markers: bool = True,
        show_clip_button: bool = True,
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
        self._show_markers = show_markers
        self._show_clip_button = show_clip_button

        self._start_time = float(times[0]) if len(times) else 0.0
        self._end_time = float(times[-1]) if len(times) else 0.0

        self._engine = _PlaybackEngine(audio_data, sample_rate)
        self._state = self.STATE_STOPPED

        self._clip_mode = False
        self._divisor_time = None
        self._entered_clip_from_pause = False
        self._edit_mode = False 
        self._tool_mode = False
        self._selected_segment = None
        self._segment_locked = False

        self._delete_divisor_mode = False
        self._selected_marker_index = None
        self._pending_deleted_indices = set()

        self._trim_mode = False
        self._trim_engine = None
        self._trim_start = None
        self._trim_end = None

        self._extract_mode = False
        self._extract_engine = None
        self._extract_start = None
        self._extract_end = None

        self._timescale_mode = False
        self._timescale_range = None
        self._timescale_preview_end = None
        self._timescale_engine = None
        self._concat_mode = False
        self._concat_engine = None
        self._concat_times = None
        self._concat_end = None
        self._vscale_mode = False
        self._vscale_range = None
        self._vscale_engine = None
        self._effect_mode = None          # 'reverse' | 'fade_in' | 'fade_out' | None
        self._effect_range = None
        self._effect_engine = None        # _PlaybackEngine over the previewed (modified) audio
        self._preview_curve_index = None
        self._preview_curve_range = None
        self._preview_curve_effect = None  # set when the mirrored curve preview is a factor-free effect
        self._preview_curve_engine = None  # playback engine for Overall when a channel is being edited

        self._noise_mode = False       # True while the two green range handles are live
        self._noise_start = None
        self._noise_end = None

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
        self._refresh_marker_mode()

    # ---------- UI ----------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._plot = _DraggablePlotWidget(self)
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

        self._autoscale_y()

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
        self._marker_times = []
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

        self._extract_playhead = pg.InfiniteLine(angle=90, movable=False,
                                                   pen=pg.mkPen(color="#ff4d4d", width=2))
        self._extract_playhead.addMarker('^', position=0.0, size=14)
        self._extract_playhead.addMarker('v', position=1.0, size=14)
        self._extract_playhead.setVisible(False)
        self._plot.addItem(self._extract_playhead)

        # Two independently draggable green handles bounding the noise
        # range during Noise Removal's selection phase (left must never
        # cross right — enforced in _set_noise_start/_set_noise_end).
        self._noise_start_line = pg.InfiniteLine(angle=90, movable=False,
                                                  pen=pg.mkPen(color="#4caf50", width=2))
        self._noise_start_line.addMarker('^', position=0.0, size=14)
        self._noise_start_line.addMarker('v', position=1.0, size=14)
        self._noise_start_line.setVisible(False)
        self._plot.addItem(self._noise_start_line)

        self._noise_end_line = pg.InfiniteLine(angle=90, movable=False,
                                                pen=pg.mkPen(color="#4caf50", width=2))
        self._noise_end_line.addMarker('^', position=0.0, size=14)
        self._noise_end_line.addMarker('v', position=1.0, size=14)
        self._noise_end_line.setVisible(False)
        self._plot.addItem(self._noise_end_line)

        layout.addWidget(self._plot)

        controls = QHBoxLayout()
        controls.setSpacing(6)

        self._play_btn = QPushButton("▶ Play")
        self._delete_marker_btn = QPushButton("🗑 Delete")
        self._pause_btn = QPushButton("⏸ Pause")
        self._reset_btn = QPushButton("⏹ Reset")
        self._clip_btn = QPushButton("✂ Clip")

        self._play_btn.clicked.connect(self._on_play_clicked)
        self._delete_marker_btn.clicked.connect(self._on_delete_marker_clicked)
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        self._reset_btn.clicked.connect(self._on_reset_clicked)
        self._clip_btn.clicked.connect(self._on_clip_clicked)

        self._divide_btn = QPushButton("✔ Divide")
        self._cancel_btn = QPushButton("✕ Cancel")
        self._divide_btn.setVisible(False)
        self._cancel_btn.setVisible(False)
        self._divide_btn.clicked.connect(self._on_divide_clicked)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)

        for btn in (self._play_btn, self._pause_btn, self._reset_btn, self._delete_marker_btn,
                    self._clip_btn, self._divide_btn, self._cancel_btn):
            btn.setFixedWidth(90)
            controls.addWidget(btn)

        controls.addStretch()

        layout.addLayout(controls)
        self._update_button_visibility()
        self._update_clip_related_visibility()

        if not self._show_markers:
            # Used for the Previous/Filtered preview pairs shown inside the
            # Noise Removal tab strip: plain play/pause/reset only, no
            # white boundary/division lines cluttering the comparison.
            self._start_boundary.setVisible(False)
            self._end_boundary.setVisible(False)
            for line in self._marker_lines:
                line.setVisible(False)

    def _update_button_visibility(self):
        playing = self._state == self.STATE_PLAYING
        paused = self._state == self.STATE_PAUSED
        self._plot.drag_enabled = playing or paused

        show_playback_controls = not self._noise_mode  # was: self._trim_mode or not self._edit_mode
        self._play_btn.setVisible(not playing and show_playback_controls)
        self._play_btn.setText("▶ Resume" if paused else "▶ Play")
        self._pause_btn.setVisible(playing and show_playback_controls)
        self._reset_btn.setVisible((playing or paused) and show_playback_controls)

        self._delete_marker_btn.setVisible(self._delete_divisor_mode)
        self._delete_marker_btn.setEnabled(
            self._selected_marker_index is not None and self._state == self.STATE_STOPPED
        )

    def _add_marker_line(self, t):
        line = pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen(color="#ffffff", width=1))
        self._plot.addItem(line)
        self._marker_lines.append(line)
        self._marker_times.append(t)

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
            and not self._extract_mode
            and not self._timescale_mode
            and not self._concat_mode
            and not self._vscale_mode
            and self._effect_mode is None
            and not self._segment_locked
            and not self._noise_mode
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

    # ------------------------------------------------------------------
    # Delete-divisor mode
    # ------------------------------------------------------------------

    def _selectable_marker_positions(self):
        """(index, time) pairs for markers not already staged for
        deletion. Boundary lines are never in this list, so they can
        never be selected."""
        return [
            (i, t) for i, t in enumerate(self._marker_times)
            if i not in self._pending_deleted_indices
        ]

    def _refresh_marker_mode(self):
        enabled = self._delete_divisor_mode and self._state == self.STATE_STOPPED
        self._plot.set_marker_mode(
            enabled,
            positions_getter=self._selectable_marker_positions,
            on_click=self._on_marker_clicked,
        )

    def _select_marker(self, index):
        if self._selected_marker_index is not None:
            self._marker_lines[self._selected_marker_index].setPen(pg.mkPen(color="#ffffff", width=1))
        self._selected_marker_index = index
        self._marker_lines[index].setPen(pg.mkPen(color="#ffeb3b", width=2))
        self._update_button_visibility()

    def _deselect_marker(self):
        if self._selected_marker_index is not None:
            self._marker_lines[self._selected_marker_index].setPen(pg.mkPen(color="#ffffff", width=1))
            self._selected_marker_index = None
            self._update_button_visibility()

    def deselect_marker(self):
        """Public entry point so a click on empty canvas space (handled
        by EntityPlotView, not this widget) can clear the current
        divisor-line selection without disturbing any staged deletions."""
        self._deselect_marker()

    def _on_marker_clicked(self, index):
        if index is None:
            return
        self._select_marker(index)

    def _on_delete_marker_clicked(self):
        if self._selected_marker_index is None:
            return
        index = self._selected_marker_index
        self._marker_lines[index].setVisible(False)   # staged, not yet committed
        self._pending_deleted_indices.add(index)
        self._selected_marker_index = None
        self._update_button_visibility()

    def enter_delete_divisor_mode(self):
        self._delete_divisor_mode = True
        self._selected_marker_index = None
        self._pending_deleted_indices = set()
        self._update_button_visibility()
        self._refresh_marker_mode()

    def exit_delete_divisor_mode(self):
        """Cancel path: un-hide any staged (but not applied) deletions."""
        if not self._delete_divisor_mode:
            return
        self._delete_divisor_mode = False
        self._deselect_marker()
        for index in self._pending_deleted_indices:
            self._marker_lines[index].setVisible(True)
        self._pending_deleted_indices = set()
        self._update_button_visibility()
        self._refresh_marker_mode()

    def commit_delete_divisor(self):
        """Apply path: permanently removes the staged lines and returns
        the set of times that were deleted, for the caller to strip out
        of the entity's division/marker list."""
        deleted_times = {self._marker_times[i] for i in self._pending_deleted_indices}
        for index in sorted(self._pending_deleted_indices, reverse=True):
            line = self._marker_lines.pop(index)
            self._plot.removeItem(line)
            del self._marker_times[index]
        self._delete_divisor_mode = False
        self._selected_marker_index = None
        self._pending_deleted_indices = set()
        self._update_button_visibility()
        self._refresh_marker_mode()
        return deleted_times

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
    # Extract preview
    # ------------------------------------------------------------------
    # The inverse of Trim: only the selected [start, end] span is kept and
    # shown, everything outside it is dropped from the preview entirely.
    # Sample indices (and therefore times) of the kept portion are left
    # untouched — the underlying data operation is a plain crop, not a
    # splice — so, unlike Trim, nothing needs to be shifted.

    def enter_extract_preview(self, start, end):
        start_idx, end_idx = self._preview_slice_indices(start, end)

        self._extract_mode = True

        preview_times = self._times[start_idx:end_idx]
        if len(preview_times) == 0:
            preview_times = np.array([start])

        self._extract_start = float(preview_times[0])
        self._extract_end = float(preview_times[-1])

        for curve, (samples, _color, _label) in zip(self._curve_items, self._series):
            kept_samples = samples[start_idx:end_idx]
            if len(kept_samples) == 0:
                kept_samples = np.zeros(1, dtype=samples.dtype)
            curve.setData(preview_times, kept_samples)

        self._start_boundary.setPos(self._extract_start)
        self._start_boundary.setVisible(True)
        self._end_boundary.setPos(self._extract_end)
        self._end_boundary.setVisible(True)

        for line, original_t in zip(self._marker_lines, self._entity.markers_for(self._channel_index)):
            if start < original_t < end:
                line.setPos(original_t)
                line.setVisible(True)
            else:
                line.setVisible(False)

        self._hover_region.setVisible(False)
        self._selection_region.setVisible(False)

        preview_audio = self._engine.audio_data[start_idx:end_idx]
        if len(preview_audio) == 0:
            preview_audio = np.zeros(
                (1,) + self._engine.audio_data.shape[1:], dtype=self._engine.audio_data.dtype
            )
        self._extract_engine = _PlaybackEngine(preview_audio, self._sample_rate)

        self._extract_playhead.setPos(self._extract_start)
        self._extract_playhead.setVisible(False)
        self._plot.setXRange(self._extract_start, max(self._extract_end, self._extract_start + 0.001), padding=0.02)

        self._refresh_segment_mode()
        self._update_button_visibility()

    def exit_extract_preview(self):
        if not self._extract_mode:
            return
        if self._state != self.STATE_STOPPED:
            self._do_reset()

        self._extract_mode = False
        self._extract_engine = None
        self._extract_start = None
        self._extract_end = None
        self._extract_playhead.setVisible(False)

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
    # Noise-removal range selection
    # ------------------------------------------------------------------
    # Unlike every other preview mode, this doesn't touch the drawn curve
    # at all — it only arms two draggable green handles (start/end of the
    # noise window) and hides everything that would otherwise clutter the
    # selection view: playback controls, the playhead, and every white
    # boundary/division line.

    def enter_noise_range_mode(self, start, end):
        self._noise_mode = True

        start = self._clamp_time(start)
        end = self._clamp_time(end)
        if start > end:
            start, end = end, start
        self._noise_start = start
        self._noise_end = end

        self._start_boundary.setVisible(False)
        self._end_boundary.setVisible(False)
        for line in self._marker_lines:
            line.setVisible(False)
        self._playhead.setVisible(False)

        self._noise_start_line.setPos(self._noise_start)
        self._noise_end_line.setPos(self._noise_end)
        self._noise_start_line.setVisible(True)
        self._noise_end_line.setVisible(True)

        self._plot.set_range_mode(
            enabled=True,
            left_getter=lambda: self._noise_start,
            left_setter=self._set_noise_start,
            right_getter=lambda: self._noise_end,
            right_setter=self._set_noise_end,
            time_to_pixel=self._plot_time_to_pixel,
        )

        self._hover_region.setVisible(False)
        self._selection_region.setVisible(False)

        self._refresh_segment_mode()
        self._update_button_visibility()

    def _set_noise_start(self, t):
        t = self._clamp_time(t)
        self._noise_start = min(t, self._noise_end)
        self._noise_start_line.setPos(self._noise_start)

    def _set_noise_end(self, t):
        t = self._clamp_time(t)
        self._noise_end = max(t, self._noise_start)
        self._noise_end_line.setPos(self._noise_end)

    def get_noise_range(self):
        return self._noise_start, self._noise_end

    def exit_noise_range_mode(self):
        if not self._noise_mode:
            return
        self._noise_mode = False
        self._noise_start = None
        self._noise_end = None

        self._noise_start_line.setVisible(False)
        self._noise_end_line.setVisible(False)
        self._plot.set_range_mode(enabled=False)

        self._start_boundary.setVisible(True)
        self._end_boundary.setVisible(True)
        for line in self._marker_lines:
            line.setVisible(True)

        self._refresh_segment_mode()
        self._update_button_visibility()

    # ------------------------------------------------------------------
    # Timescale preview
    # ------------------------------------------------------------------

    def enter_timescale_preview(self, factor: float, start, end):
        """Show a scaled version of the waveform (time-axis only) as a
        preview, restricted to [start, end]. Content before start is
        untouched; content after end keeps its shape and just shifts to
        stay contiguous with the resized segment."""
        self._timescale_mode = True
        self._timescale_range = (start, end)
        self._hover_region.setVisible(False)
        self._update_timescale_display(factor)
        self._refresh_segment_mode()

    def update_timescale_preview(self, factor: float):
        """Refresh the preview whenever the factor spinbox changes."""
        if self._timescale_mode:
            self._update_timescale_display(factor)

    def _update_timescale_display(self, factor: float):
        factor = max(0.1, float(factor))
        start, end = self._timescale_range
        start_idx, end_idx = self._preview_slice_indices(start, end)
        dt = 1.0 / self._sample_rate

        seg_len = end_idx - start_idx
        new_seg_len = max(1, round(seg_len * factor))
        delta = (new_seg_len - seg_len) * dt

        seg_start_time = self._times[start_idx] if start_idx < len(self._times) else start
        preview_seg_times = seg_start_time + np.arange(new_seg_len) * dt
        preview_times = np.concatenate([
            self._times[:start_idx],
            preview_seg_times,
            self._times[end_idx:] + delta,
        ])

        old_pos = np.arange(max(seg_len, 1))
        new_pos = np.linspace(0, max(seg_len - 1, 0), new_seg_len)

        for curve, (samples, _color, _label) in zip(self._curve_items, self._series):
            seg = samples[start_idx:end_idx]
            scaled_seg = np.interp(new_pos, old_pos, seg) if seg_len > 0 else np.zeros(new_seg_len)
            preview_samples = np.concatenate([samples[:start_idx], scaled_seg, samples[end_idx:]])
            curve.setData(preview_times, preview_samples)

        preview_end = float(preview_times[-1]) if len(preview_times) > 0 else start + 0.001
        self._timescale_preview_end = preview_end
        self._end_boundary.setPos(preview_end)

        def _shift(t):
            if t <= start:
                return t
            if t >= end:
                return t + delta
            return start + (t - start) * factor

        for line, original_t in zip(self._marker_lines, self._entity.markers_for(self._channel_index)):
            line.setPos(_shift(original_t))

        self._build_timescale_preview_engine(start_idx, end_idx, new_seg_len)
        self._plot.setXRange(self._start_time, max(preview_end, self._start_time + 0.001), padding=0.02)

    def _build_timescale_preview_engine(self, start_idx, end_idx, new_seg_len):
        """Mirrors the visual preview onto an actual playback buffer, so
        Play hears the scaled segment too instead of the original audio."""
        audio = self._engine.audio_data
        seg = audio[start_idx:end_idx]
        old_pos = np.arange(max(len(seg), 1))
        new_pos = np.linspace(0, max(len(seg) - 1, 0), new_seg_len)

        if len(seg) == 0:
            scaled_seg = np.zeros((new_seg_len,) + audio.shape[1:], dtype=audio.dtype)
        elif audio.ndim == 1:
            scaled_seg = np.interp(new_pos, old_pos, seg).astype(audio.dtype)
        else:
            scaled_seg = np.stack(
                [np.interp(new_pos, old_pos, seg[:, c]) for c in range(seg.shape[1])],
                axis=1,
            ).astype(audio.dtype)

        preview_audio = np.concatenate([audio[:start_idx], scaled_seg, audio[end_idx:]], axis=0)
        self._timescale_engine = _PlaybackEngine(preview_audio, self._sample_rate)

    def exit_timescale_preview(self):
        """Restore the original waveform display."""
        if not self._timescale_mode:
            return
        if self._state != self.STATE_STOPPED and self._playback_engine() is self._timescale_engine:
            self._do_reset()
        self._timescale_mode = False
        self._timescale_range = None
        self._timescale_preview_end = None
        self._timescale_engine = None
        for curve, (samples, _color, _label) in zip(self._curve_items, self._series):
            curve.setData(self._times, samples)
        self._start_boundary.setPos(self._start_time)
        self._end_boundary.setPos(self._end_time)

        for line, original_t in zip(self._marker_lines, self._entity.markers_for(self._channel_index)):
            line.setPos(original_t)

        self._show_overview()
        self._refresh_segment_mode()

    # ------------------------------------------------------------------
    # Concatenate preview
    # ------------------------------------------------------------------

    def enter_concat_preview(self, insert_time, extra_samples_list):
        """Splice `extra_samples_list` (one 1-D array per curve, in the same
        order as self._curve_items) into the waveform right after
        `insert_time`, as a live preview. Does not touch the current
        selection region — the selected segment itself is untouched, only
        content after it shifts right."""
        self._concat_mode = True

        insert_idx = int(np.searchsorted(self._times, insert_time, side="right"))
        dt = 1.0 / self._sample_rate

        extra_len = max((len(a) for a in extra_samples_list), default=0)

        anchor_time = self._times[insert_idx - 1] if insert_idx > 0 else self._start_time
        insert_times = anchor_time + dt * (np.arange(extra_len) + 1)

        before_times = self._times[:insert_idx]
        after_times = self._times[insert_idx:] + extra_len * dt
        preview_times = np.concatenate([before_times, insert_times, after_times])
        self._concat_times = preview_times

        for curve, (samples, _color, _label), extra in zip(
            self._curve_items, self._series, extra_samples_list
        ):
            extra = np.asarray(extra)
            preview_samples = np.concatenate(
                [samples[:insert_idx], extra, samples[insert_idx:]]
            )
            curve.setData(preview_times, preview_samples)

        self._concat_end = float(preview_times[-1]) if len(preview_times) else self._end_time
        self._end_boundary.setPos(self._concat_end)
        self._end_boundary.setVisible(True)

        added_duration = extra_len * dt

        def _shift(t):
            return t + added_duration if t >= insert_time else t

        for line, original_t in zip(self._marker_lines, self._entity.markers_for(self._channel_index)):
            line.setPos(_shift(original_t))

        self._hover_region.setVisible(False)

        orig_audio = self._engine.audio_data
        if orig_audio.ndim == 1:
            extra_audio = (
                np.asarray(extra_samples_list[0]).astype(orig_audio.dtype)
                if extra_samples_list else np.zeros(0, dtype=orig_audio.dtype)
            )
            preview_audio = np.concatenate(
                [orig_audio[:insert_idx], extra_audio, orig_audio[insert_idx:]]
            )
        else:
            if extra_samples_list:
                extra_stack = np.stack(
                    [np.asarray(a) for a in extra_samples_list], axis=1
                ).astype(orig_audio.dtype)
            else:
                extra_stack = np.zeros((0, orig_audio.shape[1]), dtype=orig_audio.dtype)
            preview_audio = np.concatenate(
                [orig_audio[:insert_idx], extra_stack, orig_audio[insert_idx:]], axis=0
            )

        self._concat_engine = _PlaybackEngine(preview_audio, self._sample_rate)

        self._plot.setXRange(self._start_time, max(self._concat_end, self._start_time + 0.001), padding=0.02)
        self._refresh_segment_mode()
        self._update_button_visibility()

    def exit_concat_preview(self):
        if not self._concat_mode:
            return
        if self._state != self.STATE_STOPPED:
            self._do_reset()

        self._concat_mode = False
        self._concat_engine = None
        self._concat_times = None
        self._concat_end = None

        for curve, (samples, _color, _label) in zip(self._curve_items, self._series):
            curve.setData(self._times, samples)

        self._end_boundary.setPos(self._end_time)
        self._end_boundary.setVisible(True)

        for line, original_t in zip(self._marker_lines, self._entity.markers_for(self._channel_index)):
            line.setPos(original_t)
            line.setVisible(True)

        self._show_overview()
        self._refresh_segment_mode()
        self._update_button_visibility()

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
        self._autoscale_y()
        self._build_vscale_preview_engine(factor)

    def _build_vscale_preview_engine(self, factor: float):
        """Mirrors the visual vertical-scale preview onto an actual
        playback buffer, so Play hears the scaled segment too instead of
        the original audio. Rebuilt on every factor change, since the
        factor is live (spinbox/slider)."""
        start, end = self._vscale_range
        start_idx, end_idx = self._preview_slice_indices(start, end)
        preview_audio = self._engine.audio_data.copy()
        preview_audio[start_idx:end_idx] = preview_audio[start_idx:end_idx] * factor
        self._vscale_engine = _PlaybackEngine(preview_audio, self._sample_rate) 

    def _autoscale_y(self):
        """Grow (or shrink back) the Y range so every curve currently drawn
        on this plot — including a live vertical-scale preview, or the
        freshly-applied result of any edit op — fits fully inside the
        view, instead of clipping at the fixed ±1.05 default."""
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
        if self._state != self.STATE_STOPPED and self._playback_engine() is self._vscale_engine:
            self._do_reset()
        self._vscale_mode = False
        self._vscale_range = None
        self._vscale_engine = None
        for curve, (samples, _color, _label) in zip(self._curve_items, self._series):
            curve.setData(self._times, samples)
        self._autoscale_y()
        self._refresh_segment_mode()

    @staticmethod
    def _compute_effect(effect, segment):
        """Factor-free preview transform applied to one segment of samples.
        Mirrors the multiply-by-factor step in _update_vertical_scale_display,
        just with reverse/fade_in/fade_out standing in for the factor."""
        if effect == "reverse":
            return segment[::-1].copy()

        if len(segment) <= 1:
            return np.zeros_like(segment)

        if effect == "fade_in":
            factors = np.linspace(0.0, 1.0, len(segment))
        elif effect == "fade_out":
            factors = np.linspace(1.0, 0.0, len(segment))
        else:
            return segment

        if segment.ndim > 1:
            factors = factors[:, np.newaxis]   # broadcast across channels for 2-D playback buffers

        return segment * factors

    def enter_effect_preview(self, effect, start, end):
        """Live preview for reverse / fade_in / fade_out over the selected
        range. Just like enter_vertical_scale_preview, no time-axis
        remapping happens here, so markers and the selection box don't move.
        Unlike vertical scale there's no factor to tweak, so the preview is
        computed once up front instead of on every slider change."""
        self._effect_mode = effect
        self._effect_range = (start, end)
        self._hover_region.setVisible(False)
        self._update_effect_display()
        self._build_effect_preview_engine()
        self._refresh_segment_mode()

    def _build_effect_preview_engine(self):
        """Mirrors the (visual-only) curve preview onto an actual playback
        buffer, so Play hears the reverse/fade_in/fade_out too instead of
        the original audio."""
        start, end = self._effect_range
        start_idx, end_idx = self._preview_slice_indices(start, end)
        preview_audio = self._engine.audio_data.copy()
        preview_audio[start_idx:end_idx] = self._compute_effect(
            self._effect_mode, preview_audio[start_idx:end_idx]
        )
        self._effect_engine = _PlaybackEngine(preview_audio, self._sample_rate)

    def _update_effect_display(self):
        start, end = self._effect_range
        start_idx, end_idx = self._preview_slice_indices(start, end)
        for curve, (samples, _color, _label) in zip(self._curve_items, self._series):
            preview = samples.copy()
            preview[start_idx:end_idx] = self._compute_effect(
                self._effect_mode, preview[start_idx:end_idx]
            )
            curve.setData(self._times, preview)
        self._autoscale_y()

    def exit_effect_preview(self):
        if self._effect_mode is None:
            return
        if self._state != self.STATE_STOPPED and self._playback_engine() is self._effect_engine:
            self._do_reset()
        self._effect_mode = None
        self._effect_range = None
        self._effect_engine = None
        for curve, (samples, _color, _label) in zip(self._curve_items, self._series):
            curve.setData(self._times, samples)
        self._autoscale_y()
        self._refresh_segment_mode()

    def preview_channel_curve_effect(self, curve_index, effect, start, end):
        """Mirror another plot's reverse/fade_in/fade_out preview onto one
        overlaid curve here — the effect counterpart of preview_channel_curve.
        Also builds _preview_curve_engine so playback hears the effect too."""
        self._preview_curve_index = curve_index
        self._preview_curve_range = (start, end)
        self._preview_curve_effect = effect
        self._update_channel_curve_effect_preview()
        self._build_preview_curve_effect_engine(curve_index, effect, start, end)

    def _build_preview_curve_effect_engine(self, curve_index, effect, start, end):
        """Build _preview_curve_engine for this (Overall) player applying
        `effect` to only the `curve_index` column of the audio buffer.
        Skipped for mono-downmixed buffers (3+ channel clips) where the
        individual channel cannot be isolated."""
        audio = self._engine.audio_data
        if audio.ndim != 2 or curve_index >= audio.shape[1]:
            return
        start_idx, end_idx = self._preview_slice_indices(start, end)
        preview_audio = audio.copy()
        preview_audio[start_idx:end_idx, curve_index] = self._compute_effect(
            effect, preview_audio[start_idx:end_idx, curve_index]
        )
        self._preview_curve_engine = _PlaybackEngine(preview_audio, self._sample_rate)

    def _update_channel_curve_effect_preview(self):
        curve_index = self._preview_curve_index
        start, end = self._preview_curve_range
        start_idx, end_idx = self._preview_slice_indices(start, end)
        samples = self._series[curve_index][0]
        preview = samples.copy()
        preview[start_idx:end_idx] = self._compute_effect(
            self._preview_curve_effect, preview[start_idx:end_idx]
        )
        self._curve_items[curve_index].setData(self._times, preview)
        self._autoscale_y()

    def preview_channel_curve(self, curve_index, factor, start, end):
        """Live-preview another plot's vertical-scale edit by scaling just
        that one overlaid curve here too. Used on the Overall plot so a
        channel being edited stays visible even if it's currently drawn
        underneath an identical/overlapping channel.
        Also builds _preview_curve_engine so playback hears the scaling too."""
        self._preview_curve_index = curve_index
        self._preview_curve_range = (start, end)
        self._update_channel_curve_preview(factor)
        self._build_preview_curve_factor_engine(curve_index, factor, start, end)

    def _build_preview_curve_factor_engine(self, curve_index, factor, start, end):
        """Build _preview_curve_engine for vertical-scale preview on one
        channel column. Skipped for mono-downmixed buffers (3+ channel clips)."""
        audio = self._engine.audio_data
        if audio.ndim != 2 or curve_index >= audio.shape[1]:
            return
        start_idx, end_idx = self._preview_slice_indices(start, end)
        preview_audio = audio.copy()
        preview_audio[start_idx:end_idx, curve_index] = (
            preview_audio[start_idx:end_idx, curve_index] * factor
        )
        self._preview_curve_engine = _PlaybackEngine(preview_audio, self._sample_rate)

    def update_channel_curve_preview(self, factor: float):
        if self._preview_curve_index is not None:
            self._update_channel_curve_preview(factor)
            self._build_preview_curve_factor_engine(
                self._preview_curve_index, factor, *self._preview_curve_range
            )

    def _update_channel_curve_preview(self, factor: float):
        curve_index = self._preview_curve_index
        start, end = self._preview_curve_range
        start_idx, end_idx = self._preview_slice_indices(start, end)
        samples = self._series[curve_index][0]
        preview = samples.copy()
        preview[start_idx:end_idx] = preview[start_idx:end_idx] * factor
        self._curve_items[curve_index].setData(self._times, preview)
        self._autoscale_y()

    def exit_channel_curve_preview(self):
        if self._preview_curve_index is None:
            return
        if self._preview_curve_engine is not None:
            if self._state != self.STATE_STOPPED and self._playback_engine() is self._preview_curve_engine:
                self._do_reset()
            self._preview_curve_engine = None
        curve_index = self._preview_curve_index
        samples = self._series[curve_index][0]
        self._curve_items[curve_index].setData(self._times, samples)
        self._preview_curve_index = None
        self._preview_curve_range = None
        self._preview_curve_effect = None
        self._autoscale_y()

    def restore_selection_visual(self):
        """Re-show the selection highlight after exiting a preview mode."""
        if self._selected_segment is not None:
            self._selection_region.setRegion(self._selected_segment)
            self._selection_region.setVisible(True)

    def _update_clip_related_visibility(self):
        self._clip_btn.setVisible(
            self._show_clip_button
            and not self._clip_mode
            and not self._edit_mode
            and not self._tool_mode
        )
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
        if self._delete_divisor_mode:
            self.exit_delete_divisor_mode()
        if self._trim_mode:
            self.exit_trim_preview()
        if self._extract_mode:
            self.exit_extract_preview()
        if self._timescale_mode:
            self.exit_timescale_preview()
        if self._concat_mode:
            self.exit_concat_preview()
        if self._vscale_mode:
            self.exit_vertical_scale_preview()
        if self._effect_mode is not None:
            self.exit_effect_preview()
        if self._preview_curve_index is not None:     # ← add
            self.exit_channel_curve_preview()
        if self._noise_mode:
            self.exit_noise_range_mode()
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

    def set_tool_mode(self, enabled: bool):
        self._tool_mode = enabled
        self._update_clip_related_visibility()
        if not enabled:
            self.exit_delete_divisor_mode()

    # ---------- view helpers ----------

    def _show_overview(self):
        end = self._overview_end_time()
        self._plot.setXRange(self._start_time, max(end, self._start_time + 0.001), padding=0.02)
        self._playhead.setVisible(False)

    def _overview_end_time(self):
        """Right-hand boundary of the full (unwindowed) view. Accounts for
        whichever preview — if any — is currently changing the waveform's
        displayed width, so the plot re-fits correctly after play/pause/
        reset no matter which preview is active."""
        if self._timescale_mode:
            return self._timescale_preview_end
        if self._concat_mode:
            return self._concat_end
        return self._end_time

    def _show_window_centered_at(self, t):
        self._plot.setXRange(t - _HALF_WINDOW_SECONDS, t + _HALF_WINDOW_SECONDS, padding=0)
        self._playhead.setPos(t)
        self._playhead.setVisible(True)

    # ---------- own playback ----------

    def _on_play_clicked(self):
        if not self._group.request_play(self):
            return
        self._state = self.STATE_PLAYING
        self._playback_engine().start_or_resume()
        self._timer.start()
        self._update_button_visibility()
        self._refresh_clip_enabled()
        self._refresh_segment_mode()
        self._refresh_marker_mode()

    def _on_pause_clicked(self):
        self._playback_engine().pause()
        self._timer.stop()
        self._state = self.STATE_PAUSED
        if not self._trim_mode and not self._extract_mode:
            if self._edit_mode:
                self._playhead.setPos(self._current_center_time())
                self._playhead.setVisible(True)
            else:
                self._show_window_centered_at(self._current_center_time())
        self._update_button_visibility()
        self._refresh_clip_enabled()
        self._refresh_segment_mode()  
        self._refresh_marker_mode()

    def _on_reset_clicked(self):
        self._do_reset()

    def _do_reset(self):
        self._playback_engine().stop()
        self._timer.stop()
        self._state = self.STATE_STOPPED
        if self._trim_mode:
            self._trim_playhead.setPos(self._trim_start)
            self._trim_playhead.setVisible(False)
        elif self._extract_mode:
            self._extract_playhead.setPos(self._extract_start)
            self._extract_playhead.setVisible(False)
        elif self._concat_mode:
            self._playhead.setVisible(False)
            self._plot.setXRange(
                self._start_time,
                max(self._concat_end, self._start_time + 0.001),
                padding=0.02,
            )
        else:
            self._show_overview()
        self._update_button_visibility()
        self._group.release(self)
        self._refresh_clip_enabled()
        self._refresh_segment_mode()  
        self._refresh_marker_mode()

    def _on_tick(self):
        engine = self._playback_engine()
        if engine.is_finished():
            self._do_reset()
            return

        if self._trim_mode:
            current_time = self._trim_start + engine.current_frame() / self._sample_rate
            self._trim_playhead.setPos(current_time)
            self._trim_playhead.setVisible(True)
            return

        if self._extract_mode:
            current_time = self._extract_start + engine.current_frame() / self._sample_rate
            self._extract_playhead.setPos(current_time)
            self._extract_playhead.setVisible(True)
            return

        current_time = self._start_time + engine.current_frame() / self._sample_rate

        if self._edit_mode:
            self._playhead.setPos(current_time)
            self._playhead.setVisible(True)
        else:
            self._show_window_centered_at(current_time)

        if self._is_driver:
            self._group.broadcast_position(current_time)

    def _playback_engine(self):
        if self._trim_mode:
            return self._trim_engine
        if self._extract_mode:
            return self._extract_engine
        if self._concat_mode:
            return self._concat_engine
        if self._timescale_mode and self._timescale_engine is not None:
            return self._timescale_engine
        if self._vscale_mode and self._vscale_engine is not None:
            return self._vscale_engine
        if self._effect_mode is not None and self._effect_engine is not None:
            return self._effect_engine
        if self._preview_curve_engine is not None:
            return self._preview_curve_engine
        return self._engine

    def _current_center_time(self):
        return self._start_time + self._playback_engine().current_frame() / self._sample_rate

    def _clamp_time(self, t):
        upper = self._concat_end if (self._concat_mode and self._concat_end is not None) else self._end_time
        return max(self._start_time, min(t, upper))

    def _on_drag_start(self):
        if self._state not in (self.STATE_PLAYING, self.STATE_PAUSED):
            return
        self._drag_was_playing = (self._state == self.STATE_PLAYING)
        if self._drag_was_playing:
            self._timer.stop()
            self._playback_engine().pause()

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

        if self._extract_mode:
            current = self._extract_start + self._extract_engine.current_frame() / self._sample_rate
            new_time = max(self._extract_start, min(current + dt, self._extract_end))
            frame = int(round((new_time - self._extract_start) * self._sample_rate))
            self._extract_engine.seek(frame)
            self._extract_playhead.setPos(new_time)
            self._extract_playhead.setVisible(True)
            return

        old_center = self._current_center_time()
        # Edit mode: graph is fixed, so dragging right should move the line
        # right (direct mapping). Non-edit mode keeps the old "pan the image"
        # feel, where dragging right reveals earlier content.
        new_time = self._clamp_time(old_center + dt) if self._edit_mode else self._clamp_time(old_center - dt)
        actual_dt = new_time - old_center

        frame = int(round((new_time - self._start_time) * self._sample_rate))
        self._playback_engine().seek(frame)

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
            self._playback_engine().start_or_resume()
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
            elif not (self._trim_mode or self._extract_mode or self._concat_mode):
                self._show_overview()
            self._play_btn.setEnabled(True)
            return

        if active_player is self:
            return

        self._play_btn.setEnabled(False)