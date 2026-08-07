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

    def __init__(self):
        super().__init__()
        self.active_player = None

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
    """PlotWidget supporting relative (grab-and-drag) seeking, like dragging
    an image: cursor movement distance == content movement distance.
    Audio is only touched on drag start/end, never on every move, so the
    drag stays smooth even while the underlying player is playing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.drag_enabled = False
        self.on_drag_start = None    # callback()
        self.on_seek = None          # callback(time_delta: float) — relative, visual step
        self.on_drag_end = None      # callback()
        self._dragging = False
        self._last_x = None
        self._pixels_per_second = None

    def _compute_pixels_per_second(self):
        view_range = self.getPlotItem().vb.viewRange()[0]  # [xmin, xmax]
        span = view_range[1] - view_range[0]
        width = max(1, self.width())
        return width / span if span > 0 else 1.0

    def mousePressEvent(self, event):
        if self.drag_enabled and event.button() == Qt.LeftButton:
            self._dragging = True
            self._last_x = event.pos().x()
            self._pixels_per_second = self._compute_pixels_per_second()
            if self.on_drag_start:
                self.on_drag_start()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_enabled and self._dragging:
            dx = event.pos().x() - self._last_x
            self._last_x = event.pos().x()
            if self.on_seek and self._pixels_per_second:
                dt = -dx / self._pixels_per_second
                self.on_seek(dt)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drag_enabled and event.button() == Qt.LeftButton:
            self._dragging = False
            self._last_x = None
            self._pixels_per_second = None
            if self.on_drag_end:
                self.on_drag_end()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class WaveformPlayer(QWidget):
    """One waveform plot with its own Play/Pause/Reset + a disabled Clip button."""

    STATE_STOPPED = "stopped"
    STATE_PLAYING = "playing"
    STATE_PAUSED = "paused"

    def __init__(
        self,
        times: np.ndarray,
        series: list,               # [(samples, color, label), ...] — >1 entry only for 'Overall'
        sample_rate: int,
        audio_data: np.ndarray,     # what's actually played: mono for a channel, mixed for overall
        group: PlaybackGroup,
        is_driver: bool = False,    # True only for the 'Overall' plot
        parent=None,
    ):
        super().__init__(parent)
        self._times = times
        self._series = series
        self._sample_rate = sample_rate
        self._group = group
        self._is_driver = is_driver

        self._start_time = float(times[0]) if len(times) else 0.0
        self._end_time = float(times[-1]) if len(times) else 0.0

        self._engine = _PlaybackEngine(audio_data, sample_rate)
        self._state = self.STATE_STOPPED

        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._on_tick)

        self._build_ui()
        self._show_overview()

        self._group.active_changed.connect(self._on_group_active_changed)
        self._group.position_broadcast.connect(self._on_driver_position)

        # If a plot elsewhere (this tab or another) is already playing/paused
        # when this widget is created, reflect that lock immediately instead
        # of waiting for the next active_changed signal.
        self._on_group_active_changed(self._group.active_player)

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

        for samples, color, _label in self._series:
            self._plot.plot(self._times, samples, pen=pg.mkPen(color=color, width=1))

        self._playhead = pg.InfiniteLine(angle=90, pen=pg.mkPen(color="#ff4d4d", width=1.5))
        self._playhead.setVisible(False)
        self._plot.addItem(self._playhead)

        layout.addWidget(self._plot)

        controls = QHBoxLayout()
        controls.setSpacing(6)

        self._play_btn = QPushButton("▶ Play")
        self._pause_btn = QPushButton("⏸ Pause")
        self._reset_btn = QPushButton("⏹ Reset")
        self._clip_btn = QPushButton("✂ Clip")
        self._clip_btn.setEnabled(False)
        self._clip_btn.setToolTip("Coming soon")

        self._play_btn.clicked.connect(self._on_play_clicked)
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        self._reset_btn.clicked.connect(self._on_reset_clicked)

        for btn in (self._play_btn, self._pause_btn, self._reset_btn, self._clip_btn):
            btn.setFixedWidth(90)
            controls.addWidget(btn)
        controls.addStretch()

        layout.addLayout(controls)
        self._update_button_visibility()

    def _update_button_visibility(self):
        playing = self._state == self.STATE_PLAYING
        paused = self._state == self.STATE_PAUSED
        self._plot.drag_enabled = playing or paused
        self._play_btn.setVisible(not playing)
        self._play_btn.setText("▶ Resume" if paused else "▶ Play")
        self._pause_btn.setVisible(playing)
        self._reset_btn.setVisible(playing or paused)

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
            return  # another plot is active; must be reset first
        self._state = self.STATE_PLAYING
        self._engine.start_or_resume()
        self._timer.start()
        self._update_button_visibility()

    def _on_pause_clicked(self):
        self._engine.pause()
        self._timer.stop()
        self._state = self.STATE_PAUSED
        self._update_button_visibility()

    def _on_reset_clicked(self):
        self._do_reset()

    def _do_reset(self):
        self._engine.stop()
        self._timer.stop()
        self._state = self.STATE_STOPPED
        self._show_overview()
        self._update_button_visibility()
        self._group.release(self)

    def _on_tick(self):
        if self._engine.is_finished():
            self._do_reset()  # "finished" collapses back to the initial Play-only state
            return

        current_time = self._start_time + self._engine.current_frame() / self._sample_rate
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
            # Freeze audio for the duration of the drag — cheap, no stream
            # churn per mouse-move. self._engine.seek() below becomes a
            # pure bookkeeping update once segment_start is None.
            self._timer.stop()
            self._engine.pause()

    def _on_seek(self, dt):
        if self._state not in (self.STATE_PLAYING, self.STATE_PAUSED):
            return

        new_time = self._clamp_time(self._current_center_time() + dt)
        frame = int(round((new_time - self._start_time) * self._sample_rate))
        self._engine.seek(frame)          # cheap: no audio restart mid-drag
        self._show_window_centered_at(new_time)

        if self._is_driver:
            self._group.broadcast_position(new_time)

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
            return  # I'm the one driving; I already updated myself in _on_tick
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