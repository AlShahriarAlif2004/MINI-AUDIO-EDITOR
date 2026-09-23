import time

import numpy as np
import pyqtgraph as pg
import sounddevice as sd
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)


_TICK_MS = 33   # same refresh cadence as WaveformPlayer / RecordingDialog


class RecordingPreviewDialog(QDialog):
    """
    Read-only playback view of a completed recording: the full waveform,
    a play/pause/resume/reset transport (identical semantics to
    WaveformPlayer's own controls, including a moving playhead), and a
    Close button on the right.

    Opened from CreateEntityDialog's "Show Recording" button; it never
    mutates the recording, only plays it back.
    """

    def __init__(self, samples, sample_rate, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Recording Preview")
        self.setMinimumSize(980, 360)

        self._samples = np.asarray(samples, dtype=np.float64)
        self._sample_rate = sample_rate
        self._duration = len(self._samples) / sample_rate if sample_rate else 0.0
        self._times = (
            np.linspace(0.0, self._duration, len(self._samples))
            if len(self._samples) else np.zeros(0)
        )

        self._played_frames = 0
        self._segment_start = None
        self._state = "stopped"   # "stopped" | "playing" | "paused"

        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._on_tick)

        self._build_ui()
        self._update_buttons()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self._plot = pg.PlotWidget()
        self._plot.setBackground(None)
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setYRange(-1.05, 1.05)
        self._plot.setLabel("bottom", "Time", "s")
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._plot.setMinimumHeight(220)
        self._plot.plot(self._times, self._samples, pen=pg.mkPen(color="#4fc3f7", width=1))

        self._playhead = pg.InfiniteLine(angle=90, pen=pg.mkPen(color="#ff4d4d", width=1.5))
        self._playhead.addMarker('^', position=0.0, size=14)
        self._playhead.addMarker('v', position=1.0, size=14)
        self._playhead.setPos(0.0)
        self._playhead.setVisible(False)
        self._plot.addItem(self._playhead)

        layout.addWidget(self._plot)

        controls = QHBoxLayout()
        controls.setSpacing(6)

        self._play_btn = QPushButton("▶ Play")
        self._pause_btn = QPushButton("⏸ Pause")
        self._reset_btn = QPushButton("⟲ Reset")
        self._close_btn = QPushButton("✕ Close")

        self._play_btn.clicked.connect(self._on_play_clicked)
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        self._reset_btn.clicked.connect(self._on_reset_clicked)
        self._close_btn.clicked.connect(self._on_close_clicked)

        for btn in (self._play_btn, self._pause_btn, self._reset_btn):
            btn.setFixedWidth(90)
            controls.addWidget(btn)

        controls.addStretch()

        self._close_btn.setFixedWidth(90)
        controls.addWidget(self._close_btn)

        layout.addLayout(controls)

    def _update_buttons(self):
        playing = self._state == "playing"
        paused = self._state == "paused"

        self._play_btn.setVisible(not playing)
        self._play_btn.setText("▶ Resume" if paused else "▶ Play")
        self._pause_btn.setVisible(playing)
        self._reset_btn.setVisible(playing or paused)
        self._playhead.setVisible(playing or paused)

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _current_frame(self):
        if self._segment_start is None:
            return self._played_frames
        elapsed = time.monotonic() - self._segment_start
        frame = self._played_frames + int(elapsed * self._sample_rate)
        return min(frame, len(self._samples))

    def _on_play_clicked(self):
        if self._played_frames >= len(self._samples):
            self._played_frames = 0
        remaining = self._samples[self._played_frames:]
        if len(remaining) == 0:
            return

        sd.play(remaining, samplerate=self._sample_rate)
        self._segment_start = time.monotonic()
        self._state = "playing"
        self._timer.start()
        self._update_buttons()

    def _on_pause_clicked(self):
        self._played_frames = self._current_frame()
        sd.stop()
        self._segment_start = None
        self._state = "paused"
        self._timer.stop()
        self._update_buttons()

    def _on_reset_clicked(self):
        sd.stop()
        self._played_frames = 0
        self._segment_start = None
        self._state = "stopped"
        self._timer.stop()
        self._playhead.setPos(0.0)
        self._update_buttons()

    def _on_tick(self):
        frame = self._current_frame()
        t = frame / self._sample_rate if self._sample_rate else 0.0
        self._playhead.setPos(t)

        if frame >= len(self._samples):
            self._on_reset_clicked()

    def _on_close_clicked(self):
        sd.stop()
        self._timer.stop()
        self.accept()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def reject(self):
        sd.stop()
        self._timer.stop()
        super().reject()

    def closeEvent(self, event):
        sd.stop()
        self._timer.stop()
        super().closeEvent(event)