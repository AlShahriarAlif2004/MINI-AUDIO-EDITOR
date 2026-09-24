import threading

import numpy as np
import pyqtgraph as pg
import sounddevice as sd
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)


_TICK_MS = 33                    # UI refresh while recording (~30 fps) — same cadence as WaveformPlayer
_DISPLAY_WINDOW_SECONDS = 12.0    # how much history the scrolling plot shows at once
_DEFAULT_SAMPLE_RATE = 44100


class RecordingDialog(QDialog):
    """
    Records mono audio from the default microphone.

    The plot behaves like a scrolling strip-chart: it starts empty, new
    samples are drawn in at the right edge as they arrive, and everything
    already on screen is pushed left to make room — once the visible
    window fills up, the oldest audio scrolls off the left.

    The transport mirrors WaveformPlayer's existing play/pause/reset state
    machine exactly: a single Start button that becomes "Resume" once
    paused, a Pause button that only appears while recording, and a Reset
    button that appears whenever there's an in-progress recording to
    discard. Stop sits apart on the right — it ends the recording and
    closes the dialog (accepted) rather than just pausing it.
    """

    STATE_IDLE = "idle"
    STATE_RECORDING = "recording"
    STATE_PAUSED = "paused"

    def __init__(self, sample_rate=_DEFAULT_SAMPLE_RATE, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Record Audio")
        self.setMinimumSize(980, 340)

        self._sample_rate = sample_rate
        self._state = self.STATE_IDLE
        self._stream = None

        # Audio actually captured so far (permanent — survives pause/resume,
        # cleared only by Reset). Guarded by _lock because the PortAudio
        # callback fires on its own thread.
        self._lock = threading.Lock()
        self._pending_chunks = []    # not yet folded into _recorded_chunks
        self._recorded_chunks = []

        # Fixed-length rolling window used only for the on-screen display.
        self._display_len = max(1, int(_DISPLAY_WINDOW_SECONDS * sample_rate))
        self._display_buffer = np.zeros(self._display_len, dtype=np.float64)
        self._display_times = np.linspace(
            -_DISPLAY_WINDOW_SECONDS, 0.0, self._display_len
        )

        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._on_tick)

        self._build_ui()
        self._update_button_visibility()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self._plot = pg.PlotWidget()
        self._plot.setBackground(None)
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setYRange(-1.05, 1.05)
        self._plot.setXRange(-_DISPLAY_WINDOW_SECONDS, 0.0)
        self._plot.setLabel("bottom", "Time", "s")
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._plot.setMinimumHeight(200)
        self._curve = self._plot.plot(
            self._display_times, self._display_buffer,
            pen=pg.mkPen(color="#4fc3f7", width=1),
        )
        layout.addWidget(self._plot)

        controls = QHBoxLayout()
        controls.setSpacing(6)

        self._start_btn = QPushButton("⏺ Start")
        self._pause_btn = QPushButton("⏸ Pause")
        self._reset_btn = QPushButton("⟲ Reset")
        self._stop_btn = QPushButton("■ Stop")

        self._start_btn.clicked.connect(self._on_start_clicked)
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        self._reset_btn.clicked.connect(self._on_reset_clicked)
        self._stop_btn.clicked.connect(self._on_stop_clicked)

        for btn in (self._start_btn, self._pause_btn, self._reset_btn):
            btn.setFixedWidth(90)
            controls.addWidget(btn)

        controls.addStretch()

        self._stop_btn.setFixedWidth(90)
        self._stop_btn.setStyleSheet(
            "QPushButton { background-color: #e57373; color: white; font-weight: 600; }"
            "QPushButton:disabled { background-color: palette(button); color: palette(disabled-text); }"
        )
        controls.addWidget(self._stop_btn)

        layout.addLayout(controls)

        self._status_label = QLabel("Ready to record.")
        self._status_label.setStyleSheet("color: #9e9e9e; font-size: 12px;")
        layout.addWidget(self._status_label)

    def _update_button_visibility(self):
        recording = self._state == self.STATE_RECORDING
        paused = self._state == self.STATE_PAUSED

        self._start_btn.setVisible(not recording)
        self._start_btn.setText("⏺ Resume" if paused else "⏺ Start")
        self._pause_btn.setVisible(recording)
        self._reset_btn.setVisible(recording or paused)
        self._stop_btn.setEnabled(recording or paused)

    # ------------------------------------------------------------------
    # Microphone I/O
    # ------------------------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status):
        # Runs on PortAudio's own thread — never touch Qt widgets here,
        # only hand the data off under the lock for the timer to pick up.
        # PortAudio itself only speaks float32/int32/int16/int8/uint8 for
        # input devices (no float64), so we capture float32 and upcast
        # here; everything downstream still sees float64, same as every
        # other audio path in the app (WavIO, MP3IO, etc.).
        chunk = indata[:, 0].astype(np.float64, copy=True)
        with self._lock:
            self._pending_chunks.append(chunk)

    def _start_stream(self):
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()

    def _stop_stream(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_start_clicked(self):
        try:
            self._start_stream()
        except Exception as exc:
            QMessageBox.warning(
                self, "Recording Failed", f"Could not access the microphone:\n{exc}"
            )
            return

        self._state = self.STATE_RECORDING
        self._status_label.setText("Recording…")
        self._timer.start()
        self._update_button_visibility()

    def _on_pause_clicked(self):
        self._stop_stream()
        self._drain_pending()
        self._timer.stop()
        self._state = self.STATE_PAUSED
        self._status_label.setText("Paused.")
        self._update_button_visibility()

    def _on_reset_clicked(self):
        self._stop_stream()
        self._timer.stop()
        with self._lock:
            self._pending_chunks = []
            self._recorded_chunks = []
        self._display_buffer[:] = 0.0
        self._curve.setData(self._display_times, self._display_buffer)
        self._state = self.STATE_IDLE
        self._status_label.setText("Ready to record.")
        self._update_button_visibility()

    def _on_stop_clicked(self):
        self._stop_stream()
        self._drain_pending()
        self._timer.stop()
        self.accept()

    def _on_tick(self):
        new_samples = self._drain_pending()
        if new_samples is None or len(new_samples) == 0:
            return
        self._scroll_display(new_samples)

    # ------------------------------------------------------------------
    # Buffer helpers
    # ------------------------------------------------------------------

    def _drain_pending(self):
        """Folds anything the callback has queued into the permanent
        recording and returns just the newly folded-in samples (or None)."""
        with self._lock:
            if not self._pending_chunks:
                return None
            new_samples = np.concatenate(self._pending_chunks)
            self._pending_chunks = []
            self._recorded_chunks.append(new_samples)
        return new_samples

    def _scroll_display(self, new_samples):
        n = len(new_samples)
        if n >= self._display_len:
            self._display_buffer = new_samples[-self._display_len:].copy()
        else:
            self._display_buffer = np.concatenate(
                [self._display_buffer[n:], new_samples]
            )
        self._curve.setData(self._display_times, self._display_buffer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_recording(self):
        with self._lock:
            return bool(self._recorded_chunks) or bool(self._pending_chunks)

    def recorded_samples(self):
        self._drain_pending()
        with self._lock:
            if not self._recorded_chunks:
                return np.zeros(0, dtype=np.float64)
            return np.concatenate(self._recorded_chunks)

    @property
    def sample_rate(self):
        return self._sample_rate

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def reject(self):
        self._stop_stream()
        self._timer.stop()
        super().reject()

    def closeEvent(self, event):
        self._stop_stream()
        self._timer.stop()
        super().closeEvent(event)