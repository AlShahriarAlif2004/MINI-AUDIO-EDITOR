from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QGraphicsDropShadowEffect,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QLinearGradient, QColor, QBrush, QPainter


class _ModeCard(QFrame):
    """A single clickable, hoverable mode tile (icon + label)."""

    clicked = Signal()

    def __init__(self, icon, label, accent, parent=None):
        super().__init__(parent)
        self.setObjectName("modeCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(172, 188)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(14)

        icon_label = QLabel(icon)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("font-size: 36px; background: transparent;")

        text_label = QLabel(label)
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setWordWrap(True)
        text_label.setStyleSheet(
            "font-size: 13px; font-weight: 600; color: #E8EAF0; background: transparent;"
        )

        layout.addWidget(icon_label)
        layout.addWidget(text_label)

        self.setStyleSheet(
            f"""
            QFrame#modeCard {{
                background-color: #1B1F27;
                border: 1px solid #2A2F39;
                border-radius: 16px;
            }}
            QFrame#modeCard:hover {{
                background-color: #20242E;
                border: 1px solid {accent};
            }}
            """
        )

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(0, 0, 0, 140))
        self.setGraphicsEffect(shadow)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class HomePage(QWidget):
    """
    Landing page: title at top, mode options laid out as a horizontal
    row of cards below it. Emits a signal per mode; MainWindow decides
    what happens next (enters the workspace editor with no workspace
    loaded — see WorkspacePage.clear_workspace).
    """

    editor_mode_requested = Signal()
    voice_recognition_mode_requested = Signal()
    echo_mode_requested = Signal()
    noise_removal_mode_requested = Signal()
    frequency_mode_requested = Signal()

    # (icon, label, signal-attr-name, accent color)
    _MODE_CARDS = [
        ("\u2702\ufe0f", "Editor Mode", "editor_mode_requested", "#6C8CFF"),
        ("\U0001F399\ufe0f", "Voice Recognition Mode", "voice_recognition_mode_requested", "#7ED9A6"),
        ("\U0001F501", "Echo Mode", "echo_mode_requested", "#F2A65A"),
        ("\U0001F9F9", "Noise Removal Mode", "noise_removal_mode_requested", "#E56B8C"),
        ("\U0001F4C8", "Frequency Mode", "frequency_mode_requested", "#9575CD"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(820, 560)
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 100, 0, 0)
        outer.setSpacing(150)
        outer.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        outer.addLayout(self._build_header())
        outer.addLayout(self._build_cards_row())
        outer.addStretch(1)

    def _build_header(self):
        header = QVBoxLayout()

        title = QLabel("Audiora")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 114px; font-weight: 700; color: #F5F6FA; letter-spacing: 1px; "
            "background: transparent;"
        )

        header.addWidget(title)
        return header

    def _build_cards_row(self):
        row = QHBoxLayout()
        row.setSpacing(22)
        row.setAlignment(Qt.AlignHCenter)

        for icon, label, signal_name, accent in self._MODE_CARDS:
            card = _ModeCard(icon, label, accent)
            card.clicked.connect(getattr(self, signal_name).emit)
            row.addWidget(card)

        return row

    # ------------------------------------------------------------------
    def paintEvent(self, event):
        """Dark gradient backdrop -- kept consistent with the other pages."""
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, QColor("#0C0E13"))
        gradient.setColorAt(1.0, QColor("#181D26"))
        painter.fillRect(self.rect(), QBrush(gradient))
        super().paintEvent(event)