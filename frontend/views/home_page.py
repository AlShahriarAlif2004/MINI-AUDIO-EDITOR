from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, Signal


class HomePage(QWidget):
    """
    Landing page: app title + a single Editor Mode button. Emits a
    signal; MainWindow decides what happens next (enters the workspace
    editor with no workspace loaded — see WorkspacePage.clear_workspace).
    """

    editor_mode_requested = Signal()
    voice_recognition_mode_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(24)

        title = QLabel("Mini Audio Editor")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: 600;")

        editor_btn = QPushButton("Editor Mode")
        voice_btn = QPushButton("Voice Recognition Mode")

        for btn in (editor_btn, voice_btn):
            btn.setFixedWidth(220)
            btn.setFixedHeight(40)

        editor_btn.clicked.connect(self.editor_mode_requested.emit)
        voice_btn.clicked.connect(self.voice_recognition_mode_requested.emit)

        layout.addWidget(title)
        layout.addWidget(editor_btn, alignment=Qt.AlignCenter)
        layout.addWidget(voice_btn, alignment=Qt.AlignCenter)