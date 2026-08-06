from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, Signal


class HomePage(QWidget):
    """
    Landing page: app title + New Workspace / Open Workspace buttons.
    Emits signals; MainWindow decides what happens next.
    """

    new_workspace_requested = Signal()
    open_workspace_requested = Signal()

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

        new_btn = QPushButton("New Workspace")
        open_btn = QPushButton("Open Workspace")

        for btn in (new_btn, open_btn):
            btn.setFixedWidth(220)
            btn.setFixedHeight(40)

        new_btn.clicked.connect(self.new_workspace_requested.emit)
        open_btn.clicked.connect(self.open_workspace_requested.emit)

        layout.addWidget(title)
        layout.addWidget(new_btn, alignment=Qt.AlignCenter)
        layout.addWidget(open_btn, alignment=Qt.AlignCenter)