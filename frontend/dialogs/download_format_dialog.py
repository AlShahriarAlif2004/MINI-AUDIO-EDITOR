from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)


class DownloadFormatDialog(QDialog):
    """Ask the user which audio format to download an entity as.

    WAV and MP3 are offered as radio buttons. They're placed in an
    explicit QButtonGroup with exclusive=True so selecting one always
    clears the other, regardless of how the layout around them changes
    later — not relying on Qt's implicit same-parent auto-exclusivity.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Download Entity")
        self.setMinimumWidth(280)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        label = QLabel("Choose a file format:")
        layout.addWidget(label)

        self._wav_radio = QRadioButton("WAV")
        self._mp3_radio = QRadioButton("MP3")
        self._wav_radio.setChecked(True)

        self._format_group = QButtonGroup(self)
        self._format_group.setExclusive(True)
        self._format_group.addButton(self._wav_radio)
        self._format_group.addButton(self._mp3_radio)

        layout.addWidget(self._wav_radio)
        layout.addWidget(self._mp3_radio)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_format(self) -> str:
        """Returns 'wav' or 'mp3'."""
        return "mp3" if self._mp3_radio.isChecked() else "wav"