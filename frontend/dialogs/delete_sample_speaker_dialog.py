from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

from backend.workspace_model import Entity


class DeleteSampleSpeakerDialog(QDialog):
    """
    "Delete Sample Speaker" step: lets the user check off one or more of
    the *current tab's* already-added sample speakers to remove. OK stays
    enabled even with nothing checked (nothing will be removed) but is
    disabled entirely if there are no sample speakers to offer in the
    first place.
    """

    def __init__(self, sample_speakers: list[Entity], parent=None):
        super().__init__(parent)
        self._sample_speakers = list(sample_speakers)

        self.setWindowTitle("Delete Sample Speaker")
        self.setMinimumWidth(320)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        label = QLabel("Select deleting speaker(s):")
        layout.addWidget(label)

        self._list = QListWidget()
        for entity in self._sample_speakers:
            item = QListWidgetItem(entity.name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, entity)
            self._list.addItem(item)
        layout.addWidget(self._list)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(bool(self._sample_speakers))
        layout.addWidget(self._buttons)

    # public API -------------------------------------------------------
    def selected_entities(self) -> list[Entity]:
        selected = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))
        return selected