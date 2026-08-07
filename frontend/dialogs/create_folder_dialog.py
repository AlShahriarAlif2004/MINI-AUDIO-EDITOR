from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QVBoxLayout,
)

from backend.workspace_model import Workspace


class CreateFolderDialog(QDialog):
    """Collect folder name."""

    def __init__(self, workspace: Workspace, parent=None):
        super().__init__(parent)
        self._workspace = workspace
        self._folder_name = ""

        self.setWindowTitle("Create Folder")
        self.setMinimumWidth(320)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Folder name")
        form.addRow("Name:", self._name_edit)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self):
        name = self._name_edit.text().strip()
        if not name:
            return
        self._folder_name = name
        self.accept()

    def folder_name(self) -> str:
        return self._folder_name
