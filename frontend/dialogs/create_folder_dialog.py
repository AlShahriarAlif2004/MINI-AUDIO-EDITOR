from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from backend.workspace_model import Folder, Workspace


class CreateFolderDialog(QDialog):
    """Collect folder name, rejecting names that already exist in target_folder."""

    def __init__(self, workspace: Workspace, target_folder: Folder, parent=None):
        super().__init__(parent)
        self._workspace = workspace
        self._target_folder = target_folder
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

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #e57373;")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._name_edit.textChanged.connect(self._validate)
        self._validate()

    def _validate(self):
        name = self._name_edit.text().strip()
        error = None
        if not name:
            error = None  # no error text yet, but OK stays disabled
        elif self._target_folder.find_child_by_name(name):
            error = f'A folder or entity named "{name}" already exists here.'

        self._error_label.setText(error or "")
        self._error_label.setVisible(bool(error))
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(bool(name) and not error)

    def _on_accept(self):
        name = self._name_edit.text().strip()
        if not name or self._target_folder.find_child_by_name(name):
            self._validate()
            return
        self._folder_name = name
        self.accept()

    def folder_name(self) -> str:
        return self._folder_name