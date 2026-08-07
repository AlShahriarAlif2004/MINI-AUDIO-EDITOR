import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from backend.workspace_model import Entity, Workspace


class CreateEntityDialog(QDialog):
    """Collect entity name and audio source (file or existing entity)."""

    def __init__(self, workspace: Workspace, parent=None):
        super().__init__(parent)
        self._workspace = workspace
        self._entity: Entity | None = None

        self.setWindowTitle("Create Entity")
        self.setMinimumWidth(420)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Entity name")
        form.addRow("Name:", self._name_edit)
        layout.addLayout(form)

        self._from_file_radio = QRadioButton("From audio file")
        self._from_entity_radio = QRadioButton("From existing entity")
        self._from_file_radio.setChecked(True)
        layout.addWidget(self._from_file_radio)
        layout.addWidget(self._from_entity_radio)

        file_row = QHBoxLayout()
        self._file_edit = QLineEdit()
        self._file_edit.setPlaceholderText("Path to audio file")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_file)
        file_row.addWidget(self._file_edit)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        self._entity_combo = QComboBox()
        for entity in self._collect_entities():
            self._entity_combo.addItem(entity.name, entity)
        layout.addWidget(self._entity_combo)

        self._from_file_radio.toggled.connect(self._update_source_enabled)
        self._update_source_enabled()

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _collect_entities(self) -> list[Entity]:
        entities: list[Entity] = []

        def walk(folder):
            for child in folder.children:
                if isinstance(child, Entity):
                    entities.append(child)
                else:
                    walk(child)

        walk(self._workspace.root)
        return entities

    def _update_source_enabled(self):
        from_file = self._from_file_radio.isChecked()
        self._file_edit.setEnabled(from_file)
        self._entity_combo.setEnabled(not from_file)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3);;All Files (*)",
        )
        if path:
            self._file_edit.setText(path)

    def _on_accept(self):
        name = self._name_edit.text().strip()
        if not name:
            return

        try:
            if self._from_file_radio.isChecked():
                path = self._file_edit.text().strip()
                if not path or not os.path.isfile(path):
                    return
                self._entity = Entity.from_file(path, name=name)
            else:
                source = self._entity_combo.currentData()
                if source is None:
                    return
                self._entity = Entity.from_entity(source, new_name=name)
        except (ValueError, OSError):
            return

        self.accept()

    def created_entity(self) -> Entity | None:
        return self._entity
