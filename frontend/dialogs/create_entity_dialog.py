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
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QLabel
)

from backend.workspace_model import Entity, Workspace
from frontend.dialogs.recording_dialog import RecordingDialog
from frontend.dialogs.recording_preview_dialog import RecordingPreviewDialog


class CreateEntityDialog(QDialog):
    """Collect entity name and audio source (file, existing entity, or a
    fresh microphone recording)."""

    def __init__(self, workspace: Workspace, target_folder, parent=None):
        super().__init__(parent)
        self._workspace = workspace
        self._target_folder = target_folder
        self._entity: Entity | None = None

        self._recorded_samples = None
        self._recorded_sample_rate = None

        self.setWindowTitle("Create Entity")
        self.setMinimumWidth(420)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Entity name")
        self._name_edit.textChanged.connect(self._validate_name)
        form.addRow("Name:", self._name_edit)
        layout.addLayout(form)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #e57373;")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

        self._from_file_radio = QRadioButton("From audio file")
        self._from_entity_radio = QRadioButton("From existing entity")
        self._from_recording_radio = QRadioButton("From voice recording")
        self._from_file_radio.setChecked(True)
        layout.addWidget(self._from_file_radio)
        layout.addWidget(self._from_entity_radio)
        layout.addWidget(self._from_recording_radio)

        file_row = QHBoxLayout()
        self._file_edit = QLineEdit()
        self._file_edit.setPlaceholderText("Path to audio file")
        self._browse_btn = QPushButton("Browse")
        self._browse_btn.clicked.connect(self._browse_file)
        file_row.addWidget(self._file_edit)
        file_row.addWidget(self._browse_btn)
        layout.addLayout(file_row)

        self._entity_combo = QComboBox()
        for entity in self._collect_entities():
            self._entity_combo.addItem(entity.name, entity)
        layout.addWidget(self._entity_combo)

        self._build_recording_section(layout)

        self._from_file_radio.toggled.connect(self._update_source_enabled)
        self._from_entity_radio.toggled.connect(self._update_source_enabled)
        self._from_recording_radio.toggled.connect(self._update_source_enabled)
        self._update_source_enabled()

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._validate_name()

    # ------------------------------------------------------------------
    # "From voice recording" section
    # ------------------------------------------------------------------

    def _build_recording_section(self, layout):
        self._recording_section = QWidget()
        row = QHBoxLayout(self._recording_section)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        # Before a recording exists: a single button, stretched across
        # the dialog's full width.
        self._start_recording_btn = QPushButton("🎙 Start Recording")
        self._start_recording_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._start_recording_btn.clicked.connect(self._on_start_recording_clicked)
        row.addWidget(self._start_recording_btn, 1)

        # After a recording exists: two buttons splitting the row evenly
        # — re-record on the left, review the take on the right.
        self._record_again_btn = QPushButton("🎙 Record Again")
        self._record_again_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._record_again_btn.clicked.connect(self._on_start_recording_clicked)
        row.addWidget(self._record_again_btn, 1)

        self._show_recording_btn = QPushButton("👁 Show Recording")
        self._show_recording_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._show_recording_btn.clicked.connect(self._on_show_recording_clicked)
        row.addWidget(self._show_recording_btn, 1)

        self._record_again_btn.setVisible(False)
        self._show_recording_btn.setVisible(False)

        layout.addWidget(self._recording_section)

    def _on_start_recording_clicked(self):
        dialog = RecordingDialog(parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted or not dialog.has_recording():
            return

        self._recorded_samples = dialog.recorded_samples()
        self._recorded_sample_rate = dialog.sample_rate

        self._start_recording_btn.setVisible(False)
        self._record_again_btn.setVisible(True)
        self._show_recording_btn.setVisible(True)

    def _on_show_recording_clicked(self):
        if self._recorded_samples is None:
            return
        preview = RecordingPreviewDialog(
            self._recorded_samples, self._recorded_sample_rate, parent=self
        )
        preview.exec()

    # ------------------------------------------------------------------

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
        from_recording = self._from_recording_radio.isChecked()

        self._file_edit.setVisible(from_file)
        self._browse_btn.setVisible(from_file)
        self._entity_combo.setVisible(not from_file and not from_recording)
        self._recording_section.setVisible(from_recording)
        self.adjustSize()

    def _validate_name(self):
        name = self._name_edit.text().strip()
        error = None
        if name and self._target_folder.find_child_by_name(name):
            error = f'A folder or entity named "{name}" already exists here.'

        self._error_label.setText(error or "")
        self._error_label.setVisible(bool(error))
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(bool(name) and not error)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.ogg);;All Files (*)",
        )
        if path:
            self._file_edit.setText(path)

    def _on_accept(self):
        name = self._name_edit.text().strip()
        if not name or self._target_folder.find_child_by_name(name):
            self._validate_name()
            return

        try:
            if self._from_file_radio.isChecked():
                path = self._file_edit.text().strip()
                if not path or not os.path.isfile(path):
                    return
                self._entity = Entity.from_file(path, name=name)
            elif self._from_entity_radio.isChecked():
                source = self._entity_combo.currentData()
                if source is None:
                    return
                self._entity = Entity.from_entity(source, new_name=name)
            else:
                if self._recorded_samples is None or len(self._recorded_samples) == 0:
                    return
                self._entity = Entity.from_recording(
                    self._recorded_samples, self._recorded_sample_rate, name=name
                )
        except (ValueError, OSError):
            return

        self.accept()

    def created_entity(self) -> Entity | None:
        return self._entity