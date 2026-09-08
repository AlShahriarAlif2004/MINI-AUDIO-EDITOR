from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
)

from backend.workspace_model import Entity, Folder, Workspace


class AddSampleSpeakerDialog(QDialog):
    """
    "Add Sample Speaker" step of the Voice Recognition speaker-recognition
    workflow: lets the user pick any entity in the workspace to add to the
    current tab's Sample Speakers list.

    Every entity in the workspace is offered (unlike NoiseSourceDialog,
    there's no channel-count restriction here), listed by name. If the
    workspace has no entities at all, the combo box is empty and OK stays
    disabled — there's nothing valid to pick.
    """

    def __init__(self, workspace: Workspace, parent=None):
        super().__init__(parent)
        self._workspace = workspace

        self.setWindowTitle("Add Sample Speaker")
        self.setMinimumWidth(320)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        label = QLabel("Select sample speaker:")
        layout.addWidget(label)

        self._combo = QComboBox()
        for entity in self._collect_entities():
            self._combo.addItem(entity.name, entity)
        layout.addWidget(self._combo)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(self._combo.count() > 0)
        layout.addWidget(self._buttons)

    def _collect_entities(self) -> list[Entity]:
        entities: list[Entity] = []

        def walk(folder: Folder):
            for child in folder.children:
                if isinstance(child, Entity):
                    entities.append(child)
                else:
                    walk(child)

        if self._workspace is not None:
            walk(self._workspace.root)
        return entities

    # public API -------------------------------------------------------
    def selected_entity(self) -> Entity | None:
        return self._combo.currentData()