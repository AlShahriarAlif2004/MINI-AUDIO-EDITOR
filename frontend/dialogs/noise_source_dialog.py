from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)

from backend.workspace_model import Entity, Workspace


class NoiseSourceDialog(QDialog):
    """
    First step of Noise Removal: ask the user where the noise profile
    should come from.

      - "Existing Signal" — the classic flow: draw a noise window on the
        entity's own Overall plot (Filter -> Apply, two stages).
      - "Entity Channel"  — borrow the noise profile from a channel of
        any entity in the workspace, picked from a dropdown. Mono
        entities are listed by name alone ("entity_1"); multi-channel
        entities list one entry per channel ("entity_1 - Channel 1").
        This path skips straight to Apply/Cancel — there's no
        range-selection stage since the whole reference channel is used.
    """

    def __init__(self, workspace: Workspace, current_entity: Entity, parent=None):
        super().__init__(parent)
        self._workspace = workspace
        self._current_entity = current_entity

        self.setWindowTitle("Noise Removal")
        self.setMinimumWidth(360)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        label = QLabel("Specify the noise from:")
        layout.addWidget(label)

        self._existing_radio = QRadioButton("Existing Signal")
        self._entity_radio = QRadioButton("Entity Channel")
        self._existing_radio.setChecked(True)
        layout.addWidget(self._existing_radio)
        layout.addWidget(self._entity_radio)

        self._entity_combo = QComboBox()
        for entity, channel_index in self._collect_entity_channels():
            if channel_index is None:
                text = entity.name
            else:
                text = f"{entity.name} - Channel {channel_index + 1}"
            self._entity_combo.addItem(text, (entity, channel_index))
        layout.addWidget(self._entity_combo)

        self._existing_radio.toggled.connect(self._update_combo_enabled)
        self._update_combo_enabled()

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

    def _collect_entity_channels(self) -> list[tuple[Entity, int | None]]:
        """One entry per channel of every entity in the workspace (mono
        entities contribute a single entry with channel_index=None)."""
        options: list[tuple[Entity, int | None]] = []

        def walk(folder):
            for child in folder.children:
                if isinstance(child, Entity):
                    if child.clip.num_channels == 1:
                        options.append((child, None))
                    else:
                        for i in range(child.clip.num_channels):
                            options.append((child, i))
                else:
                    walk(child)

        if self._workspace is not None:
            walk(self._workspace.root)
        return options

    def _update_combo_enabled(self):
        self._entity_combo.setVisible(self._entity_radio.isChecked())

    def _on_accept(self):
        if self._entity_radio.isChecked() and self._entity_combo.currentData() is None:
            return
        self.accept()

    # public API -----------------------------------------------------
    def selected_mode(self) -> str:
        """Returns 'entity' or 'existing'."""
        return "entity" if self._entity_radio.isChecked() else "existing"

    def selected_entity_channel(self) -> tuple[Entity | None, int | None]:
        data = self._entity_combo.currentData()
        if data is None:
            return None, None
        return data