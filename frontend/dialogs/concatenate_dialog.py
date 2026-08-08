from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from backend.workspace_model import Entity, Workspace


class ConcatenateDialog(QDialog):
    """
    Lets the user pick, in order, one or more *portions* to append. A
    portion is one of an entity's Overall-plot segments as bounded by its
    division markers (Entity.get_segments()) — so an entity divided at,
    say, t=2 into "0..2" and "2..5" offers two separate, independently
    selectable portions, not just the whole clip. Only portions belonging
    to entities whose channel count matches the target are offered.

    Picks are queued in a list (first pick on top); each row has a remove
    ("x") button. OK only becomes available once the list is non-empty.
    """

    def __init__(self, workspace: Workspace, channel_count: int, parent=None):
        super().__init__(parent)
        self._workspace = workspace
        self._channel_count = channel_count
        self._portions: list[tuple[Entity, float, float]] = []   # ordered, first pick on top

        self.setWindowTitle("Concatenate")
        self.setMinimumWidth(420)
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        info = QLabel(
            f"Pick portions to append, in order "
            f"(only portions of entities with {self._channel_count} channel"
            f"{'s' if self._channel_count != 1 else ''} are shown):"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        pick_row = QHBoxLayout()
        self._combo = QComboBox()
        for entity, seg_start, seg_end in self._collect_matching_portions():
            self._combo.addItem(
                f"{entity.name}: {seg_start:.2f}s – {seg_end:.2f}s",
                (entity, seg_start, seg_end),
            )
        self._add_btn = QPushButton("Add")
        self._add_btn.clicked.connect(self._on_add_clicked)
        pick_row.addWidget(self._combo, 1)
        pick_row.addWidget(self._add_btn)
        layout.addLayout(pick_row)

        self._list = QListWidget()
        layout.addWidget(self._list)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        layout.addWidget(self._buttons)

        self._add_btn.setEnabled(self._combo.count() > 0)

    def _collect_matching_portions(self) -> list[tuple[Entity, float, float]]:
        """One entry per division-bounded segment of every matching entity."""
        portions: list[tuple[Entity, float, float]] = []

        def walk(folder):
            for child in folder.children:
                if isinstance(child, Entity):
                    if child.clip.num_channels == self._channel_count:
                        for seg_start, seg_end in child.get_segments():
                            portions.append((child, seg_start, seg_end))
                else:
                    walk(child)

        if self._workspace is not None:
            walk(self._workspace.root)
        return portions

    # ------------------------------------------------------------------
    def _on_add_clicked(self):
        data = self._combo.currentData()
        if data is None:
            return
        entity, seg_start, seg_end = data

        self._portions.append((entity, seg_start, seg_end))

        label = QLabel(f"{entity.name}: {seg_start:.2f}s – {seg_end:.2f}s")

        remove_btn = QToolButton()
        remove_btn.setText("✕")
        remove_btn.setFixedSize(20, 20)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(6, 2, 6, 2)
        row_layout.addWidget(label, 1)
        row_layout.addWidget(remove_btn)

        item = QListWidgetItem()
        item.setSizeHint(row.sizeHint())
        self._list.addItem(item)
        self._list.setItemWidget(item, row)

        remove_btn.clicked.connect(lambda checked=False, it=item: self._on_remove_clicked(it))

        self._update_ok_enabled()

    def _on_remove_clicked(self, item: QListWidgetItem):
        index = self._list.row(item)
        if index < 0:
            return
        self._list.takeItem(index)
        del self._portions[index]
        self._update_ok_enabled()

    def _update_ok_enabled(self):
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(bool(self._portions))

    # public API ---------------------------------------------------------
    def selected_portions(self) -> list[tuple[Entity, float, float]]:
        """Ordered list of (source_entity, segment_start, segment_end)."""
        return list(self._portions)