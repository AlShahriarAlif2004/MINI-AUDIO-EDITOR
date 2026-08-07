from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTabBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from backend.workspace_model import Entity, Folder, Workspace
from frontend.dialogs.create_entity_dialog import CreateEntityDialog
from frontend.widgets.sidebar_tree import SidebarTree


class WorkspacePage(QWidget):
    """VSCode-like workspace editor shell."""

    new_workspace_requested = Signal()
    open_workspace_requested = Signal()
    exit_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._workspace: Workspace | None = None
        self._entity_tabs: dict[str, int] = {}  # entity.id -> tab index
        self._tab_entities: dict[int, Entity] = {}  # tab index -> entity
        self._build_ui()

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(8, 6, 8, 6)

        self._menu_btn = QToolButton()
        self._menu_btn.setText("Menu")
        self._menu_btn.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self._menu_btn)
        menu.addAction("New", self.new_workspace_requested.emit)
        menu.addAction("Open", self.open_workspace_requested.emit)
        menu.addSeparator()
        menu.addAction("Exit", self.exit_requested.emit)
        self._menu_btn.setMenu(menu)
        top_bar.addWidget(self._menu_btn)
        top_bar.addStretch()
        root_layout.addLayout(top_bar)

        splitter = QSplitter(Qt.Horizontal)

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(4, 4, 4, 4)

        action_row = QHBoxLayout()
        new_folder_btn = QPushButton("New Folder")
        new_entity_btn = QPushButton("New Entity")
        new_folder_btn.clicked.connect(self._create_folder)
        new_entity_btn.clicked.connect(self._create_entity)
        action_row.addWidget(new_folder_btn)
        action_row.addWidget(new_entity_btn)
        sidebar_layout.addLayout(action_row)

        self._tree = SidebarTree()
        self._tree.entity_clicked.connect(self._open_entity_tab)
        sidebar_layout.addWidget(self._tree)

        splitter.addWidget(sidebar)

        editor = QWidget()
        editor_layout = QVBoxLayout(editor)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(0)

        self._tab_bar = QTabBar()
        self._tab_bar.setTabsClosable(False)
        self._tab_bar.currentChanged.connect(self._on_tab_changed)
        editor_layout.addWidget(self._tab_bar)

        self._content_stack = QStackedWidget()
        self._empty_label = QLabel("No entity selected")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._content_stack.addWidget(self._empty_label)
        editor_layout.addWidget(self._content_stack)

        splitter.addWidget(editor)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([240, 960])

        root_layout.addWidget(splitter)

    def set_workspace(self, workspace: Workspace):
        self._workspace = workspace
        self._clear_tabs()
        self._tree.populate(workspace.root)
        self._content_stack.setCurrentWidget(self._empty_label)

    def workspace(self) -> Workspace | None:
        return self._workspace

    def _clear_tabs(self):
        while self._tab_bar.count() > 0:
            self._tab_bar.removeTab(0)
        while self._content_stack.count() > 1:
            self._content_stack.removeWidget(self._content_stack.widget(1))
        self._entity_tabs.clear()
        self._tab_entities.clear()

    def _target_folder(self) -> Folder | None:
        if not self._workspace:
            return None

        selected = self._tree.selected_folder()
        return selected if selected is not None else self._workspace.root

    def _create_folder(self):
        if not self._workspace:
            return

        parent = self._target_folder()
        if parent is None:
            return

        name, ok = QInputDialog.getText(self, "New Folder", "Folder name:")
        if not ok:
            return

        name = name.strip()
        if not name or parent.find_child_by_name(name):
            return

        parent.add_child(Folder(name))
        self._workspace.mark_dirty()
        self._tree.populate(self._workspace.root)

    def _create_entity(self):
        if not self._workspace:
            return

        parent = self._target_folder()
        if parent is None:
            return

        dialog = CreateEntityDialog(self._workspace, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        entity = dialog.created_entity()
        if entity is None or parent.find_child_by_name(entity.name):
            return

        parent.add_child(entity)
        self._workspace.mark_dirty()
        self._tree.populate(self._workspace.root)
        self._open_entity_tab(entity)

    def _open_entity_tab(self, entity: Entity):
        if entity.id in self._entity_tabs:
            self._tab_bar.setCurrentIndex(self._entity_tabs[entity.id])
            return

        index = self._tab_bar.addTab(entity.name)
        placeholder = QLabel(f"Entity: {entity.name}")
        placeholder.setAlignment(Qt.AlignCenter)
        self._content_stack.addWidget(placeholder)

        self._entity_tabs[entity.id] = index
        self._tab_entities[index] = entity
        self._tab_bar.setCurrentIndex(index)

    def _on_tab_changed(self, index: int):
        if index < 0:
            self._content_stack.setCurrentWidget(self._empty_label)
            return

        widget_index = index + 1  # index 0 is the empty label
        if widget_index < self._content_stack.count():
            self._content_stack.setCurrentIndex(widget_index)
        else:
            self._content_stack.setCurrentWidget(self._empty_label)
