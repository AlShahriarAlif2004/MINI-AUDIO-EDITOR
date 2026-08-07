from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
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
from frontend.dialogs.create_folder_dialog import CreateFolderDialog
from frontend.widgets.sidebar_tree import SidebarTree
from frontend.widgets.entity_plot import EntityPlotView


class WorkspacePage(QWidget):
    """VSCode-like workspace editor shell."""

    new_workspace_requested = Signal()
    open_workspace_requested = Signal()
    exit_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._workspace: Workspace | None = None
        self._build_ui()

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(8, 6, 8, 6)

        self._menu_btn = QToolButton()
        self._menu_btn.setText("File")
        self._menu_btn.setFixedWidth(70)
        self._menu_btn.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self._menu_btn)
        menu.addAction("New", self.new_workspace_requested.emit)
        menu.addAction("Open", self.open_workspace_requested.emit)
        menu.addSeparator()
        menu.addAction("Exit", self.exit_requested.emit)
        self._menu_btn.setMenu(menu)
        self._menu_btn.setStyleSheet("QToolButton::menu-indicator { image: none; width: 0px; }")
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
        self._tab_bar.setTabsClosable(True)
        self._tab_bar.setExpanding(False)
        self._tab_bar.setElideMode(Qt.TextElideMode.ElideRight)
        self._tab_bar.currentChanged.connect(self._on_tab_changed)
        self._tab_bar.tabCloseRequested.connect(self._close_tab)
        self._tab_bar.setStyleSheet("""
            QTabBar::tab {
                padding: 6px 12px;
                min-width: 80px;
                max-width: 160px;
            }
            QTabBar::tab:selected {
                background-color: #2f81f7;
                color: white;
                font-weight: 600;
            }
            QTabBar::close-button {
                subcontrol-position: right;
                margin-left: 4px;
            }
            QTabBar::close-button:hover {
                background-color: rgba(128, 128, 128, 0.2);
                border-radius: 2px;
            }
        """)
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
            widget = self._content_stack.widget(1)
            self._content_stack.removeWidget(widget)
            widget.deleteLater()

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

        dialog = CreateFolderDialog(self._workspace, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        name = dialog.folder_name()
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

    def _find_entity_tab(self, entity_id: str) -> int:
        for i in range(self._tab_bar.count()):
            if self._tab_bar.tabData(i) == entity_id:
                return i
        return -1

    def _on_entity_modified(self):
        if self._workspace:
            self._workspace.mark_dirty()

    def _open_entity_tab(self, entity: Entity):
        index = self._find_entity_tab(entity.id)
        if index != -1:
            self._tab_bar.setCurrentIndex(index)
            return

        index = self._tab_bar.addTab(entity.name)
        self._tab_bar.setTabData(index, entity.id)

        entity_view = EntityPlotView(entity)
        entity_view.entity_modified.connect(self._on_entity_modified)
        self._content_stack.addWidget(entity_view)
        self._tab_bar.setCurrentIndex(index)
        self._content_stack.setCurrentWidget(entity_view)

    def _close_tab(self, index: int):
        widget_index = index + 1
        widget = self._content_stack.widget(widget_index)
        if widget:
            widget.shutdown()          # <-- release any playback/clip locks first
            self._content_stack.removeWidget(widget)
            widget.deleteLater()
        self._tab_bar.removeTab(index)

    def _on_tab_changed(self, index: int):
        if index < 0:
            self._content_stack.setCurrentWidget(self._empty_label)
            return

        widget_index = index + 1  # index 0 is the empty label
        if widget_index < self._content_stack.count():
            self._content_stack.setCurrentIndex(widget_index)
        else:
            self._content_stack.setCurrentWidget(self._empty_label)
