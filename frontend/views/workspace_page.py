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

    _SELECTED_BTN_STYLE = "QToolButton { background-color: #2f81f7; color: white; font-weight: 600; }"  # <-- added

    new_workspace_requested = Signal()
    open_workspace_requested = Signal()
    exit_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._workspace: Workspace | None = None
        self._mode = "file"
        self._build_ui()

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(8, 6, 8, 6)

        self._file_btn = QToolButton()
        self._file_btn.setText("File")
        self._file_btn.setFixedWidth(70)
        self._file_btn.clicked.connect(self._on_file_button_clicked)

        self._file_menu = QMenu(self._file_btn)
        self._file_menu.addAction("New", self.new_workspace_requested.emit)
        self._file_menu.addAction("Open", self.open_workspace_requested.emit)
        self._file_menu.addSeparator()
        self._file_menu.addAction("Exit", self.exit_requested.emit)

        self._edit_btn = QToolButton()
        self._edit_btn.setText("Edit")
        self._edit_btn.setFixedWidth(70)
        self._edit_btn.clicked.connect(self._on_edit_button_clicked)

        top_bar.addWidget(self._file_btn)
        top_bar.addWidget(self._edit_btn)
        top_bar.addStretch()
        root_layout.addLayout(top_bar)

        self._file_btn.setStyleSheet(self._SELECTED_BTN_STYLE)  # File is selected by default

        splitter = QSplitter(Qt.Horizontal)

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(4, 4, 4, 4)

        action_row = QHBoxLayout()
        self._new_folder_btn = QPushButton("New Folder")
        self._new_entity_btn = QPushButton("New Entity")
        self._new_folder_btn.clicked.connect(self._create_folder)
        self._new_entity_btn.clicked.connect(self._create_entity)
        action_row.addWidget(self._new_folder_btn)
        action_row.addWidget(self._new_entity_btn)
        sidebar_layout.addLayout(action_row)

        self._tree = SidebarTree()
        self._tree.entity_clicked.connect(self._open_entity_tab)
        sidebar_layout.addWidget(self._tree)

        self._edit_options_panel = QWidget()
        edit_options_layout = QVBoxLayout(self._edit_options_panel)
        edit_options_layout.setContentsMargins(4, 4, 4, 4)
        edit_options_layout.setSpacing(6)

        self._segment_label = QLabel("No selection")
        self._segment_label.setStyleSheet("font-weight: 600;")
        edit_options_layout.addWidget(self._segment_label)

        for label in ("Trim", "Time Scale", "Vertical Scale", "Reverse", "Shift",
                      "Fade In", "Fade Out", "Concatenate"):
            edit_options_layout.addWidget(QPushButton(label))  # wired up later

        edit_options_layout.addStretch()
        self._edit_options_panel.setVisible(False)
        sidebar_layout.addWidget(self._edit_options_panel)

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

    def _on_file_button_clicked(self):
        if self._mode == "file":
            # Already selected — clicking again opens the dropdown.
            self._file_menu.exec(self._file_btn.mapToGlobal(self._file_btn.rect().bottomLeft()))
            return
        self._set_mode("file")

    def _on_edit_button_clicked(self):
        self._set_mode("edit")

    def _force_idle_all_players(self):
        """Any plot that's playing/paused (in this tab or any other open
        tab) must go idle before we switch top-level modes."""
        for i in range(1, self._content_stack.count()):
            widget = self._content_stack.widget(i)
            if hasattr(widget, "shutdown"):
                widget.shutdown()

    def _clear_all_plot_selections(self):
        self._edit_options_panel.setVisible(False)
        for i in range(1, self._content_stack.count()):
            widget = self._content_stack.widget(i)
            if hasattr(widget, "clear_all_selection"):
                widget.clear_all_selection()

    def _set_mode(self, mode):
        if mode == self._mode:
            return

        self._clear_all_plot_selections()

        if mode == "edit":
            self._force_idle_all_players()

        self._mode = mode

        self._file_btn.setStyleSheet(self._SELECTED_BTN_STYLE if mode == "file" else "")
        self._edit_btn.setStyleSheet(self._SELECTED_BTN_STYLE if mode == "edit" else "")

        is_edit = (mode == "edit")
        self._new_folder_btn.setVisible(not is_edit)
        self._new_entity_btn.setVisible(not is_edit)
        self._tree.setVisible(not is_edit)

        for i in range(1, self._content_stack.count()):
            widget = self._content_stack.widget(i)
            if hasattr(widget, "set_edit_mode"):
                widget.set_edit_mode(is_edit)

    def set_workspace(self, workspace: Workspace):
        self._workspace = workspace
        self._clear_tabs()
        self._set_mode("file") 
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

    def _on_segment_selected(self, entity, channel_index, start, end):
        label = "Overall" if channel_index is None else f"Channel {channel_index + 1}"
        self._segment_label.setText(f"{label}: {start:.2f}s – {end:.2f}s")
        self._edit_options_panel.setVisible(True)

    def _on_segment_deselected(self):
        self._edit_options_panel.setVisible(False)

    def _open_entity_tab(self, entity: Entity):
        index = self._find_entity_tab(entity.id)
        if index != -1:
            self._tab_bar.setCurrentIndex(index)
            return

        index = self._tab_bar.addTab(entity.name)
        self._tab_bar.setTabData(index, entity.id)

        entity_view = EntityPlotView(entity)
        entity_view.entity_modified.connect(self._on_entity_modified)
        entity_view.segment_selected.connect(self._on_segment_selected)
        entity_view.segment_deselected.connect(self._on_segment_deselected)
        entity_view.set_edit_mode(self._mode == "edit")
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
        self._clear_all_plot_selections()

        if index < 0:
            self._content_stack.setCurrentWidget(self._empty_label)
            return

        widget_index = index + 1  # index 0 is the empty label
        if widget_index < self._content_stack.count():
            self._content_stack.setCurrentIndex(widget_index)
        else:
            self._content_stack.setCurrentWidget(self._empty_label)
