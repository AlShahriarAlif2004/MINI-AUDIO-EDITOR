from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTabBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from backend.workspace_model import Entity, Folder, Workspace
from backend.audio_io import WavIO, MP3IO, OggIO
from frontend.dialogs.create_entity_dialog import CreateEntityDialog
from frontend.dialogs.create_folder_dialog import CreateFolderDialog
from frontend.dialogs.download_format_dialog import DownloadFormatDialog
from frontend.widgets.sidebar_tree import SidebarTree
from frontend.widgets.voice_entity_view import VoiceEntityView
from frontend.widgets.waveform_player import PlaybackGroup


class VoiceRecognitionPage(QWidget):
    """
    Voice Recognition mode's editor shell.

    Structurally a stripped-down sibling of WorkspacePage: the same File
    menu (New/Open/Exit), the same sidebar (tree + New Folder/New Entity
    + right-click Rename/Delete/Download), the same tab bar and
    "No Workspace" placeholder — but there is no Edit/Tool menu at all,
    and opening an entity tab shows VoiceEntityView (single averaged
    plot + speaker-recognition buttons) instead of EntityPlotView.
    """

    new_workspace_requested = Signal()
    open_workspace_requested = Signal()
    exit_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._workspace: Workspace | None = None
        self._build_ui()
        self._update_sidebar_for_workspace()
        self._playback_group = PlaybackGroup.get_instance()

    # ------------------------------------------------------------------
    # Sidebar actions — New Folder / New Entity
    # ------------------------------------------------------------------

    def _create_folder(self):
        self._create_folder_in(self._target_folder())

    def _create_entity(self):
        self._create_entity_in(self._target_folder())

    def _create_folder_in(self, parent: Folder | None):
        if not self._workspace or parent is None:
            return
        dialog = CreateFolderDialog(self._workspace, parent, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name = dialog.folder_name()
        if not name:
            return
        parent.add_child(Folder(name))
        self._workspace.mark_dirty()
        self._tree.populate(self._workspace.root)

    def _create_entity_in(self, parent: Folder | None):
        if not self._workspace or parent is None:
            return
        dialog = CreateEntityDialog(self._workspace, parent, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        entity = dialog.created_entity()
        if entity is None:
            return
        parent.add_child(entity)
        self._workspace.mark_dirty()
        self._tree.populate(self._workspace.root)
        self._open_entity_tab(entity)

    # ------------------------------------------------------------------
    # Sidebar right-click context menus
    # ------------------------------------------------------------------

    def _show_folder_context_menu(self, folder: Folder, global_pos):
        if not self._workspace:
            return

        menu = QMenu(self)
        new_entity_action = menu.addAction("New Entity")
        new_folder_action = menu.addAction("New Folder")
        menu.addSeparator()
        rename_action = menu.addAction("Rename")
        delete_action = menu.addAction("Delete")

        action = menu.exec(global_pos)
        if action is new_entity_action:
            self._create_entity_in(folder)
        elif action is new_folder_action:
            self._create_folder_in(folder)
        elif action is rename_action:
            self._rename_item(folder)
        elif action is delete_action:
            self._delete_item(folder)

    def _show_entity_context_menu(self, entity: Entity, global_pos):
        if not self._workspace:
            return

        menu = QMenu(self)
        rename_action = menu.addAction("Rename")
        delete_action = menu.addAction("Delete")
        download_action = menu.addAction("Download")

        action = menu.exec(global_pos)
        if action is rename_action:
            self._rename_item(entity)
        elif action is delete_action:
            self._delete_item(entity)
        elif action is download_action:
            self._download_entity(entity)

    def _rename_item(self, item):
        if not self._workspace or item.parent is None:
            return

        parent = item.parent
        while True:
            name, ok = QInputDialog.getText(
                self, "Rename", "New name:", QLineEdit.Normal, item.name
            )
            if not ok:
                return

            name = name.strip()
            if not name or name == item.name:
                return

            clash = parent.find_child_by_name(name)
            if clash is not None and clash is not item:
                QMessageBox.warning(
                    self,
                    "Rename Failed",
                    f'A folder or entity named "{name}" already exists here.',
                )
                continue
            break

        item.name = name
        if isinstance(item, Entity):
            index = self._find_entity_tab(item.id)
            if index != -1:
                self._tab_bar.setTabText(index, name)

        self._workspace.mark_dirty()
        self._tree.populate(self._workspace.root)

    def _delete_item(self, item):
        if not self._workspace or item.parent is None:
            return

        extra = " This will delete everything inside it." if isinstance(item, Folder) else ""
        reply = QMessageBox.question(
            self,
            "Delete",
            f'Are you sure you want to delete "{item.name}"?{extra}',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        for entity in self._collect_entities_under(item):
            index = self._find_entity_tab(entity.id)
            if index != -1:
                self._close_tab(index)

        item.parent.remove_child(item)
        self._workspace.mark_dirty()
        self._tree.populate(self._workspace.root)

    def _collect_entities_under(self, item) -> list[Entity]:
        if isinstance(item, Entity):
            return [item]

        entities: list[Entity] = []

        def walk(folder):
            for child in folder.children:
                if isinstance(child, Entity):
                    entities.append(child)
                else:
                    walk(child)

        walk(item)
        return entities

    def _download_entity(self, entity: Entity):
        format_dialog = DownloadFormatDialog(self)
        if format_dialog.exec() != QDialog.DialogCode.Accepted:
            return
        file_format = format_dialog.selected_format()   # "wav", "mp3", or "ogg"

        if file_format == "mp3":
            default_name = f"{entity.name}.mp3"
            file_filter = "MP3 Files (*.mp3)"
        elif file_format == "ogg":
            default_name = f"{entity.name}.ogg"
            file_filter = "OGG Files (*.ogg)"
        else:
            default_name = f"{entity.name}.wav"
            file_filter = "WAV Files (*.wav)"

        path, _ = QFileDialog.getSaveFileName(
            self, "Download Entity", default_name, file_filter,
        )
        if not path:
            return
        if not path.lower().endswith(f".{file_format}"):
            path += f".{file_format}"

        try:
            if file_format == "mp3":
                MP3IO.unload(entity.clip, path)
            elif file_format == "ogg":
                OggIO.unload(entity.clip, path)
            else:
                WavIO.unload(entity.clip, path)
        except (ValueError, OSError) as exc:
            QMessageBox.critical(self, "Download Failed", f"Could not download entity:\n{exc}")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

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
        self._file_btn.setStyleSheet(
            "QToolButton { background-color: #2f81f7; color: white; font-weight: 600; }"
        )

        self._file_menu = QMenu(self._file_btn)
        self._file_menu.addAction("New",  self.new_workspace_requested.emit)
        self._file_menu.addAction("Open", self.open_workspace_requested.emit)
        self._file_menu.addSeparator()
        self._file_menu.addAction("Exit", self.exit_requested.emit)

        top_bar.addWidget(self._file_btn)
        top_bar.addStretch()   # no Edit / Tool buttons in Voice Recognition mode
        root_layout.addLayout(top_bar)

        self._splitter = QSplitter(Qt.Horizontal)

        # ---- sidebar ----
        sidebar = QWidget()
        # Match the light color QTreeWidget normally paints itself with,
        # directly on the container, so the sidebar reads as distinct
        # from the editor pane even when the tree is hidden and only the
        # "No Workspace" label is shown (same fix as WorkspacePage).
        sidebar.setAutoFillBackground(True)
        sidebar.setStyleSheet("background-color: palette(base);")
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
        self._tree.folder_context_menu_requested.connect(self._show_folder_context_menu)
        self._tree.entity_context_menu_requested.connect(self._show_entity_context_menu)
        sidebar_layout.addWidget(self._tree)

        self._no_workspace_label = QLabel("No Workspace")
        self._no_workspace_label.setAlignment(Qt.AlignCenter)
        self._no_workspace_label.setStyleSheet(
            "color: white; font-style: italic; padding: 24px 0;"
        )
        sidebar_layout.addWidget(self._no_workspace_label)

        self._splitter.addWidget(sidebar)

        # ---- editor area ----
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

        self._splitter.addWidget(editor)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([240, 960])

        root_layout.addWidget(self._splitter)

    def _on_file_button_clicked(self):
        # No File/Edit/Tool mode toggle here — there is only ever "File"
        # mode, so the button's sole job is opening the menu.
        self._file_menu.exec(self._file_btn.mapToGlobal(self._file_btn.rect().bottomLeft()))

    # ------------------------------------------------------------------
    # Sidebar state
    # ------------------------------------------------------------------

    def _update_sidebar_for_workspace(self):
        """Tree/'No Workspace' placeholder are mutually exclusive. New
        Folder/New Entity stay visible always — just disabled until a
        workspace is loaded, rather than disappearing."""
        has_workspace = self._workspace is not None

        self._tree.setVisible(has_workspace)
        self._no_workspace_label.setVisible(not has_workspace)
        self._new_folder_btn.setEnabled(has_workspace)
        self._new_entity_btn.setEnabled(has_workspace)

    def _target_folder(self) -> Folder | None:
        if not self._workspace:
            return None
        selected = self._tree.selected_folder()
        return selected if selected is not None else self._workspace.root

    # ------------------------------------------------------------------
    # Workspace / tab management
    # ------------------------------------------------------------------

    def set_workspace(self, workspace: Workspace):
        self._workspace = workspace
        self._clear_tabs()
        self._tree.populate(workspace.root)
        self._content_stack.setCurrentWidget(self._empty_label)
        self._update_sidebar_for_workspace()
        self._splitter.setSizes([240, 960])

    def clear_workspace(self):
        """Return to the empty, no-workspace-loaded state — same idea as
        WorkspacePage.clear_workspace()."""
        self._workspace = None
        self._clear_tabs()
        self._tree.clear()
        self._content_stack.setCurrentWidget(self._empty_label)
        self._update_sidebar_for_workspace()
        self._splitter.setSizes([240, 960])

    def workspace(self) -> Workspace | None:
        return self._workspace

    def _clear_tabs(self):
        while self._tab_bar.count() > 0:
            self._tab_bar.removeTab(0)
        while self._content_stack.count() > 1:
            widget = self._content_stack.widget(1)
            self._content_stack.removeWidget(widget)
            widget.deleteLater()

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

        entity_view = VoiceEntityView(entity, self._workspace)
        entity_view.entity_modified.connect(self._on_entity_modified)
        self._content_stack.addWidget(entity_view)
        self._tab_bar.setCurrentIndex(index)
        self._content_stack.setCurrentWidget(entity_view)

    def _close_tab(self, index: int):
        widget_index = index + 1
        widget = self._content_stack.widget(widget_index)

        if widget:
            widget.shutdown()
            self._content_stack.removeWidget(widget)
            widget.deleteLater()
        self._tab_bar.removeTab(index)

    def _force_stop_active_playback(self):
        active = self._playback_group.active_player
        if active is not None and hasattr(active, "force_idle"):
            active.force_idle()

    def _on_tab_changed(self, index: int):
        self._force_stop_active_playback()

        if index < 0:
            self._content_stack.setCurrentWidget(self._empty_label)
            return

        widget_index = index + 1   # index 0 is the empty label
        if widget_index < self._content_stack.count():
            self._content_stack.setCurrentIndex(widget_index)
        else:
            self._content_stack.setCurrentWidget(self._empty_label)