from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator
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
from backend.audio_io import WavIO
from frontend.dialogs.create_entity_dialog import CreateEntityDialog
from frontend.dialogs.create_folder_dialog import CreateFolderDialog
from frontend.dialogs.concatenate_dialog import ConcatenateDialog
from frontend.widgets.sidebar_tree import SidebarTree
from frontend.widgets.entity_plot import EntityPlotView
from frontend.widgets.waveform_player import PlaybackGroup


# ---------------------------------------------------------------------------
# Compact factor input: [−] [ 1.0 ] [+]
# ---------------------------------------------------------------------------

class _FactorInput(QWidget):
    """A compact number input with decrement/increment buttons for the
    time-scale factor.  Emits value_changed(float) on every change."""

    value_changed = Signal(float)

    _MIN = 0.1
    _MAX = 99.9
    _STEP = 0.1

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 1.0
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        btn_style = (
            "QPushButton {"
            "  min-width: 24px; max-width: 24px;"
            "  min-height: 26px; max-height: 26px;"
            "  font-size: 15px; font-weight: 700;"
            "  border: 1px solid palette(mid);"
            "  background: palette(button);"
            "}"
            "QPushButton:hover  { background: palette(light); }"
            "QPushButton:pressed{ background: palette(dark);  }"
        )

        self._dec_btn = QPushButton("−")
        self._dec_btn.setStyleSheet(btn_style)
        self._dec_btn.clicked.connect(self._decrement)

        self._edit = QLineEdit("1.0")
        self._edit.setFixedWidth(58)
        self._edit.setFixedHeight(26)
        self._edit.setAlignment(Qt.AlignCenter)
        self._edit.setStyleSheet(
            "QLineEdit {"
            "  border-top: 1px solid palette(mid);"
            "  border-bottom: 1px solid palette(mid);"
            "  border-left: none; border-right: none;"
            "  border-radius: 0;"
            "  font-size: 13px;"
            "}"
        )
        validator = QDoubleValidator(self._MIN, self._MAX, 1, self._edit)
        validator.setNotation(QDoubleValidator.StandardNotation)
        self._edit.setValidator(validator)
        self._edit.textEdited.connect(self._on_text_edited)
        self._edit.editingFinished.connect(self._on_editing_finished)

        self._inc_btn = QPushButton("+")
        self._inc_btn.setStyleSheet(btn_style)
        self._inc_btn.clicked.connect(self._increment)

        layout.addWidget(self._dec_btn)
        layout.addWidget(self._edit)
        layout.addWidget(self._inc_btn)

    # ------------------------------------------------------------------
    def _decrement(self):
        self._set_value(round(max(self._MIN, self._value - self._STEP), 1))

    def _increment(self):
        self._set_value(round(min(self._MAX, self._value + self._STEP), 1))

    def _on_text_edited(self, text: str):
        try:
            val = float(text)
            if self._MIN <= val <= self._MAX:
                self._value = round(val, 1)
                self.value_changed.emit(self._value)
        except ValueError:
            pass

    def _on_editing_finished(self):
        try:
            val = float(self._edit.text())
            self._set_value(max(self._MIN, min(self._MAX, round(val, 1))))
        except ValueError:
            self._set_value(self._value)

    def _set_value(self, val: float):
        self._value = round(val, 1)
        self._edit.setText(f"{self._value:.1f}")
        self.value_changed.emit(self._value)

    # public API -------------------------------------------------------
    def value(self) -> float:
        return self._value

    def setValue(self, val: float):
        self._value = round(float(val), 1)
        self._edit.setText(f"{self._value:.1f}")


# ---------------------------------------------------------------------------
# Per-operation runtime state (one instance per entry in _OP_SPECS)
# ---------------------------------------------------------------------------

class _OpEntry:
    """Bundles the widgets + live state for a single edit-mode operation
    (Trim, Extract, Time Scale, ...). Built from an _OP_SPECS dict."""

    __slots__ = (
        "spec", "button", "panel", "apply_btn", "cancel_btn",
        "factor_input", "active", "active_view",
    )

    def __init__(self, spec, button, panel, apply_btn, cancel_btn, factor_input):
        self.spec = spec
        self.button = button
        self.panel = panel
        self.apply_btn = apply_btn
        self.cancel_btn = cancel_btn
        self.factor_input = factor_input
        self.active = False
        self.active_view = None


# ---------------------------------------------------------------------------
# WorkspacePage
# ---------------------------------------------------------------------------

class WorkspacePage(QWidget):
    """VSCode-like workspace editor shell."""

    _SELECTED_BTN_STYLE  = "QToolButton { background-color: #2f81f7; color: white; font-weight: 600; }"
    _SELECTED_OPTION_STYLE = "background-color: #2f81f7; color: white; font-weight: 600;"

    # Operations that are only valid when the *overall* plot is selected.
    _OVERALL_ONLY_OPS = {"Trim", "Extract", "Time Scale", "Concatenate"}

    # ------------------------------------------------------------------
    # Declarative description of the 8 edit-mode operations. Each entry
    # drives: panel construction, button wiring, mutual-exclusion
    # cancellation, and the generic apply/cancel handlers below.
    #
    #   key      -- internal identifier
    #   label    -- button text (also the _segment_buttons dict key)
    #   begin    -- EntityPlotView method name to start the op (no args,
    #               except "concat" which is special-cased: it opens a
    #               dialog first and passes the chosen sources)
    #   apply    -- EntityPlotView method name to commit the op
    #   cancel   -- EntityPlotView method name to abort/restore
    #   rebuild_signal -- EntityPlotView signal that means "content
    #               changed, rebuild the tab" (connected once per view)
    #   factor   -- True if this op has a live-preview factor spinner
    #   preview  -- EntityPlotView method name for live preview (only
    #               when factor is True)
    #   needs_dialog -- True only for Concatenate, which must collect
    #               source portions via ConcatenateDialog before begin
    # ------------------------------------------------------------------
    _OP_SPECS = [
        dict(key="trim", label="Trim",
             begin="begin_trim", apply="apply_trim", cancel="cancel_trim",
             rebuild_signal="entity_trimmed", factor=False, needs_dialog=False),
        dict(key="extract", label="Extract",
             begin="begin_extract", apply="apply_extract", cancel="cancel_extract",
             rebuild_signal="entity_extracted", factor=False, needs_dialog=False),
        dict(key="timescale", label="Time Scale",
             begin="begin_timescale", apply="apply_timescale", cancel="cancel_timescale",
             rebuild_signal="entity_timescaled", factor=True,
             preview="update_timescale_preview", needs_dialog=False),
        dict(key="vscale", label="Vertical Scale",
             begin="begin_vertical_scale", apply="apply_vertical_scale", cancel="cancel_vertical_scale",
             rebuild_signal="entity_vscaled", factor=True,
             preview="update_vertical_scale_preview", needs_dialog=False),
        dict(key="reverse", label="Reverse",
             begin="begin_reverse", apply="apply_reverse", cancel="cancel_reverse",
             rebuild_signal="entity_reversed", factor=False, needs_dialog=False),
        dict(key="fadein", label="Fade In",
             begin="begin_fade_in", apply="apply_fade_in", cancel="cancel_fade_in",
             rebuild_signal="entity_faded_in", factor=False, needs_dialog=False),
        dict(key="fadeout", label="Fade Out",
             begin="begin_fade_out", apply="apply_fade_out", cancel="cancel_fade_out",
             rebuild_signal="entity_faded_out", factor=False, needs_dialog=False),
        dict(key="concat", label="Concatenate",
             begin="begin_concatenate", apply="apply_concatenate", cancel="cancel_concatenate",
             rebuild_signal="entity_concatenated", factor=False, needs_dialog=True),
    ]

    # Tool-mode operations. Only entries with a matching label here get
    # real Apply/Cancel wiring; every other Tool button stays a plain
    # disabled placeholder (see _build_ui).
    _TOOL_OP_SPECS = [
        dict(key="deldivisor", label="Delete Divisor",
             begin="begin_delete_divisor", apply="apply_delete_divisor", cancel="cancel_delete_divisor",
             rebuild_signal="entity_divisor_deleted", factor=False, needs_dialog=False),
    ]

    new_workspace_requested = Signal()
    open_workspace_requested = Signal()
    exit_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._workspace: Workspace | None = None
        self._mode = "file"
        self._ops: dict[str, _OpEntry] = {}   # populated in _build_ui
        self._build_ui()
        self._playback_group = PlaybackGroup.get_instance()
        self._playback_group.active_changed.connect(self._on_playback_active_changed)

    def _create_folder(self):
        self._create_folder_in(self._target_folder())

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

    def _create_entity(self):
        self._create_entity_in(self._target_folder())

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
        if not self._workspace or self._mode != "file":
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
        if not self._workspace or self._mode != "file":
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

        # Close any open tabs for the entity, or every entity nested inside
        # a deleted folder, before it disappears from the workspace.
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
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Download Entity",
            f"{entity.name}.wav",
            "WAV Files (*.wav)",
        )
        if not path:
            return
        if not path.lower().endswith(".wav"):
            path += ".wav"

        try:
            WavIO.unload(entity.clip, path)
        except (ValueError, OSError) as exc:
            QMessageBox.critical(
                self,
                "Download Failed",
                f"Could not download entity:\n{exc}",
            )

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

        self._file_menu = QMenu(self._file_btn)
        self._file_menu.addAction("New",  self.new_workspace_requested.emit)
        self._file_menu.addAction("Open", self.open_workspace_requested.emit)
        self._file_menu.addSeparator()
        self._file_menu.addAction("Exit", self.exit_requested.emit)

        self._edit_btn = QToolButton()
        self._edit_btn.setText("Edit")
        self._edit_btn.setFixedWidth(70)
        self._edit_btn.clicked.connect(self._on_edit_button_clicked)

        self._tool_btn = QToolButton()
        self._tool_btn.setText("Tool")
        self._tool_btn.setFixedWidth(70)
        self._tool_btn.clicked.connect(self._on_tool_button_clicked)

        top_bar.addWidget(self._file_btn)
        top_bar.addWidget(self._edit_btn)
        top_bar.addWidget(self._tool_btn)
        top_bar.addStretch()
        root_layout.addLayout(top_bar)

        self._file_btn.setStyleSheet(self._SELECTED_BTN_STYLE)

        splitter = QSplitter(Qt.Horizontal)

        # ---- sidebar ----
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
        self._tree.folder_context_menu_requested.connect(self._show_folder_context_menu)
        self._tree.entity_context_menu_requested.connect(self._show_entity_context_menu)
        sidebar_layout.addWidget(self._tree)

        # ---- edit-options panel (always present; shown only in edit mode) ----
        self._edit_options_panel = QWidget()
        edit_options_layout = QVBoxLayout(self._edit_options_panel)
        edit_options_layout.setContentsMargins(4, 4, 4, 4)
        edit_options_layout.setSpacing(6)

        self._segment_label = QLabel("No selection")
        self._segment_label.setStyleSheet("font-weight: 600;")
        edit_options_layout.addWidget(self._segment_label)

        self._segment_buttons: dict[str, QPushButton] = {}

        # Section headers shown above the first operation of each group.
        _SECTION_TITLES = {
            "trim": "Cutting",
            "timescale": "Scaling",
            "reverse": "Reversing",
            "fadein": "Fading",
            "concat": "Concatenating",
        }

        for spec in self._OP_SPECS:
            section_title = _SECTION_TITLES.get(spec["key"])
            if section_title is not None:
                section_label = QLabel(section_title)
                section_label.setStyleSheet("font-weight: 600; margin-top: 8px;")
                edit_options_layout.addWidget(section_label)

            btn = QPushButton(spec["label"])
            btn.setEnabled(False)          # disabled until a segment is selected
            edit_options_layout.addWidget(btn)
            self._segment_buttons[spec["label"]] = btn

            panel, apply_btn, cancel_btn, factor_input = self._build_op_panel(spec)
            edit_options_layout.addWidget(panel)

            entry = _OpEntry(spec, btn, panel, apply_btn, cancel_btn, factor_input)
            self._ops[spec["key"]] = entry

            key = spec["key"]
            btn.clicked.connect(lambda checked=False, k=key: self._on_op_clicked(k))
            apply_btn.clicked.connect(lambda checked=False, k=key: self._on_op_apply_clicked(k))
            cancel_btn.clicked.connect(lambda checked=False, k=key: self._on_op_cancel_clicked(k))
            if factor_input is not None:
                factor_input.value_changed.connect(
                    lambda value, k=key: self._on_op_factor_changed(k, value)
                )

        edit_options_layout.addStretch()
        self._edit_options_panel.setVisible(False)   # shown when edit mode is active
        sidebar_layout.addWidget(self._edit_options_panel)

        # ---- tool-options panel (always present; shown only in tool mode) ----
        self._tool_options_panel = QWidget()
        tool_options_layout = QVBoxLayout(self._tool_options_panel)
        tool_options_layout.setContentsMargins(4, 4, 4, 4)
        tool_options_layout.setSpacing(6)

        _TOOL_SECTIONS = [
            ("General Tools", ["Delete Divisor"]),
            ("Filtering Tools", ["Noise Removal"]),
            ("Echo Tools", ["Add Echo", "Remove Echo"]),
        ]
        _tool_specs_by_label = {spec["label"]: spec for spec in self._TOOL_OP_SPECS}

        self._tool_buttons: dict[str, QPushButton] = {}

        for section_title, labels in _TOOL_SECTIONS:
            section_label = QLabel(section_title)
            section_label.setStyleSheet("font-weight: 600; margin-top: 8px;")
            tool_options_layout.addWidget(section_label)

            for label in labels:
                spec = _tool_specs_by_label.get(label)
                btn = QPushButton(label)
                tool_options_layout.addWidget(btn)
                self._tool_buttons[label] = btn

                if spec is None:
                    btn.setEnabled(False)   # not implemented yet
                    continue

                panel, apply_btn, cancel_btn, factor_input = self._build_op_panel(spec)
                tool_options_layout.addWidget(panel)

                entry = _OpEntry(spec, btn, panel, apply_btn, cancel_btn, factor_input)
                self._ops[spec["key"]] = entry

                key = spec["key"]
                btn.clicked.connect(lambda checked=False, k=key: self._on_op_clicked(k))
                apply_btn.clicked.connect(lambda checked=False, k=key: self._on_op_apply_clicked(k))
                cancel_btn.clicked.connect(lambda checked=False, k=key: self._on_op_cancel_clicked(k))

        tool_options_layout.addStretch()
        self._tool_options_panel.setVisible(False)   # shown when tool mode is active
        sidebar_layout.addWidget(self._tool_options_panel)

        splitter.addWidget(sidebar)

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

        splitter.addWidget(editor)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([240, 960])

        root_layout.addWidget(splitter)

    def _build_op_panel(self, spec):
        """Build the Apply/Cancel (and, for factor ops, the factor row)
        panel for one operation. Returns (panel, apply_btn, cancel_btn,
        factor_input_or_None)."""
        panel = QWidget()

        if spec["factor"]:
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(16, 0, 0, 0)
            layout.setSpacing(4)

            factor_row = QHBoxLayout()
            factor_row.setSpacing(6)
            factor_row.addWidget(QLabel("Factor"))
            factor_input = _FactorInput()
            factor_row.addWidget(factor_input)
            factor_row.addStretch()
            layout.addLayout(factor_row)

            btn_row = QHBoxLayout()
            btn_row.setSpacing(6)
            apply_btn = QPushButton("Apply")
            cancel_btn = QPushButton("Cancel")
            btn_row.addWidget(apply_btn)
            btn_row.addWidget(cancel_btn)
            layout.addLayout(btn_row)
        else:
            layout = QHBoxLayout(panel)
            layout.setContentsMargins(16, 0, 0, 0)
            layout.setSpacing(6)
            apply_btn = QPushButton("Apply")
            cancel_btn = QPushButton("Cancel")
            layout.addWidget(apply_btn)
            layout.addWidget(cancel_btn)
            factor_input = None

        panel.setVisible(False)
        return panel, apply_btn, cancel_btn, factor_input

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def _on_file_button_clicked(self):
        if self._mode == "file":
            self._file_menu.exec(self._file_btn.mapToGlobal(self._file_btn.rect().bottomLeft()))
            return
        self._set_mode("file")

    def _on_edit_button_clicked(self):
        self._set_mode("edit")

    def _on_tool_button_clicked(self):
        self._set_mode("tool")

    def _force_idle_all_players(self):
        for i in range(1, self._content_stack.count()):
            widget = self._content_stack.widget(i)
            if hasattr(widget, "shutdown"):
                widget.shutdown()

    def _clear_all_plot_selections(self):
        """Clear selection on all entity views and reset sidebar state."""
        if self._mode == "edit":
            # In edit mode keep the panel; just reset button/label state
            self._refresh_option_buttons(has_selection=False)
        else:
            self._edit_options_panel.setVisible(False)

        for i in range(1, self._content_stack.count()):
            widget = self._content_stack.widget(i)
            if hasattr(widget, "clear_all_selection"):
                widget.clear_all_selection()

    def _deselect_all_divisor_selections(self):
        """Deselect any highlighted divisor line on every open tab —
        called unconditionally on mode switch, before anything else."""
        for i in range(1, self._content_stack.count()):
            widget = self._content_stack.widget(i)
            if hasattr(widget, "clear_divisor_selection"):
                widget.clear_divisor_selection()

    def _on_playback_active_changed(self, active_player):
        """Any plot playing or paused anywhere freezes the whole sidebar
        options panel (Trim, Time Scale, Apply/Cancel, etc). It only comes
        back once every plot is fully stopped."""
        self._edit_options_panel.setEnabled(active_player is None)
        self._tool_options_panel.setEnabled(active_player is None)

    def _set_mode(self, mode: str):
        if mode == self._mode:
            return

        self._deselect_all_divisor_selections()
        self._cancel_all_ops()
        self._clear_all_plot_selections()

        if mode == "edit":
            self._force_idle_all_players()

        self._mode = mode

        self._file_btn.setStyleSheet(self._SELECTED_BTN_STYLE if mode == "file" else "")
        self._edit_btn.setStyleSheet(self._SELECTED_BTN_STYLE if mode == "edit" else "")
        self._tool_btn.setStyleSheet(self._SELECTED_BTN_STYLE if mode == "tool" else "")

        is_edit = (mode == "edit")
        is_tool = (mode == "tool")
        hides_tree = is_edit or is_tool

        self._new_folder_btn.setVisible(not hides_tree)
        self._new_entity_btn.setVisible(not hides_tree)
        self._tree.setVisible(not hides_tree)

        # Show / hide the two mutually-exclusive sidebar panels
        self._edit_options_panel.setVisible(is_edit)
        self._tool_options_panel.setVisible(is_tool)
        if is_edit:
            self._refresh_option_buttons(has_selection=False)

        for i in range(1, self._content_stack.count()):
            widget = self._content_stack.widget(i)
            if hasattr(widget, "set_edit_mode"):
                widget.set_edit_mode(is_edit)
            if hasattr(widget, "set_tool_mode"):
                widget.set_tool_mode(is_tool)

    # ------------------------------------------------------------------
    # Workspace / tab management
    # ------------------------------------------------------------------

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

    def _find_entity_tab(self, entity_id: str) -> int:
        for i in range(self._tab_bar.count()):
            if self._tab_bar.tabData(i) == entity_id:
                return i
        return -1

    def _on_entity_modified(self):
        if self._workspace:
            self._workspace.mark_dirty()

    # ------------------------------------------------------------------
    # Segment selection → sidebar state
    # ------------------------------------------------------------------

    def _refresh_option_buttons(self, has_selection: bool, channel_index=None):
        """Enable/disable every option button depending on what is selected."""
        if has_selection and channel_index is not None:
            label_text = f"Channel {channel_index + 1}"
        elif has_selection:
            label_text = "Overall"
        else:
            label_text = "No selection"
        self._segment_label.setText(label_text)

        is_overall = has_selection and (channel_index is None)

        for label, btn in self._segment_buttons.items():
            if not has_selection:
                btn.setEnabled(False)
            elif label in self._OVERALL_ONLY_OPS:
                btn.setEnabled(is_overall)
            else:
                btn.setEnabled(True)

    def _refresh_tab_bar_lock(self):
        locked = any(entry.active for entry in self._ops.values())
        self._tab_bar.setEnabled(not locked)

    def _on_segment_selected(self, entity, channel_index, start, end):
        # Update label to include the time range
        label_text = "Overall" if channel_index is None else f"Channel {channel_index + 1}"
        self._segment_label.setText(f"{label_text}: {start:.2f}s – {end:.2f}s")
        # Update button states
        is_overall = (channel_index is None)
        for label, btn in self._segment_buttons.items():
            if label in self._OVERALL_ONLY_OPS:
                btn.setEnabled(is_overall)
            else:
                btn.setEnabled(True)

    def _on_segment_deselected(self):
        # Don't hide the panel — just reset button/label state
        if self._mode == "edit":
            self._refresh_option_buttons(has_selection=False)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _current_entity_view(self) -> EntityPlotView | None:
        widget = self._content_stack.currentWidget()
        return widget if isinstance(widget, EntityPlotView) else None

    # ------------------------------------------------------------------
    # Generic edit-mode operation handling
    #
    # Every one of the 8 operations (Trim, Extract, Time Scale, Vertical
    # Scale, Reverse, Fade In, Fade Out, Concatenate) goes through these
    # same six methods, parameterized by its _OP_SPECS entry. Only
    # Concatenate has a real branch (it needs a dialog for source
    # selection before it can "begin").
    # ------------------------------------------------------------------

    def _collapse_op_ui(self, key: str):
        entry = self._ops[key]
        entry.active = False
        entry.active_view = None
        entry.button.setStyleSheet("")
        entry.panel.setVisible(False)
        if entry.factor_input is not None:
            entry.factor_input.setValue(1.0)
        self._refresh_tab_bar_lock()

    def _cancel_op_if_active(self, key: str):
        entry = self._ops[key]
        if entry.active and entry.active_view is not None:
            view = entry.active_view
            self._collapse_op_ui(key)
            getattr(view, entry.spec["cancel"])()
        elif entry.active:
            self._collapse_op_ui(key)

    def _cancel_all_ops(self, except_key: str | None = None):
        for key in self._ops:
            if key != except_key:
                self._cancel_op_if_active(key)

    def _on_op_clicked(self, key: str):
        entry = self._ops[key]
        if entry.active:
            return

        # Only one operation panel can be open at a time.
        self._cancel_all_ops(except_key=key)

        view = self._current_entity_view()
        if view is None:
            return

        if entry.spec["needs_dialog"]:
            # Concatenate: collect source portions via dialog first.
            channel_count = view.entity.clip.num_channels
            dialog = ConcatenateDialog(self._workspace, channel_count, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            sources = dialog.selected_portions()
            if not sources:
                return
            began = getattr(view, entry.spec["begin"])(sources)
        else:
            began = getattr(view, entry.spec["begin"])()

        if not began:
            return

        entry.active = True
        entry.active_view = view
        if entry.factor_input is not None:
            entry.factor_input.setValue(1.0)
        entry.button.setStyleSheet(self._SELECTED_OPTION_STYLE)
        entry.panel.setVisible(True)
        self._refresh_tab_bar_lock()

    def _on_op_factor_changed(self, key: str, factor: float):
        """Live-preview: update the plot instantly as the factor changes."""
        entry = self._ops[key]
        if entry.active and entry.active_view is not None:
            getattr(entry.active_view, entry.spec["preview"])(factor)

    def _on_op_apply_clicked(self, key: str):
        entry = self._ops[key]
        if not entry.active or entry.active_view is None:
            return
        view = entry.active_view
        factor = entry.factor_input.value() if entry.factor_input is not None else None
        self._collapse_op_ui(key)
        if factor is None:
            getattr(view, entry.spec["apply"])()          # → rebuild (clears selection)
        else:
            getattr(view, entry.spec["apply"])(factor)    # → rebuild (clears selection)

    def _on_op_cancel_clicked(self, key: str):
        entry = self._ops[key]
        if not entry.active or entry.active_view is None:
            return
        view = entry.active_view
        self._collapse_op_ui(key)
        getattr(view, entry.spec["cancel"])()              # restores graph + keeps selection

    # ------------------------------------------------------------------
    # Entity-view factory / tab lifecycle
    # ------------------------------------------------------------------

    def _make_entity_view(self, entity: Entity) -> EntityPlotView:
        entity_view = EntityPlotView(entity)
        entity_view.entity_modified.connect(self._on_entity_modified)
        entity_view.segment_selected.connect(self._on_segment_selected)
        entity_view.segment_deselected.connect(self._on_segment_deselected)
        for entry in self._ops.values():
            signal = getattr(entity_view, entry.spec["rebuild_signal"])
            signal.connect(lambda entity=entity: self._rebuild_entity_tab(entity))
        entity_view.set_edit_mode(self._mode == "edit")
        entity_view.set_tool_mode(self._mode == "tool")
        return entity_view

    def _open_entity_tab(self, entity: Entity):
        index = self._find_entity_tab(entity.id)
        if index != -1:
            self._tab_bar.setCurrentIndex(index)
            return

        index = self._tab_bar.addTab(entity.name)
        self._tab_bar.setTabData(index, entity.id)

        entity_view = self._make_entity_view(entity)
        self._content_stack.addWidget(entity_view)
        self._tab_bar.setCurrentIndex(index)
        self._content_stack.setCurrentWidget(entity_view)

    def _collapse_ops_for_widget(self, widget):
        """Collapse any operation panel that's mid-flight on `widget`,
        for every one of the 8 ops (not just Trim/Extract/Time Scale —
        see the discussion on the earlier stale-reference bug)."""
        for key, entry in self._ops.items():
            if widget is entry.active_view:
                self._collapse_op_ui(key)

    def _rebuild_entity_tab(self, entity: Entity):
        index = self._find_entity_tab(entity.id)
        if index == -1:
            return
        widget_index = index + 1
        old_widget   = self._content_stack.widget(widget_index)
        was_current  = (self._tab_bar.currentIndex() == index)

        self._collapse_ops_for_widget(old_widget)

        self._content_stack.removeWidget(old_widget)
        old_widget.deleteLater()

        entity_view = self._make_entity_view(entity)
        self._content_stack.insertWidget(widget_index, entity_view)
        if was_current:
            self._content_stack.setCurrentWidget(entity_view)

        # Reset selection state after rebuild
        if self._mode == "edit":
            self._refresh_option_buttons(has_selection=False)

    def _close_tab(self, index: int):
        widget_index = index + 1
        widget = self._content_stack.widget(widget_index)

        self._collapse_ops_for_widget(widget)

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
        self._cancel_all_ops()
        self._force_stop_active_playback()
        self._clear_all_plot_selections()

        if index < 0:
            self._content_stack.setCurrentWidget(self._empty_label)
            return

        widget_index = index + 1   # index 0 is the empty label
        if widget_index < self._content_stack.count():
            self._content_stack.setCurrentIndex(widget_index)
        else:
            self._content_stack.setCurrentWidget(self._empty_label)