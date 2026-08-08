from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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
# WorkspacePage
# ---------------------------------------------------------------------------

class WorkspacePage(QWidget):
    """VSCode-like workspace editor shell."""

    _SELECTED_BTN_STYLE  = "QToolButton { background-color: #2f81f7; color: white; font-weight: 600; }"
    _SELECTED_OPTION_STYLE = "background-color: #2f81f7; color: white; font-weight: 600;"

    # Operations that are only valid when the *overall* plot is selected.
    _OVERALL_ONLY_OPS = {"Trim", "Extract", "Time Scale", "Concatenate"}

    new_workspace_requested = Signal()
    open_workspace_requested = Signal()
    exit_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._workspace: Workspace | None = None
        self._mode = "file"
        self._trim_active = False
        self._trim_active_view = None
        self._extract_active = False
        self._extract_active_view = None
        self._timescale_active = False
        self._timescale_active_view = None
        self._vscale_active = False
        self._vscale_active_view = None
        self._reverse_active = False
        self._reverse_active_view = None
        self._fadein_active = False
        self._fadein_active_view = None
        self._fadeout_active = False
        self._fadeout_active_view = None
        self._concat_active = False
        self._concat_active_view = None
        self._build_ui()
        self._playback_group = PlaybackGroup.get_instance()
        self._playback_group.active_changed.connect(self._on_playback_active_changed)

    def _create_folder(self):
        if not self._workspace:
            return
        parent = self._target_folder()
        if parent is None:
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
        if not self._workspace:
            return
        parent = self._target_folder()
        if parent is None:
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

        top_bar.addWidget(self._file_btn)
        top_bar.addWidget(self._edit_btn)
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

        for label in ("Trim", "Extract", "Time Scale", "Vertical Scale", "Reverse",
                      "Fade In", "Fade Out", "Concatenate"):
            btn = QPushButton(label)
            btn.setEnabled(False)          # disabled until a segment is selected
            edit_options_layout.addWidget(btn)
            self._segment_buttons[label] = btn

            if label == "Trim":
                self._trim_btn = btn
                btn.clicked.connect(self._on_trim_clicked)

                self._trim_panel = QWidget()
                trim_panel_layout = QHBoxLayout(self._trim_panel)
                trim_panel_layout.setContentsMargins(16, 0, 0, 0)
                trim_panel_layout.setSpacing(6)
                self._trim_apply_btn  = QPushButton("Apply")
                self._trim_cancel_btn = QPushButton("Cancel")
                self._trim_apply_btn.clicked.connect(self._on_trim_apply_clicked)
                self._trim_cancel_btn.clicked.connect(self._on_trim_cancel_clicked)
                trim_panel_layout.addWidget(self._trim_apply_btn)
                trim_panel_layout.addWidget(self._trim_cancel_btn)
                self._trim_panel.setVisible(False)
                edit_options_layout.addWidget(self._trim_panel)

            elif label == "Extract":
                self._extract_btn = btn
                btn.clicked.connect(self._on_extract_clicked)

                self._extract_panel = QWidget()
                extract_panel_layout = QHBoxLayout(self._extract_panel)
                extract_panel_layout.setContentsMargins(16, 0, 0, 0)
                extract_panel_layout.setSpacing(6)
                self._extract_apply_btn  = QPushButton("Apply")
                self._extract_cancel_btn = QPushButton("Cancel")
                self._extract_apply_btn.clicked.connect(self._on_extract_apply_clicked)
                self._extract_cancel_btn.clicked.connect(self._on_extract_cancel_clicked)
                extract_panel_layout.addWidget(self._extract_apply_btn)
                extract_panel_layout.addWidget(self._extract_cancel_btn)
                self._extract_panel.setVisible(False)
                edit_options_layout.addWidget(self._extract_panel)

            elif label == "Time Scale":
                self._timescale_btn = btn
                btn.clicked.connect(self._on_timescale_clicked)

                self._timescale_panel = QWidget()
                ts_panel_layout = QVBoxLayout(self._timescale_panel)
                ts_panel_layout.setContentsMargins(16, 0, 0, 0)
                ts_panel_layout.setSpacing(4)

                factor_row = QHBoxLayout()
                factor_row.setSpacing(6)
                factor_row.addWidget(QLabel("Factor"))
                self._factor_input = _FactorInput()
                self._factor_input.value_changed.connect(self._on_timescale_factor_changed)
                factor_row.addWidget(self._factor_input)
                factor_row.addStretch()
                ts_panel_layout.addLayout(factor_row)

                ts_btn_row = QHBoxLayout()
                ts_btn_row.setSpacing(6)
                self._ts_apply_btn  = QPushButton("Apply")
                self._ts_cancel_btn = QPushButton("Cancel")
                self._ts_apply_btn.clicked.connect(self._on_timescale_apply_clicked)
                self._ts_cancel_btn.clicked.connect(self._on_timescale_cancel_clicked)
                ts_btn_row.addWidget(self._ts_apply_btn)
                ts_btn_row.addWidget(self._ts_cancel_btn)
                ts_panel_layout.addLayout(ts_btn_row)

                self._timescale_panel.setVisible(False)
                edit_options_layout.addWidget(self._timescale_panel)

            elif label == "Vertical Scale":
                self._vscale_btn = btn
                btn.clicked.connect(self._on_vscale_clicked)

                self._vscale_panel = QWidget()
                vs_panel_layout = QVBoxLayout(self._vscale_panel)
                vs_panel_layout.setContentsMargins(16, 0, 0, 0)
                vs_panel_layout.setSpacing(4)

                vs_factor_row = QHBoxLayout()
                vs_factor_row.setSpacing(6)
                vs_factor_row.addWidget(QLabel("Factor"))
                self._vs_factor_input = _FactorInput()
                self._vs_factor_input.value_changed.connect(self._on_vscale_factor_changed)
                vs_factor_row.addWidget(self._vs_factor_input)
                vs_factor_row.addStretch()
                vs_panel_layout.addLayout(vs_factor_row)

                vs_btn_row = QHBoxLayout()
                vs_btn_row.setSpacing(6)
                self._vs_apply_btn  = QPushButton("Apply")
                self._vs_cancel_btn = QPushButton("Cancel")
                self._vs_apply_btn.clicked.connect(self._on_vscale_apply_clicked)
                self._vs_cancel_btn.clicked.connect(self._on_vscale_cancel_clicked)
                vs_btn_row.addWidget(self._vs_apply_btn)
                vs_btn_row.addWidget(self._vs_cancel_btn)
                vs_panel_layout.addLayout(vs_btn_row)

                self._vscale_panel.setVisible(False)
                edit_options_layout.addWidget(self._vscale_panel)

            elif label == "Reverse":
                self._reverse_btn = btn
                btn.clicked.connect(self._on_reverse_clicked)

                self._reverse_panel = QWidget()
                reverse_panel_layout = QHBoxLayout(self._reverse_panel)
                reverse_panel_layout.setContentsMargins(16, 0, 0, 0)
                reverse_panel_layout.setSpacing(6)
                self._reverse_apply_btn  = QPushButton("Apply")
                self._reverse_cancel_btn = QPushButton("Cancel")
                self._reverse_apply_btn.clicked.connect(self._on_reverse_apply_clicked)
                self._reverse_cancel_btn.clicked.connect(self._on_reverse_cancel_clicked)
                reverse_panel_layout.addWidget(self._reverse_apply_btn)
                reverse_panel_layout.addWidget(self._reverse_cancel_btn)
                self._reverse_panel.setVisible(False)
                edit_options_layout.addWidget(self._reverse_panel)

            elif label == "Fade In":
                self._fadein_btn = btn
                btn.clicked.connect(self._on_fadein_clicked)

                self._fadein_panel = QWidget()
                fadein_panel_layout = QHBoxLayout(self._fadein_panel)
                fadein_panel_layout.setContentsMargins(16, 0, 0, 0)
                fadein_panel_layout.setSpacing(6)
                self._fadein_apply_btn  = QPushButton("Apply")
                self._fadein_cancel_btn = QPushButton("Cancel")
                self._fadein_apply_btn.clicked.connect(self._on_fadein_apply_clicked)
                self._fadein_cancel_btn.clicked.connect(self._on_fadein_cancel_clicked)
                fadein_panel_layout.addWidget(self._fadein_apply_btn)
                fadein_panel_layout.addWidget(self._fadein_cancel_btn)
                self._fadein_panel.setVisible(False)
                edit_options_layout.addWidget(self._fadein_panel)

            elif label == "Fade Out":
                self._fadeout_btn = btn
                btn.clicked.connect(self._on_fadeout_clicked)

                self._fadeout_panel = QWidget()
                fadeout_panel_layout = QHBoxLayout(self._fadeout_panel)
                fadeout_panel_layout.setContentsMargins(16, 0, 0, 0)
                fadeout_panel_layout.setSpacing(6)
                self._fadeout_apply_btn  = QPushButton("Apply")
                self._fadeout_cancel_btn = QPushButton("Cancel")
                self._fadeout_apply_btn.clicked.connect(self._on_fadeout_apply_clicked)
                self._fadeout_cancel_btn.clicked.connect(self._on_fadeout_cancel_clicked)
                fadeout_panel_layout.addWidget(self._fadeout_apply_btn)
                fadeout_panel_layout.addWidget(self._fadeout_cancel_btn)
                self._fadeout_panel.setVisible(False)
                edit_options_layout.addWidget(self._fadeout_panel)

            elif label == "Concatenate":
                self._concat_btn = btn
                btn.clicked.connect(self._on_concatenate_clicked)

                self._concat_panel = QWidget()
                concat_panel_layout = QHBoxLayout(self._concat_panel)
                concat_panel_layout.setContentsMargins(16, 0, 0, 0)
                concat_panel_layout.setSpacing(6)
                self._concat_apply_btn  = QPushButton("Apply")
                self._concat_cancel_btn = QPushButton("Cancel")
                self._concat_apply_btn.clicked.connect(self._on_concat_apply_clicked)
                self._concat_cancel_btn.clicked.connect(self._on_concat_cancel_clicked)
                concat_panel_layout.addWidget(self._concat_apply_btn)
                concat_panel_layout.addWidget(self._concat_cancel_btn)
                self._concat_panel.setVisible(False)
                edit_options_layout.addWidget(self._concat_panel)

        edit_options_layout.addStretch()
        self._edit_options_panel.setVisible(False)   # shown when edit mode is active
        sidebar_layout.addWidget(self._edit_options_panel)

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

    def _on_playback_active_changed(self, active_player):
        """Any plot playing or paused anywhere freezes the whole sidebar
        options panel (Trim, Time Scale, Apply/Cancel, etc). It only comes
        back once every plot is fully stopped."""
        self._edit_options_panel.setEnabled(active_player is None)

    def _set_mode(self, mode: str):
        if mode == self._mode:
            return

        self._cancel_trim_if_active()
        self._cancel_extract_if_active()
        self._cancel_timescale_if_active()
        self._cancel_vscale_if_active()
        self._cancel_reverse_if_active()
        self._cancel_fadein_if_active()
        self._cancel_fadeout_if_active()
        self._cancel_concat_if_active()
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

        # Show / hide edit-options panel
        self._edit_options_panel.setVisible(is_edit)
        if is_edit:
            self._refresh_option_buttons(has_selection=False)

        for i in range(1, self._content_stack.count()):
            widget = self._content_stack.widget(i)
            if hasattr(widget, "set_edit_mode"):
                widget.set_edit_mode(is_edit)

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
        locked = (
            self._trim_active or self._extract_active or self._timescale_active or self._vscale_active
            or self._reverse_active or self._fadein_active or self._fadeout_active
            or self._concat_active
        )
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
    # Trim
    # ------------------------------------------------------------------

    def _collapse_trim_ui(self):
        self._trim_active = False
        self._trim_active_view = None
        self._trim_btn.setStyleSheet("")
        self._trim_panel.setVisible(False)
        self._refresh_tab_bar_lock()

    def _cancel_trim_if_active(self):
        if self._trim_active and self._trim_active_view is not None:
            view = self._trim_active_view
            self._collapse_trim_ui()
            view.cancel_trim()
        elif self._trim_active:
            self._collapse_trim_ui()

    def _on_trim_cancelled_externally(self):
        if self._trim_active:
            self._collapse_trim_ui()

    def _on_trim_clicked(self):
        if self._trim_active:
            return
        # Fold timescale if it was open
        self._cancel_extract_if_active()
        self._cancel_timescale_if_active()
        self._cancel_vscale_if_active()
        self._cancel_reverse_if_active()
        self._cancel_fadein_if_active()
        self._cancel_fadeout_if_active()
        self._cancel_concat_if_active()
        view = self._current_entity_view()
        if view is None or not view.begin_trim():
            return
        self._trim_active = True
        self._trim_active_view = view
        self._trim_btn.setStyleSheet(self._SELECTED_OPTION_STYLE)
        self._trim_panel.setVisible(True)
        self._refresh_tab_bar_lock()

    def _on_trim_apply_clicked(self):
        if not self._trim_active or self._trim_active_view is None:
            return
        view = self._trim_active_view
        self._collapse_trim_ui()
        view.apply_trim()          # entity_trimmed → _rebuild_entity_tab (clears selection)

    def _on_trim_cancel_clicked(self):
        if not self._trim_active or self._trim_active_view is None:
            return
        view = self._trim_active_view
        self._collapse_trim_ui()
        view.cancel_trim()         # restores graph + keeps selection

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------

    def _collapse_extract_ui(self):
        self._extract_active = False
        self._extract_active_view = None
        self._extract_btn.setStyleSheet("")
        self._extract_panel.setVisible(False)
        self._refresh_tab_bar_lock()

    def _cancel_extract_if_active(self):
        if self._extract_active and self._extract_active_view is not None:
            view = self._extract_active_view
            self._collapse_extract_ui()
            view.cancel_extract()
        elif self._extract_active:
            self._collapse_extract_ui()

    def _on_extract_clicked(self):
        if self._extract_active:
            return
        # Fold any other open op
        self._cancel_trim_if_active()
        self._cancel_timescale_if_active()
        self._cancel_vscale_if_active()
        self._cancel_reverse_if_active()
        self._cancel_fadein_if_active()
        self._cancel_fadeout_if_active()
        self._cancel_concat_if_active()
        view = self._current_entity_view()
        if view is None or not view.begin_extract():
            return
        self._extract_active = True
        self._extract_active_view = view
        self._extract_btn.setStyleSheet(self._SELECTED_OPTION_STYLE)
        self._extract_panel.setVisible(True)
        self._refresh_tab_bar_lock()

    def _on_extract_apply_clicked(self):
        if not self._extract_active or self._extract_active_view is None:
            return
        view = self._extract_active_view
        self._collapse_extract_ui()
        view.apply_extract()       # entity_extracted → _rebuild_entity_tab (clears selection)

    def _on_extract_cancel_clicked(self):
        if not self._extract_active or self._extract_active_view is None:
            return
        view = self._extract_active_view
        self._collapse_extract_ui()
        view.cancel_extract()      # restores graph + keeps selection

    # ------------------------------------------------------------------
    # Time Scale
    # ------------------------------------------------------------------

    def _collapse_timescale_ui(self):
        self._timescale_active = False
        self._timescale_active_view = None
        self._timescale_btn.setStyleSheet("")
        self._timescale_panel.setVisible(False)
        self._factor_input.setValue(1.0)
        self._refresh_tab_bar_lock()

    def _cancel_timescale_if_active(self):
        if self._timescale_active and self._timescale_active_view is not None:
            view = self._timescale_active_view
            self._collapse_timescale_ui()
            view.cancel_timescale()   # restores graph + keeps selection
        elif self._timescale_active:
            self._collapse_timescale_ui()

    def _on_timescale_clicked(self):
        if self._timescale_active:
            return
        # Fold trim if it was open
        self._cancel_trim_if_active()
        self._cancel_extract_if_active()
        self._cancel_vscale_if_active()
        self._cancel_reverse_if_active()
        self._cancel_fadein_if_active()
        self._cancel_fadeout_if_active()
        self._cancel_concat_if_active()
        view = self._current_entity_view()
        if view is None or not view.begin_timescale():
            return
        self._timescale_active = True
        self._timescale_active_view = view
        self._factor_input.setValue(1.0)
        self._timescale_btn.setStyleSheet(self._SELECTED_OPTION_STYLE)
        self._timescale_panel.setVisible(True)
        self._refresh_tab_bar_lock()

    def _on_timescale_factor_changed(self, factor: float):
        """Live-preview: update plot instantly as the factor changes."""
        if self._timescale_active and self._timescale_active_view is not None:
            self._timescale_active_view.update_timescale_preview(factor)

    def _on_timescale_apply_clicked(self):
        if not self._timescale_active or self._timescale_active_view is None:
            return
        factor = self._factor_input.value()
        view   = self._timescale_active_view
        self._collapse_timescale_ui()
        view.apply_timescale(factor)   # entity_timescaled → rebuild (clears selection)

    def _on_timescale_cancel_clicked(self):
        if not self._timescale_active or self._timescale_active_view is None:
            return
        view = self._timescale_active_view
        self._collapse_timescale_ui()
        view.cancel_timescale()        # restores graph + keeps selection

    # ------------------------------------------------------------------
    # Vertical Scale
    # ------------------------------------------------------------------
    
    def _collapse_vscale_ui(self):
        self._vscale_active = False
        self._vscale_active_view = None
        self._vscale_btn.setStyleSheet("")
        self._vscale_panel.setVisible(False)
        self._vs_factor_input.setValue(1.0)
        self._refresh_tab_bar_lock()
    
    def _cancel_vscale_if_active(self):
        if self._vscale_active and self._vscale_active_view is not None:
            view = self._vscale_active_view
            self._collapse_vscale_ui()
            view.cancel_vertical_scale()
        elif self._vscale_active:
            self._collapse_vscale_ui()
    
    def _on_vscale_clicked(self):
        if self._vscale_active:
            return
        self._cancel_trim_if_active()
        self._cancel_extract_if_active()
        self._cancel_timescale_if_active()
        self._cancel_reverse_if_active()
        self._cancel_fadein_if_active()
        self._cancel_fadeout_if_active()
        self._cancel_concat_if_active()
        view = self._current_entity_view()
        if view is None or not view.begin_vertical_scale():
            return
        self._vscale_active = True
        self._vscale_active_view = view
        self._vs_factor_input.setValue(1.0)
        self._vscale_btn.setStyleSheet(self._SELECTED_OPTION_STYLE)
        self._vscale_panel.setVisible(True)
        self._refresh_tab_bar_lock()
    
    def _on_vscale_factor_changed(self, factor: float):
        if self._vscale_active and self._vscale_active_view is not None:
            self._vscale_active_view.update_vertical_scale_preview(factor)
    
    def _on_vscale_apply_clicked(self):
        if not self._vscale_active or self._vscale_active_view is None:
            return
        factor = self._vs_factor_input.value()
        view   = self._vscale_active_view
        self._collapse_vscale_ui()
        view.apply_vertical_scale(factor)
    
    def _on_vscale_cancel_clicked(self):
        if not self._vscale_active or self._vscale_active_view is None:
            return
        view = self._vscale_active_view
        self._collapse_vscale_ui()
        view.cancel_vertical_scale()

    # ------------------------------------------------------------------
    # Reverse
    # ------------------------------------------------------------------

    def _collapse_reverse_ui(self):
        self._reverse_active = False
        self._reverse_active_view = None
        self._reverse_btn.setStyleSheet("")
        self._reverse_panel.setVisible(False)
        self._refresh_tab_bar_lock()

    def _cancel_reverse_if_active(self):
        if self._reverse_active and self._reverse_active_view is not None:
            view = self._reverse_active_view
            self._collapse_reverse_ui()
            view.cancel_effect()
        elif self._reverse_active:
            self._collapse_reverse_ui()

    def _on_reverse_clicked(self):
        if self._reverse_active:
            return
        self._cancel_trim_if_active()
        self._cancel_extract_if_active()
        self._cancel_timescale_if_active()
        self._cancel_vscale_if_active()
        self._cancel_fadein_if_active()
        self._cancel_fadeout_if_active()
        self._cancel_concat_if_active()
        view = self._current_entity_view()
        if view is None or not view.begin_reverse():
            return
        self._reverse_active = True
        self._reverse_active_view = view
        self._reverse_btn.setStyleSheet(self._SELECTED_OPTION_STYLE)
        self._reverse_panel.setVisible(True)
        self._refresh_tab_bar_lock()

    def _on_reverse_apply_clicked(self):
        if not self._reverse_active or self._reverse_active_view is None:
            return
        view = self._reverse_active_view
        self._collapse_reverse_ui()
        view.apply_effect()

    def _on_reverse_cancel_clicked(self):
        if not self._reverse_active or self._reverse_active_view is None:
            return
        view = self._reverse_active_view
        self._collapse_reverse_ui()
        view.cancel_effect()

    # ------------------------------------------------------------------
    # Fade In
    # ------------------------------------------------------------------

    def _collapse_fadein_ui(self):
        self._fadein_active = False
        self._fadein_active_view = None
        self._fadein_btn.setStyleSheet("")
        self._fadein_panel.setVisible(False)
        self._refresh_tab_bar_lock()

    def _cancel_fadein_if_active(self):
        if self._fadein_active and self._fadein_active_view is not None:
            view = self._fadein_active_view
            self._collapse_fadein_ui()
            view.cancel_effect()
        elif self._fadein_active:
            self._collapse_fadein_ui()

    def _on_fadein_clicked(self):
        if self._fadein_active:
            return
        self._cancel_trim_if_active()
        self._cancel_extract_if_active()
        self._cancel_timescale_if_active()
        self._cancel_vscale_if_active()
        self._cancel_reverse_if_active()
        self._cancel_fadeout_if_active()
        self._cancel_concat_if_active()
        view = self._current_entity_view()
        if view is None or not view.begin_fade_in():
            return
        self._fadein_active = True
        self._fadein_active_view = view
        self._fadein_btn.setStyleSheet(self._SELECTED_OPTION_STYLE)
        self._fadein_panel.setVisible(True)
        self._refresh_tab_bar_lock()

    def _on_fadein_apply_clicked(self):
        if not self._fadein_active or self._fadein_active_view is None:
            return
        view = self._fadein_active_view
        self._collapse_fadein_ui()
        view.apply_effect()

    def _on_fadein_cancel_clicked(self):
        if not self._fadein_active or self._fadein_active_view is None:
            return
        view = self._fadein_active_view
        self._collapse_fadein_ui()
        view.cancel_effect()

    # ------------------------------------------------------------------
    # Fade Out
    # ------------------------------------------------------------------

    def _collapse_fadeout_ui(self):
        self._fadeout_active = False
        self._fadeout_active_view = None
        self._fadeout_btn.setStyleSheet("")
        self._fadeout_panel.setVisible(False)
        self._refresh_tab_bar_lock()

    def _cancel_fadeout_if_active(self):
        if self._fadeout_active and self._fadeout_active_view is not None:
            view = self._fadeout_active_view
            self._collapse_fadeout_ui()
            view.cancel_effect()
        elif self._fadeout_active:
            self._collapse_fadeout_ui()

    def _on_fadeout_clicked(self):
        if self._fadeout_active:
            return
        self._cancel_trim_if_active()
        self._cancel_extract_if_active()
        self._cancel_timescale_if_active()
        self._cancel_vscale_if_active()
        self._cancel_reverse_if_active()
        self._cancel_fadein_if_active()
        self._cancel_concat_if_active()
        view = self._current_entity_view()
        if view is None or not view.begin_fade_out():
            return
        self._fadeout_active = True
        self._fadeout_active_view = view
        self._fadeout_btn.setStyleSheet(self._SELECTED_OPTION_STYLE)
        self._fadeout_panel.setVisible(True)
        self._refresh_tab_bar_lock()

    def _on_fadeout_apply_clicked(self):
        if not self._fadeout_active or self._fadeout_active_view is None:
            return
        view = self._fadeout_active_view
        self._collapse_fadeout_ui()
        view.apply_effect()

    def _on_fadeout_cancel_clicked(self):
        if not self._fadeout_active or self._fadeout_active_view is None:
            return
        view = self._fadeout_active_view
        self._collapse_fadeout_ui()
        view.cancel_effect()

    # ------------------------------------------------------------------
    # Concatenate
    # ------------------------------------------------------------------

    def _collapse_concat_ui(self):
        self._concat_active = False
        self._concat_active_view = None
        self._concat_btn.setStyleSheet("")
        self._concat_panel.setVisible(False)
        self._refresh_tab_bar_lock()

    def _cancel_concat_if_active(self):
        if self._concat_active and self._concat_active_view is not None:
            view = self._concat_active_view
            self._collapse_concat_ui()
            view.cancel_concatenate()
        elif self._concat_active:
            self._collapse_concat_ui()

    def _on_concatenate_clicked(self):
        if self._concat_active:
            return
        self._cancel_trim_if_active()
        self._cancel_extract_if_active()
        self._cancel_timescale_if_active()
        self._cancel_vscale_if_active()
        self._cancel_reverse_if_active()
        self._cancel_fadein_if_active()
        self._cancel_fadeout_if_active()

        view = self._current_entity_view()
        if view is None:
            return

        channel_count = view.entity.clip.num_channels
        dialog = ConcatenateDialog(self._workspace, channel_count, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        sources = dialog.selected_portions()
        if not sources or not view.begin_concatenate(sources):
            return

        self._concat_active = True
        self._concat_active_view = view
        self._concat_btn.setStyleSheet(self._SELECTED_OPTION_STYLE)
        self._concat_panel.setVisible(True)
        self._refresh_tab_bar_lock()

    def _on_concat_apply_clicked(self):
        if not self._concat_active or self._concat_active_view is None:
            return
        view = self._concat_active_view
        self._collapse_concat_ui()
        view.apply_concatenate()   # entity_concatenated → rebuild (clears selection)

    def _on_concat_cancel_clicked(self):
        if not self._concat_active or self._concat_active_view is None:
            return
        view = self._concat_active_view
        self._collapse_concat_ui()
        view.cancel_concatenate()  # restores graph + keeps selection

    # ------------------------------------------------------------------
    # Entity-view factory / tab lifecycle
    # ------------------------------------------------------------------

    def _make_entity_view(self, entity: Entity) -> EntityPlotView:
        entity_view = EntityPlotView(entity)
        entity_view.entity_modified.connect(self._on_entity_modified)
        entity_view.segment_selected.connect(self._on_segment_selected)
        entity_view.segment_deselected.connect(self._on_segment_deselected)
        entity_view.entity_trimmed.connect(lambda: self._rebuild_entity_tab(entity))
        entity_view.entity_extracted.connect(lambda: self._rebuild_entity_tab(entity))
        entity_view.entity_timescaled.connect(lambda: self._rebuild_entity_tab(entity))
        entity_view.entity_vscaled.connect(lambda: self._rebuild_entity_tab(entity))
        entity_view.entity_reversed.connect(lambda: self._rebuild_entity_tab(entity))
        entity_view.entity_faded_in.connect(lambda: self._rebuild_entity_tab(entity))
        entity_view.entity_faded_out.connect(lambda: self._rebuild_entity_tab(entity))
        entity_view.entity_concatenated.connect(lambda: self._rebuild_entity_tab(entity))
        entity_view.trim_cancelled.connect(self._on_trim_cancelled_externally)
        entity_view.set_edit_mode(self._mode == "edit")
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

    def _rebuild_entity_tab(self, entity: Entity):
        index = self._find_entity_tab(entity.id)
        if index == -1:
            return
        widget_index = index + 1
        old_widget   = self._content_stack.widget(widget_index)
        was_current  = (self._tab_bar.currentIndex() == index)

        if old_widget is self._trim_active_view:
            self._collapse_trim_ui()
        if old_widget is self._extract_active_view:
            self._collapse_extract_ui()
        if old_widget is self._timescale_active_view:
            self._collapse_timescale_ui()
        if old_widget is self._vscale_active_view:     # ← add (both methods; use `widget` instead of `old_widget` in _close_tab)
            self._collapse_vscale_ui()
        if old_widget is self._reverse_active_view:
            self._collapse_reverse_ui()
        if old_widget is self._fadein_active_view:
            self._collapse_fadein_ui()
        if old_widget is self._fadeout_active_view:
            self._collapse_fadeout_ui()
        if old_widget is self._concat_active_view:
            self._collapse_concat_ui()

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
        if widget is self._trim_active_view:
            self._collapse_trim_ui()
        if widget is self._extract_active_view:
            self._collapse_extract_ui()
        if widget is self._timescale_active_view:
            self._collapse_timescale_ui()
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
        self._cancel_trim_if_active()
        self._cancel_extract_if_active()
        self._cancel_timescale_if_active()
        self._cancel_vscale_if_active()
        self._cancel_reverse_if_active()
        self._cancel_fadein_if_active()
        self._cancel_fadeout_if_active()
        self._cancel_concat_if_active()
        self._force_stop_active_playback()      # ← add, before selection clearing
        self._clear_all_plot_selections()
    
        if index < 0:
            self._content_stack.setCurrentWidget(self._empty_label)
            return

        widget_index = index + 1   # index 0 is the empty label
        if widget_index < self._content_stack.count():
            self._content_stack.setCurrentIndex(widget_index)
        else:
            self._content_stack.setCurrentWidget(self._empty_label)