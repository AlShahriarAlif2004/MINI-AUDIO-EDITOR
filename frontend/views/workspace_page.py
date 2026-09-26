from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QDoubleValidator, QIntValidator, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
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
from frontend.dialogs.concatenate_dialog import ConcatenateDialog
from frontend.dialogs.download_format_dialog import DownloadFormatDialog
from frontend.dialogs.noise_source_dialog import NoiseSourceDialog
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
# Echo parameters input: three fields for occurrence, delay, decay
# ---------------------------------------------------------------------------

class _EchoParametersInput(QWidget):
    """Three input fields for echo effect parameters with +/- spinners.
    Emits parameters_changed(occurrence, delay, decay) on every change."""

    parameters_changed = Signal(int, float, float)

    def __init__(self, default_delay: float, parent=None):
        super().__init__(parent)
        self._occurrence = 2
        self._delay = default_delay
        self._decay = 0.0
        self._build(default_delay)

    def _build(self, default_delay: float):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Occurrence field with +/- buttons
        occ_row = QHBoxLayout()
        occ_label = QLabel("Occurrence:")
        occ_label.setFixedWidth(80)
        
        btn_style = (
            "QPushButton {"
            "  min-width: 24px; max-width: 24px;"
            "  min-height: 24px; max-height: 24px;"
            "  font-size: 13px; font-weight: 700;"
            "  border: 1px solid palette(mid);"
            "  background: palette(button);"
            "}"
            "QPushButton:hover  { background: palette(light); }"
            "QPushButton:pressed{ background: palette(dark);  }"
        )
        
        self._occ_dec_btn = QPushButton("−")
        self._occ_dec_btn.setStyleSheet(btn_style)
        self._occ_dec_btn.setFixedWidth(24)
        self._occ_dec_btn.setFixedHeight(24)
        self._occ_dec_btn.clicked.connect(self._decrement_occurrence)
        
        self._occurrence_input = QLineEdit(str(self._occurrence))
        self._occurrence_input.setValidator(QIntValidator(1, 1000, self._occurrence_input))
        self._occurrence_input.setMaximumWidth(60)
        self._occurrence_input.setFixedHeight(24)
        self._occurrence_input.textChanged.connect(self._on_occurrence_changed)
        
        self._occ_inc_btn = QPushButton("+")
        self._occ_inc_btn.setStyleSheet(btn_style)
        self._occ_inc_btn.setFixedWidth(24)
        self._occ_inc_btn.setFixedHeight(24)
        self._occ_inc_btn.clicked.connect(self._increment_occurrence)
        
        occ_row.addWidget(occ_label)
        occ_row.addWidget(self._occ_dec_btn)
        occ_row.addWidget(self._occurrence_input)
        occ_row.addWidget(self._occ_inc_btn)
        occ_row.addStretch()
        layout.addLayout(occ_row)

        # Delay field
        delay_row = QHBoxLayout()
        delay_label = QLabel("Delay (s):")
        delay_label.setFixedWidth(80)
        self._delay_input = QLineEdit(f"{default_delay:.4f}")
        delay_validator = QDoubleValidator(0.001, 100.0, 4, self._delay_input)
        delay_validator.setNotation(QDoubleValidator.StandardNotation)
        self._delay_input.setValidator(delay_validator)
        self._delay_input.setMaximumWidth(100)
        self._delay_input.setFixedHeight(24)
        self._delay_input.textChanged.connect(self._on_delay_changed)
        delay_row.addWidget(delay_label)
        delay_row.addWidget(self._delay_input)
        delay_row.addStretch()
        layout.addLayout(delay_row)

        # Decay field with +/- buttons
        decay_row = QHBoxLayout()
        decay_label = QLabel("Decay:")
        decay_label.setFixedWidth(80)
        
        self._decay_dec_btn = QPushButton("−")
        self._decay_dec_btn.setStyleSheet(btn_style)
        self._decay_dec_btn.setFixedWidth(24)
        self._decay_dec_btn.setFixedHeight(24)
        self._decay_dec_btn.clicked.connect(self._decrement_decay)
        
        self._decay_input = QLineEdit(f"{self._decay:.1f}")
        decay_validator = QDoubleValidator(0.0, 1.0, 1, self._decay_input)
        decay_validator.setNotation(QDoubleValidator.StandardNotation)
        self._decay_input.setValidator(decay_validator)
        self._decay_input.setMaximumWidth(60)
        self._decay_input.setFixedHeight(24)
        self._decay_input.textChanged.connect(self._on_decay_changed)
        
        self._decay_inc_btn = QPushButton("+")
        self._decay_inc_btn.setStyleSheet(btn_style)
        self._decay_inc_btn.setFixedWidth(24)
        self._decay_inc_btn.setFixedHeight(24)
        self._decay_inc_btn.clicked.connect(self._increment_decay)
        
        decay_row.addWidget(decay_label)
        decay_row.addWidget(self._decay_dec_btn)
        decay_row.addWidget(self._decay_input)
        decay_row.addWidget(self._decay_inc_btn)
        decay_row.addStretch()
        layout.addLayout(decay_row)

    def _decrement_occurrence(self):
        val = max(1, self._occurrence - 1)
        self._occurrence = val
        self._occurrence_input.setText(str(val))

    def _increment_occurrence(self):
        val = min(1000, self._occurrence + 1)
        self._occurrence = val
        self._occurrence_input.setText(str(val))

    def _decrement_decay(self):
        val = max(0.0, round(self._decay - 0.1, 1))
        self._decay = val
        self._decay_input.setText(f"{val:.1f}")

    def _increment_decay(self):
        val = min(1.0, round(self._decay + 0.1, 1))
        self._decay = val
        self._decay_input.setText(f"{val:.1f}")

    def _on_occurrence_changed(self):
        try:
            val = int(self._occurrence_input.text())
            if val >= 1:
                self._occurrence = val
                self.parameters_changed.emit(self._occurrence, self._delay, self._decay)
        except ValueError:
            pass

    def _on_delay_changed(self):
        try:
            val = float(self._delay_input.text())
            if val > 0:
                self._delay = val
                self.parameters_changed.emit(self._occurrence, self._delay, self._decay)
        except ValueError:
            pass

    def _on_decay_changed(self):
        try:
            val = float(self._decay_input.text())
            if 0.0 <= val <= 1.0:
                self._decay = round(val, 1)
                self.parameters_changed.emit(self._occurrence, self._delay, self._decay)
        except ValueError:
            pass

    def set_parameters(self, occurrence: int, delay: float, decay: float):
        """Set all parameters at once (used for initial setup)."""
        self._occurrence = occurrence
        self._delay = delay
        self._decay = round(decay, 1)
        self._occurrence_input.blockSignals(True)
        self._delay_input.blockSignals(True)
        self._decay_input.blockSignals(True)
        
        self._occurrence_input.setText(str(occurrence))
        self._delay_input.setText(f"{delay:.4f}")
        self._decay_input.setText(f"{decay:.1f}")
        
        self._occurrence_input.blockSignals(False)
        self._delay_input.blockSignals(False)
        self._decay_input.blockSignals(False)
        
        # Emit the signal after setting
        self.parameters_changed.emit(self._occurrence, self._delay, self._decay)

    def get_parameters(self) -> tuple[int, float, float]:
        """Return current parameters."""
        return self._occurrence, self._delay, self._decay


# ---------------------------------------------------------------------------
# Per-operation runtime state (one instance per entry in _OP_SPECS)
# ---------------------------------------------------------------------------

class _OpEntry:
    """Bundles the widgets + live state for a single edit-mode operation
    (Trim, Extract, Time Scale, ...). Built from an _OP_SPECS dict."""

    __slots__ = (
        "spec", "button", "panel", "apply_btn", "cancel_btn",
        "factor_input", "echo_parameters_input", "echo_parameters_slot",
        "active", "active_view",
    )

    def __init__(self, spec, button, panel, apply_btn, cancel_btn, factor_input, echo_parameters_input=None):
        self.spec = spec
        self.button = button
        self.panel = panel
        self.apply_btn = apply_btn
        self.cancel_btn = cancel_btn
        self.factor_input = factor_input
        self.echo_parameters_input = echo_parameters_input
        self.echo_parameters_slot = None
        self.active = False
        self.active_view = None


# ---------------------------------------------------------------------------
# WorkspacePage
# ---------------------------------------------------------------------------

class _Spinner(QWidget):
    """Small rotating-arc busy indicator, advanced by a QTimer tick."""

    def __init__(self, parent=None, diameter=48):
        super().__init__(parent)
        self.setFixedSize(diameter, diameter)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    def start(self):
        self._angle = 0
        self._timer.start(30)

    def stop(self):
        self._timer.stop()

    def _advance(self):
        self._angle = (self._angle + 45) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("white"))
        pen.setWidth(4)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        rect = self.rect().adjusted(4, 4, -4, -4)
        painter.translate(rect.center())
        painter.rotate(self._angle)
        painter.translate(-rect.center())
        painter.drawArc(rect, 0, 270 * 16)
        painter.end()


class _BusyOverlay(QWidget):
    """Full-page bluish, semi-transparent veil shown while a slow,
    synchronous backend call (repeated noise-removal passes) runs on
    the GUI thread. It doesn't make the app responsive -- everything
    underneath is still blocked -- it just makes that visible instead
    of the window looking frozen, as long as the caller pumps the
    event loop (QApplication.processEvents()) while the work runs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self._spinner = _Spinner(self)
        layout.addWidget(self._spinner, alignment=Qt.AlignCenter)

        self._label = QLabel("REMOVING NOISE")
        self._label.setStyleSheet(
            "color: white; font-weight: 600; font-size: 14px; letter-spacing: 1px;"
        )
        self._label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._label)

    def show_message(self, text):
        self._label.setText(text)
        self.setGeometry(self.parentWidget().rect())
        self.raise_()
        self.setVisible(True)
        self._spinner.start()

    def hide_overlay(self):
        self._spinner.stop()
        self.setVisible(False)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, QColor(30, 41, 110, 110))
        gradient.setColorAt(1.0, QColor(90, 50, 150, 110))
        painter.fillRect(self.rect(), gradient)
        painter.end()


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
        # Noise Removal doesn't fit the generic single-shot Apply/Cancel
        # shape: it has a Filter/Cancel stage followed by an Apply/Cancel
        # stage. It still uses the shared begin/active/tab-lock machinery
        # (see _on_op_clicked), but its panel and button wiring are
        # custom — see _build_noise_panel/_on_noise_action_clicked/
        # _on_noise_cancel_clicked below.
        dict(key="noise", label="Noise Removal",
             begin="begin_noise_removal", apply="apply_noise_removal", cancel="cancel_noise_removal",
             rebuild_signal="entity_noise_removed", factor=False, needs_dialog=False,
             custom_panel=True),
        dict(key="echo", label="Add Echo",
             begin="begin_echo", apply="apply_echo", cancel="cancel_echo",
             rebuild_signal="entity_echo_added", factor=False, needs_dialog=False,
             preview="update_echo_preview"),
        # Detect Echo doesn't fit the generic single-shot Apply/Cancel shape
        # either: it starts as Analyze/Cancel, and once Analyze runs its
        # (read-only) blind-detection algorithm, Analyze is hidden and
        # Cancel becomes OK. See _build_detect_echo_panel/
        # _on_detect_echo_analyze_clicked/_on_detect_echo_cancel_clicked.
        dict(key="detectecho", label="Detect Echo",
             begin="begin_detect_echo", apply="apply_detect_echo", cancel="cancel_detect_echo",
             rebuild_signal="entity_echo_detected", factor=False, needs_dialog=False),
        # Show Frequency Domain doesn't fit the plain single Apply/Cancel
        # shape either -- like Detect Echo, it starts as Show/Cancel, and
        # once Show computes the X[k] spectra (the slow step, behind the
        # busy overlay) Show is hidden and Cancel becomes OK. Unlike
        # Detect Echo, OK never undoes anything: the Time Domain/Frequency
        # Domain toggle it reveals on the entity view stays put. See
        # _build_frequency_domain_panel/_on_freqdomain_show_clicked/
        # _on_freqdomain_close_clicked.
        dict(key="freqdomain", label="Show Frequency Domain",
             begin="begin_frequency_domain", apply="show_frequency_domain", cancel="close_frequency_domain",
             rebuild_signal="entity_frequency_domain_shown", factor=False, needs_dialog=False),
        dict(key="equalizer", label="Equalizer",
             begin="begin_equalizer", apply="apply_equalizer", cancel="cancel_equalizer",
             rebuild_signal="entity_equalizer_applied", factor=False, needs_dialog=False,
             custom_panel=True),
    ]

    new_workspace_requested = Signal()
    open_workspace_requested = Signal()
    exit_requested = Signal()

    def __init__(self, parent=None, editor_type="editor"):
        super().__init__(parent)
        self._workspace: Workspace | None = None
        self._mode = "file"
        self._editor_type = editor_type
        self._ops: dict[str, _OpEntry] = {}   # populated in _build_ui
        self._build_ui()
        self._busy_overlay = _BusyOverlay(self)
        self._update_sidebar_for_workspace()
        self._playback_group = PlaybackGroup.get_instance()
        self._playback_group.active_changed.connect(self._on_playback_active_changed)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._busy_overlay.isVisible():
            self._busy_overlay.setGeometry(self.rect())

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
            self,
            "Download Entity",
            default_name,
            file_filter,
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
        if self._editor_type == "editor":
            top_bar.addWidget(self._edit_btn)
        top_bar.addWidget(self._tool_btn)

        top_bar.addStretch()
        _MODE_TITLES = {
            "editor": "Editor Mode",
            "echo": "Echo Mode",
            "noise": "Noise Removal",
            "frequency": "Frequency Mode",
        }
        self._mode_title_label = QLabel(_MODE_TITLES.get(self._editor_type, ""))
        self._mode_title_label.setStyleSheet(
            "color: #FFFFFF; font-weight: 700; font-size: 14px; background: transparent;"
        )
        top_bar.addWidget(self._mode_title_label)
        top_bar.addStretch()

        top_bar_widget = QWidget()
        top_bar_widget.setLayout(top_bar)
        top_bar_widget.setFixedHeight(32)
        root_layout.addWidget(top_bar_widget)

        self._file_btn.setStyleSheet(self._SELECTED_BTN_STYLE)

        self._splitter = QSplitter(Qt.Horizontal)
        root_layout.setStretchFactor(self._splitter, 1)

        # ---- sidebar ----
        sidebar = QWidget()
        # Match the light color QTreeWidget normally paints itself with
        # (Fusion style's "base" color) directly on the container, so the
        # sidebar stays visually distinct from the editor pane even when
        # the tree is hidden and only the "No Workspace" label is shown.
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

        # Shown instead of the (then-empty) tree when no workspace is
        # loaded, so an unopened editor is visually distinct from a real
        # workspace that just happens to have zero folders/entities.
        self._no_workspace_label = QLabel("No Workspace")
        self._no_workspace_label.setAlignment(Qt.AlignCenter)
        # Keep the empty-state label readable against the light sidebar
        # background in both the workspace and voice-recognition shells.
        self._no_workspace_label.setStyleSheet(
            "color: white; font-style: italic; padding: 24px 0;"
        )
        sidebar_layout.addWidget(self._no_workspace_label)

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

        if self._editor_type == "echo":
            _TOOL_SECTIONS = [
                ("Echo Tools", ["Add Echo", "Detect Echo"]),
            ]
        elif self._editor_type == "noise":
            _TOOL_SECTIONS = [
                ("Filtering Tools", ["Noise Removal"]),
            ]
        elif self._editor_type == "frequency":
            _TOOL_SECTIONS = [
                # Both are now fully implemented.
                ("Frequency Tools", ["Show Frequency Domain", "Equalizer"]),
            ]
        else:
            _TOOL_SECTIONS = [
                ("General Tools", ["Delete Divisor"]),
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
                btn.setEnabled(False)
                tool_options_layout.addWidget(btn)
                self._tool_buttons[label] = btn

                if spec is None:
                    continue

                if spec.get("key") == "echo":
                    # Echo panel with three parameter inputs
                    # Default delay will be set dynamically in begin_echo
                    panel, apply_btn, cancel_btn, echo_input = self._build_echo_panel(1.0)
                elif spec.get("key") == "detectecho":
                    panel, apply_btn, cancel_btn, echo_input = self._build_detect_echo_panel()
                elif spec.get("key") == "freqdomain":
                    panel, apply_btn, cancel_btn, echo_input = self._build_frequency_domain_panel()
                elif spec.get("key") == "equalizer":
                    panel, apply_btn, cancel_btn, echo_input = self._build_equalizer_panel()
                elif spec.get("custom_panel"):
                    panel, apply_btn, cancel_btn, echo_input = self._build_noise_panel()
                    echo_input = None
                else:
                    panel, apply_btn, cancel_btn, factor_input = self._build_op_panel(spec)
                    echo_input = None
                tool_options_layout.addWidget(panel)

                entry = _OpEntry(spec, btn, panel, apply_btn, cancel_btn, 
                                factor_input if spec.get("key") != "echo" else None,
                                echo_input if spec.get("key") == "echo" else None)
                self._ops[spec["key"]] = entry

                key = spec["key"]
                btn.clicked.connect(lambda checked=False, k=key: self._on_op_clicked(k))
                if spec.get("key") == "detectecho":
                    apply_btn.clicked.connect(lambda checked=False, k=key: self._on_detect_echo_analyze_clicked(k))
                    cancel_btn.clicked.connect(lambda checked=False, k=key: self._on_detect_echo_cancel_clicked(k))
                elif spec.get("key") == "freqdomain":
                    apply_btn.clicked.connect(lambda checked=False, k=key: self._on_freqdomain_show_clicked(k))
                    cancel_btn.clicked.connect(lambda checked=False, k=key: self._on_freqdomain_close_clicked(k))
                elif spec.get("key") == "equalizer":
                    apply_btn.clicked.connect(lambda checked=False, k=key: self._on_equalizer_find_clicked(k))
                    cancel_btn.clicked.connect(lambda checked=False, k=key: self._on_equalizer_cancel_clicked(k))
                elif spec.get("custom_panel"):
                    apply_btn.clicked.connect(lambda checked=False, k=key: self._on_noise_action_clicked(k))
                    cancel_btn.clicked.connect(lambda checked=False, k=key: self._on_noise_cancel_clicked(k))
                else:
                    apply_btn.clicked.connect(lambda checked=False, k=key: self._on_op_apply_clicked(k))
                    cancel_btn.clicked.connect(lambda checked=False, k=key: self._on_op_cancel_clicked(k))

        tool_options_layout.addStretch()
        self._tool_options_panel.setVisible(False)   # shown when tool mode is active
        sidebar_layout.addWidget(self._tool_options_panel)

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
        self._refresh_tool_buttons()

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

    def _build_noise_panel(self):
        """Noise Removal's panel: starts as Filter/Cancel; after Filter is
        clicked the left button's label is swapped to Apply (see
        _on_noise_action_clicked) and Cancel's meaning changes from
        "close the panel" to "go back to range selection" (see
        _on_noise_cancel_clicked) — both keyed off the button's own text,
        so no extra state needs to live on _OpEntry."""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(16, 0, 0, 0)
        layout.setSpacing(6)
        action_btn = QPushButton("Filter")
        cancel_btn = QPushButton("Cancel")
        layout.addWidget(action_btn)
        layout.addWidget(cancel_btn)
        panel.setVisible(False)
        return panel, action_btn, cancel_btn, None

    def _build_echo_panel(self, default_delay: float):
        """Echo panel: three input fields (occurrence, delay, decay) with
        Apply/Cancel buttons. Parameters are shown vertically and update
        preview in real time."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 0, 0, 0)
        layout.setSpacing(4)

        # Echo parameters inputs
        echo_input = _EchoParametersInput(default_delay)
        layout.addWidget(echo_input)

        # Apply/Cancel buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        apply_btn = QPushButton("Apply")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        panel.setVisible(False)
        return panel, apply_btn, cancel_btn, echo_input

    def _build_detect_echo_panel(self):
        """Detect Echo's panel: starts as Analyze/Cancel. Once Analyze
        has run, the Analyze button is hidden and Cancel is relabeled OK
        (see _on_detect_echo_analyze_clicked) -- either button always
        just closes the panel and restores the previous view, since
        detection never modifies the entity's actual clip."""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(16, 0, 0, 0)
        layout.setSpacing(6)
        analyze_btn = QPushButton("Analyze")
        cancel_btn = QPushButton("Cancel")
        layout.addWidget(analyze_btn)
        layout.addWidget(cancel_btn)
        panel.setVisible(False)
        return panel, analyze_btn, cancel_btn, None

    def _build_frequency_domain_panel(self):
        """Show Frequency Domain's panel: starts as Show/Cancel. Once
        Show has computed the X[k] spectra, the Show button is hidden
        and Cancel is relabeled OK (see _on_freqdomain_show_clicked) --
        either button just closes the panel; OK additionally leaves the
        Time Domain/Frequency Domain toggle it revealed in place."""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(16, 0, 0, 0)
        layout.setSpacing(6)
        show_btn = QPushButton("Show")
        cancel_btn = QPushButton("Cancel")
        layout.addWidget(show_btn)
        layout.addWidget(cancel_btn)
        panel.setVisible(False)
        return panel, show_btn, cancel_btn, None

    def _build_equalizer_panel(self):
        """Equalizer's panel: two phases controlled by the same two buttons.

        Phase 1 (Find):
            Frequency: [input]
            k = — (computed display)
            [Find]  [Cancel]

        Phase 2 (Coefficient editor, shown after Find):
            A scroll area with one Real/Imag pair per channel.
            [Apply]  [Cancel]

        The panel widget holds a QVBoxLayout; the coefficient scroll area is
        added/removed programmatically by the handler methods so it lives
        entirely in Python-level state (_equalizer_coeff_scroll, etc.).
        """
        # Persistent references so handlers can access them
        self._equalizer_coeff_scroll = None   # QScrollArea; created on Find
        self._equalizer_coeff_inputs = []     # list of (real_edit, imag_edit) per channel

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 4, 0, 4)
        layout.setSpacing(6)

        # ---- Frequency row ------------------------------------------------
        freq_row = QHBoxLayout()
        freq_row.setSpacing(6)
        freq_label = QLabel("Frequency (Hz):")
        freq_label.setFixedWidth(110)
        self._equalizer_freq_edit = QLineEdit("0")
        self._equalizer_freq_edit.setFixedWidth(90)
        self._equalizer_freq_edit.setFixedHeight(26)
        self._equalizer_freq_edit.setAlignment(Qt.AlignRight)
        freq_validator = QDoubleValidator(0.0, 9_999_999.0, 2)
        freq_validator.setNotation(QDoubleValidator.StandardNotation)
        self._equalizer_freq_edit.setValidator(freq_validator)
        freq_row.addWidget(freq_label)
        freq_row.addWidget(self._equalizer_freq_edit)
        freq_row.addStretch()
        layout.addLayout(freq_row)

        # ---- k display row ------------------------------------------------
        k_row = QHBoxLayout()
        k_row.setSpacing(6)
        self._equalizer_k_label = QLabel("k = \u2014")
        self._equalizer_k_label.setStyleSheet("font-style: italic; color: palette(dark);")
        k_row.addWidget(self._equalizer_k_label)
        k_row.addStretch()
        layout.addLayout(k_row)

        # Connect freq edit to live k update
        self._equalizer_freq_edit.textEdited.connect(self._on_equalizer_freq_changed)

        # ---- Find / Cancel buttons ----------------------------------------
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        find_btn  = QPushButton("Find")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(find_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        # The "apply" btn slot in _OpEntry is repurposed as "Find" here.
        panel.setVisible(False)
        return panel, find_btn, cancel_btn, None

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

        self._update_sidebar_for_workspace()

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
        self._update_sidebar_for_workspace()
        self._splitter.setSizes([240, 960])

    def clear_workspace(self):
        """Return to the empty, no-workspace-loaded state — distinct from
        a real workspace with zero folders/entities, since the sidebar
        shows the 'No Workspace' placeholder instead of an empty tree."""
        self._workspace = None
        self._clear_tabs()
        self._set_mode("file")
        self._tree.clear()
        self._content_stack.setCurrentWidget(self._empty_label)
        self._update_sidebar_for_workspace()
        self._splitter.setSizes([240, 960])

    def workspace(self) -> Workspace | None:
        return self._workspace

    def _update_sidebar_for_workspace(self):
        """
        Tree and the 'No Workspace' placeholder are mutually exclusive
        and only ever shown in File mode. New Folder/New Entity stay
        visible in File mode regardless — they're just disabled until a
        workspace is actually loaded, rather than disappearing. Edit/Tool
        modes are themselves disabled with nothing loaded, since both
        operate on entities.
        """
        has_workspace = self._workspace is not None
        in_file_mode = self._mode == "file"

        self._edit_btn.setEnabled(has_workspace)
        self._tool_btn.setEnabled(has_workspace)

        self._tree.setVisible(in_file_mode and has_workspace)
        self._no_workspace_label.setVisible(in_file_mode and not has_workspace)

        self._new_folder_btn.setVisible(in_file_mode)
        self._new_entity_btn.setVisible(in_file_mode)
        self._new_folder_btn.setEnabled(has_workspace)
        self._new_entity_btn.setEnabled(has_workspace)

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

    def _refresh_tool_buttons(self):
        has_selected_tab = self._tab_bar.currentIndex() >= 0
        implemented_labels = {
            spec["label"] for spec in self._TOOL_OP_SPECS
        }
        for label, btn in self._tool_buttons.items():
            btn.setEnabled(has_selected_tab and label in implemented_labels)

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
        if entry.echo_parameters_input is not None and entry.echo_parameters_slot is not None:
            try:
                entry.echo_parameters_input.parameters_changed.disconnect(entry.echo_parameters_slot)
            except TypeError:
                pass
            entry.echo_parameters_slot = None
        if entry.spec.get("key") == "detectecho":
            entry.apply_btn.setVisible(True)
            entry.apply_btn.setText("Analyze")
            entry.cancel_btn.setText("Cancel")
        elif entry.spec.get("key") == "freqdomain":
            entry.apply_btn.setVisible(True)
            entry.apply_btn.setText("Show")
            entry.cancel_btn.setText("Cancel")
        elif entry.spec.get("key") == "equalizer":
            entry.apply_btn.setVisible(True)
            entry.apply_btn.setText("Find")
            entry.cancel_btn.setText("Cancel")
            # Also tear down the coefficient scroll area if present
            if hasattr(self, "_equalizer_coeff_scroll") and self._equalizer_coeff_scroll is not None:
                panel_layout = entry.panel.layout()
                if panel_layout is not None:
                    panel_layout.removeWidget(self._equalizer_coeff_scroll)
                self._equalizer_coeff_scroll.hide()
                self._equalizer_coeff_scroll.deleteLater()
                self._equalizer_coeff_scroll = None
                self._equalizer_coeff_inputs = []
            # Reset freq label & edit
            if hasattr(self, "_equalizer_k_label") and self._equalizer_k_label is not None:
                self._equalizer_k_label.setText("k = \u2014")
            if hasattr(self, "_equalizer_freq_edit") and self._equalizer_freq_edit is not None:
                self._equalizer_freq_edit.setText("0")
        elif entry.spec.get("custom_panel"):
            entry.apply_btn.setText("Filter")
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
            if key == "concat":
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
                return
        elif key == "noise":
            # Noise Removal: ask where the noise profile comes from first.
            # Picking an entity channel skips the Filter stage entirely —
            # the panel opens straight into Apply/Cancel.
            source_dialog = NoiseSourceDialog(self._workspace, view.entity, self)
            if source_dialog.exec() != QDialog.DialogCode.Accepted:
                return
            if source_dialog.selected_mode() == "entity":
                source_entity, source_channel = source_dialog.selected_entity_channel()
                if source_entity is None:
                    return
                self._busy_overlay.show_message("REMOVING NOISE")
                QApplication.processEvents()
                began = view.begin_noise_removal_from_entity(source_entity, source_channel)
                self._busy_overlay.hide_overlay()
                if began:
                    entry.apply_btn.setText("Apply")
            else:
                began = view.begin_noise_removal()
                if began:
                    entry.apply_btn.setText("Filter")
        elif key == "echo":
            # Echo: get default delay from entity, then start echo mode
            clip = view.entity.clip
            start_time = clip.get_time(clip.start_index)
            end_time = clip.get_time(clip.end_index())
            default_delay = end_time - start_time
            
            # Begin echo with default parameters
            began = getattr(view, entry.spec["begin"])()
            if began:
                # Connect parameter changes to preview
                entry.echo_parameters_slot = lambda o, d, dc, k=key: self._on_echo_parameters_changed(k, o, d, dc)
                entry.echo_parameters_input.parameters_changed.connect(entry.echo_parameters_slot)
                
                # Set parameters and trigger initial preview
                entry.echo_parameters_input.set_parameters(2, default_delay, 0.0)
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
        if key == "echo":
            occurrence, delay, decay = entry.echo_parameters_input.get_parameters()
            view.update_echo_preview(occurrence, delay, decay)

    def _on_op_factor_changed(self, key: str, factor: float):
        """Live-preview: update the plot instantly as the factor changes."""
        entry = self._ops[key]
        if entry.active and entry.active_view is not None:
            getattr(entry.active_view, entry.spec["preview"])(factor)

    def _on_echo_parameters_changed(self, key: str, occurrence: int, delay: float, decay: float):
        """Live-preview for echo: update the plot as parameters change."""
        entry = self._ops[key]
        if entry.active and entry.active_view is not None:
            entry.active_view.update_echo_preview(occurrence, delay, decay)

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
    # Noise Removal's two-stage panel (Filter/Cancel -> Apply/Cancel)
    # ------------------------------------------------------------------

    def _on_noise_action_clicked(self, key: str):
        entry = self._ops[key]
        if not entry.active or entry.active_view is None:
            return
        view = entry.active_view

        if entry.apply_btn.text() == "Filter":
            self._busy_overlay.show_message("REMOVING NOISE")
            QApplication.processEvents()
            view.filter_noise()
            self._busy_overlay.hide_overlay()
            entry.apply_btn.setText("Apply")
        else:
            self._collapse_op_ui(key)
            view.apply_noise_removal()          # → rebuild (clears selection)

    def _on_noise_cancel_clicked(self, key: str):
        entry = self._ops[key]
        if not entry.active or entry.active_view is None:
            return
        view = entry.active_view
        self._collapse_op_ui(key)
        view.cancel_noise_removal()

    # ------------------------------------------------------------------
    # Detect Echo's two-stage panel (Analyze/Cancel -> OK)
    # ------------------------------------------------------------------

    def _on_detect_echo_analyze_clicked(self, key: str):
        entry = self._ops[key]
        if not entry.active or entry.active_view is None:
            return
        view = entry.active_view
        self._busy_overlay.show_message("ANALYZING ECHO")
        QApplication.processEvents()
        view.analyze_echo()
        self._busy_overlay.hide_overlay()
        entry.apply_btn.setVisible(False)
        entry.cancel_btn.setText("OK")

    def _on_detect_echo_cancel_clicked(self, key: str):
        entry = self._ops[key]
        if not entry.active or entry.active_view is None:
            return
        view = entry.active_view
        self._collapse_op_ui(key)
        view.cancel_detect_echo()

    # ------------------------------------------------------------------
    # Show Frequency Domain's two-stage panel (Show/Cancel -> OK)
    # ------------------------------------------------------------------

    def _on_freqdomain_show_clicked(self, key: str):
        entry = self._ops[key]
        if not entry.active or entry.active_view is None:
            return
        view = entry.active_view
        self._busy_overlay.show_message("COMPUTING FREQUENCY DOMAIN")
        QApplication.processEvents()
        view.show_frequency_domain()
        self._busy_overlay.hide_overlay()
        entry.apply_btn.setVisible(False)
        entry.cancel_btn.setText("OK")

    def _on_freqdomain_close_clicked(self, key: str):
        entry = self._ops[key]
        if not entry.active or entry.active_view is None:
            return
        view = entry.active_view
        self._collapse_op_ui(key)
        view.close_frequency_domain()

    # ------------------------------------------------------------------
    # Equalizer: Find → coefficient editor → Apply/Cancel
    # ------------------------------------------------------------------

    def _equalizer_compute_k(self) -> tuple[int, float]:
        """Compute k from the current frequency input and the active entity.
        Returns (k, f_quantized) where f_quantized = k * fs / N."""
        view = self._current_entity_view()
        if view is None:
            return 0, 0.0
        clip = view.entity.clip
        fs = clip.sample_rate
        N = len(clip.channels[0].samples)
        try:
            f = float(self._equalizer_freq_edit.text())
        except ValueError:
            f = 0.0
        # For a real signal, spectrum is only meaningful up to Nyquist (fs/2)
        f = max(0.0, min(f, float(fs) / 2.0))
        k = round(f * N / fs)
        k = max(0, min(k, N - 1))
        f_quantized = k * fs / N
        return k, f_quantized

    def _on_equalizer_freq_changed(self, text: str):
        """Live-update the k display when the frequency text changes."""
        entry = self._ops.get("equalizer")
        if entry is None or not entry.active:
            return
        k, f_q = self._equalizer_compute_k()
        self._equalizer_k_label.setText(
            f"k = {k}   (f\u2090 = {f_q:.2f} Hz)"
        )

    def _on_equalizer_find_clicked(self, key: str):
        """'Find' button: compute k, call view.find_equalizer_coefficients(k),
        then transition the panel to Phase 2 (coefficient editor)."""
        entry = self._ops.get(key)
        if entry is None or not entry.active or entry.active_view is None:
            return
        view = entry.active_view

        k, f_q = self._equalizer_compute_k()
        self._equalizer_k_label.setText(f"k = {k}   (f\u2090 = {f_q:.2f} Hz)")

        coefficients = view.find_equalizer_coefficients(k)
        if not coefficients:
            return

        # Tear down any pre-existing coefficient scroll area
        panel_layout = entry.panel.layout()
        if self._equalizer_coeff_scroll is not None:
            panel_layout.removeWidget(self._equalizer_coeff_scroll)
            self._equalizer_coeff_scroll.hide()
            self._equalizer_coeff_scroll.deleteLater()
            self._equalizer_coeff_scroll = None
        self._equalizer_coeff_inputs = []

        # Build the coefficient scroll area
        coeff_container = QWidget()
        coeff_layout = QVBoxLayout(coeff_container)
        coeff_layout.setContentsMargins(0, 0, 0, 0)
        coeff_layout.setSpacing(6)

        for ch_idx, coeff in enumerate(coefficients):
            ch_label = QLabel(f"Ch {ch_idx + 1}  X[{k}]:")
            ch_label.setStyleSheet("font-weight: 600; margin-top: 4px;")
            coeff_layout.addWidget(ch_label)

            real_row = QHBoxLayout()
            real_label = QLabel("  Real:")
            real_label.setFixedWidth(44)
            real_edit = QLineEdit(f"{coeff.real:.6g}")
            real_edit.setFixedHeight(24)
            real_row.addWidget(real_label)
            real_row.addWidget(real_edit)
            coeff_layout.addLayout(real_row)

            imag_row = QHBoxLayout()
            imag_label = QLabel("  Imag:")
            imag_label.setFixedWidth(44)
            imag_edit = QLineEdit(f"{coeff.imag:.6g}")
            imag_edit.setFixedHeight(24)
            imag_row.addWidget(imag_label)
            imag_row.addWidget(imag_edit)
            coeff_layout.addLayout(imag_row)

            self._equalizer_coeff_inputs.append((real_edit, imag_edit))

            # Connect to live preview
            real_edit.textEdited.connect(
                lambda _text, k=k, key=key: self._on_equalizer_coeff_changed(k, key)
            )
            imag_edit.textEdited.connect(
                lambda _text, k=k, key=key: self._on_equalizer_coeff_changed(k, key)
            )

        # Apply / Cancel buttons (second phase)
        btn_row = QHBoxLayout()
        apply_btn2 = QPushButton("Apply")
        cancel_btn2 = QPushButton("Cancel")
        apply_btn2.clicked.connect(lambda: self._on_equalizer_apply_clicked(key))
        cancel_btn2.clicked.connect(lambda: self._on_equalizer_cancel_clicked(key))
        btn_row.addWidget(apply_btn2)
        btn_row.addWidget(cancel_btn2)
        coeff_layout.addLayout(btn_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(coeff_container)
        scroll.setMaximumHeight(280)
        self._equalizer_coeff_scroll = scroll

        # Insert the scroll above the Find/Cancel button row.
        # The layout is: freq_row, k_row, btn_row(Find/Cancel).
        # We want the scroll between k_row (index 1) and btn_row (index 2).
        panel_layout.insertWidget(2, scroll)

        # Hide the original Find/Cancel buttons while in phase 2
        entry.apply_btn.setVisible(False)
        entry.cancel_btn.setVisible(False)

    def _on_equalizer_coeff_changed(self, k: int, key: str):
        """Parse all coefficient inputs and push a live preview update."""
        entry = self._ops.get(key)
        if entry is None or not entry.active or entry.active_view is None:
            return
        view = entry.active_view

        coefficients = []
        for real_edit, imag_edit in self._equalizer_coeff_inputs:
            try:
                r = float(real_edit.text())
            except ValueError:
                r = 0.0
            try:
                i = float(imag_edit.text())
            except ValueError:
                i = 0.0
            coefficients.append(complex(r, i))

        view.update_equalizer_preview(k, coefficients)

    def _on_equalizer_apply_clicked(self, key: str):
        """Commit the equalizer changes."""
        entry = self._ops.get(key)
        if entry is None or not entry.active or entry.active_view is None:
            return
        view = entry.active_view
        self._collapse_op_ui(key)
        view.apply_equalizer()

    def _on_equalizer_cancel_clicked(self, key: str):
        """Revert equalizer and close the panel."""
        entry = self._ops.get(key)
        if entry is None:
            return
        # Restore Find/Cancel visibility before collapse
        if entry.apply_btn is not None:
            entry.apply_btn.setVisible(True)
        if entry.cancel_btn is not None:
            entry.cancel_btn.setVisible(True)
        if entry.active and entry.active_view is not None:
            view = entry.active_view
            self._collapse_op_ui(key)
            view.cancel_equalizer()
        else:
            self._collapse_op_ui(key)

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

    def _collapse_ops_for_widget(self, widget, skip_keys: set = None):
        """Collapse any operation panel that's mid-flight on `widget`,
        for every one of the 8 ops (not just Trim/Extract/Time Scale —
        see the discussion on the earlier stale-reference bug)."""
        if skip_keys is None:
            skip_keys = set()
        for key, entry in self._ops.items():
            if widget is entry.active_view:
                if key not in skip_keys:
                    self._collapse_op_ui(key)

    def _rebuild_entity_tab(self, entity: Entity):
        index = self._find_entity_tab(entity.id)
        if index == -1:
            return
        widget_index = index + 1
        old_widget   = self._content_stack.widget(widget_index)
        was_current  = (self._tab_bar.currentIndex() == index)

        # Check if echo is active on old widget (only editors built with
        # editor_type="echo" register an "echo" op at all -- e.g. the
        # noise/filtering editor never does, so this must not assume the
        # key exists).
        echo_active = False
        echo_params_input = None
        echo_original_clip = None
        echo_parameters = None
        echo_entry = self._ops.get("echo")
        if echo_entry is not None and old_widget is echo_entry.active_view:
            echo_active = True
            echo_params_input = echo_entry.echo_parameters_input
            echo_original_clip = old_widget._echo_original_clip.copy()
            echo_parameters = old_widget._echo_parameters

        equalizer_active = False
        eq_orig_clip = None
        eq_k = None
        eq_coeffs = None
        equalizer_entry = self._ops.get("equalizer")
        if equalizer_entry is not None and old_widget is equalizer_entry.active_view:
            equalizer_active = True
            eq_orig_clip = old_widget._equalizer_original_clip
            eq_k = old_widget._equalizer_k
            eq_coeffs = old_widget._equalizer_coefficients

        self._collapse_ops_for_widget(old_widget, skip_keys={"echo", "equalizer"})

        self._content_stack.removeWidget(old_widget)
        old_widget.deleteLater()

        entity_view = self._make_entity_view(entity)
        
        # If echo was active, restore the connection to the new view
        if echo_active and echo_params_input is not None:
            entry = echo_entry
            entry.active_view = entity_view
            entry.active = True
            entity_view._echo_original_clip = echo_original_clip
            entity_view._echo_parameters = echo_parameters
            entity_view._lock_selection(True)
            if entry.echo_parameters_slot is not None:
                try:
                    echo_params_input.parameters_changed.disconnect(entry.echo_parameters_slot)
                except TypeError:
                    pass
            entry.echo_parameters_slot = lambda o, d, dc, k="echo": self._on_echo_parameters_changed(k, o, d, dc)
            echo_params_input.parameters_changed.connect(entry.echo_parameters_slot)
            
        if equalizer_active:
            entry = equalizer_entry
            entry.active_view = entity_view
            entry.active = True
            entity_view._equalizer_active = True
            entity_view._equalizer_original_clip = eq_orig_clip
            entity_view._equalizer_k = eq_k
            entity_view._equalizer_coefficients = eq_coeffs
            entity_view._lock_selection(True)
        
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
        self._refresh_tool_buttons()

        if index < 0:
            self._content_stack.setCurrentWidget(self._empty_label)
            return

        widget_index = index + 1   # index 0 is the empty label
        if widget_index < self._content_stack.count():
            self._content_stack.setCurrentIndex(widget_index)
        else:
            self._content_stack.setCurrentWidget(self._empty_label)