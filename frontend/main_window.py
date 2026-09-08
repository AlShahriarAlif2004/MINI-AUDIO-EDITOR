from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QMessageBox, QStackedWidget

from backend.workspace_model import Workspace
from frontend.views.home_page import HomePage
from frontend.views.workspace_page import WorkspacePage
from frontend.views.voice_recognition_page import VoiceRecognitionPage
from frontend.workspace_io import create_workspace, open_workspace


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mini Audio Editor")

        self._current_workspace: Workspace | None = None
        self._active_editor_page = None   # whichever of the two pages below is on screen

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home_page = HomePage()
        self.workspace_page = WorkspacePage()
        self.voice_recognition_page = VoiceRecognitionPage()
        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.workspace_page)
        self.stack.addWidget(self.voice_recognition_page)

        self.home_page.editor_mode_requested.connect(self._on_editor_mode)
        self.home_page.voice_recognition_mode_requested.connect(self._on_voice_recognition_mode)

        # Both editor pages share the same File > New/Open/Exit handlers —
        # _active_editor_page tracks which one is currently on screen.
        self.workspace_page.new_workspace_requested.connect(self._on_new_workspace)
        self.workspace_page.open_workspace_requested.connect(self._on_open_workspace)
        self.workspace_page.exit_requested.connect(self._on_exit_workspace)

        self.voice_recognition_page.new_workspace_requested.connect(self._on_new_workspace)
        self.voice_recognition_page.open_workspace_requested.connect(self._on_open_workspace)
        self.voice_recognition_page.exit_requested.connect(self._on_exit_workspace)

    def _on_editor_mode(self):
        """Enter the (full Edit/Tool) workspace editor with nothing
        loaded yet — the sidebar shows its 'No Workspace' placeholder
        until File > New or File > Open is used."""
        self._enter_empty_editor(self.workspace_page)

    def _on_voice_recognition_mode(self):
        """Enter Voice Recognition mode with nothing loaded yet — same
        empty-state placeholder, File-menu-only shell."""
        self._enter_empty_editor(self.voice_recognition_page)

    def _enter_empty_editor(self, page):
        if not self._confirm_leave_workspace():
            return

        self._current_workspace = None
        self._active_editor_page = page
        page.clear_workspace()
        self.stack.setCurrentWidget(page)
        self.setWindowTitle("Mini Audio Editor")

    def _on_new_workspace(self):
        if not self._confirm_leave_workspace():
            return

        workspace = create_workspace(self)
        if workspace:
            self._enter_workspace(workspace)

    def _on_open_workspace(self):
        if not self._confirm_leave_workspace():
            return

        workspace = open_workspace(self)
        if workspace:
            self._enter_workspace(workspace)

    def _on_exit_workspace(self):
        if not self._confirm_leave_workspace():
            return
        self._leave_workspace()

    def _enter_workspace(self, workspace: Workspace):
        self._current_workspace = workspace
        self._active_editor_page.set_workspace(workspace)
        self.stack.setCurrentWidget(self._active_editor_page)
        self.setWindowTitle(f"Mini Audio Editor — {workspace.name}")

    def _leave_workspace(self):
        self._current_workspace = None
        self._active_editor_page = None
        self.stack.setCurrentWidget(self.home_page)
        self.setWindowTitle("Mini Audio Editor")

    def _confirm_leave_workspace(self) -> bool:
        """Prompt to save if the current workspace has unsaved changes."""
        if self._current_workspace is None or not self._current_workspace.dirty:
            return True

        reply = QMessageBox.question(
            self,
            "Save Workspace",
            "Save changes before leaving?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )

        if reply == QMessageBox.Cancel:
            return False
        if reply == QMessageBox.Save:
            try:
                self._current_workspace.save()
            except OSError as exc:
                QMessageBox.critical(
                    self,
                    "Save Failed",
                    f"Could not save workspace:\n{exc}",
                )
                return False
        return True

    def closeEvent(self, event: QCloseEvent):
        if not self._confirm_leave_workspace():
            event.ignore()
            return
        super().closeEvent(event)
