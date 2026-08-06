from PySide6.QtWidgets import QMainWindow, QStackedWidget
from frontend.views.home_page import HomePage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mini Audio Editor")
        self.resize(480, 360)  # small size for the home page

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home_page = HomePage()
        self.stack.addWidget(self.home_page)

        self.home_page.new_workspace_requested.connect(self._on_new_workspace)
        self.home_page.open_workspace_requested.connect(self._on_open_workspace)

    def _on_new_workspace(self):
        # placeholder — next step will add the "name your workspace" flow
        print("New Workspace clicked")

    def _on_open_workspace(self):
        # placeholder — next step will add the folder/file picker
        print("Open Workspace clicked")