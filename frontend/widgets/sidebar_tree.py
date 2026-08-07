from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QStyleFactory

from backend.workspace_model import Entity, Folder, WorkspaceItem


class SidebarTree(QTreeWidget):
    """Tree view of the workspace folder/entity hierarchy."""

    entity_clicked = Signal(object)  # Entity
    folder_clicked = Signal(object)  # Folder

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyle(QStyleFactory.create("Fusion"))
        self.setHeaderHidden(True)
        self.setIndentation(16)
        self.setExpandsOnDoubleClick(False)

    def populate(self, root: Folder):
        self.clear()
        for child in root.children:
            self.addTopLevelItem(self._make_item(child))
        self.expandAll()

    def _make_item(self, item: WorkspaceItem) -> QTreeWidgetItem:
        tree_item = QTreeWidgetItem([item.name])
        tree_item.setData(0, Qt.UserRole, item)

        if isinstance(item, Folder):
            if item.children:
                for child in item.children:
                    tree_item.addChild(self._make_item(child))
            else:
                # Hidden placeholder so the expand arrow always shows,
                # even for an empty folder. It never renders since it's
                # hidden, but its presence makes childCount() > 0, which
                # is what actually drives the branch indicator reliably
                # (unlike ChildIndicatorPolicy.ShowIndicator, which some
                # styles — Fusion included — ignore for zero-child items).
                placeholder = QTreeWidgetItem([""])
                placeholder.setHidden(True)
                tree_item.addChild(placeholder)

        return tree_item

    def selected_folder(self) -> Folder | None:
        items = self.selectedItems()
        if not items:
            return None

        selected = items[0].data(0, Qt.UserRole)
        if isinstance(selected, Folder):
            return selected
        if isinstance(selected, Entity) and selected.parent is not None:
            return selected.parent
        return None

    def mouseReleaseEvent(self, event: QMouseEvent):
        item = self.itemAt(event.pos())
        if item is not None and event.button() == Qt.LeftButton:
            data = item.data(0, Qt.UserRole)
            if isinstance(data, Entity):
                self.setCurrentItem(item)
                self.entity_clicked.emit(data)
                event.accept()
                return
            if isinstance(data, Folder):
                self.setCurrentItem(item)
                item.setExpanded(not item.isExpanded())
                self.folder_clicked.emit(data)
                event.accept()
                return

        if item is None and event.button() == Qt.LeftButton:
            # Clicked empty space — deselect so new items land in root,
            # matching VSCode behavior.
            self.clearSelection()
            self.setCurrentItem(None)

        super().mouseReleaseEvent(event)
