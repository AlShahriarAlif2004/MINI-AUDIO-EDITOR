import os
import shutil

from PySide6.QtWidgets import QFileDialog, QMessageBox

from backend.workspace_model import Workspace


def sync_workspace_name(workspace: Workspace) -> None:
    """Keep workspace.name in sync with its folder basename."""
    if workspace.path:
        workspace.name = os.path.basename(os.path.normpath(workspace.path))


def create_workspace(parent) -> Workspace | None:
    """Ask the user for a location/name and create a new workspace folder."""
    default_path = os.path.join(os.path.expanduser("~"), "untitled")
    path, _ = QFileDialog.getSaveFileName(
        parent,
        "Create Workspace",
        default_path,
    )
    if not path:
        return None

    workspace_dir = path

    if os.path.exists(workspace_dir):
        reply = QMessageBox.question(
            parent,
            "Replace Existing",
            f'"{workspace_dir}" already exists.\nDo you want to replace it?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return None

        if os.path.isdir(workspace_dir):
            shutil.rmtree(workspace_dir)
        else:
            os.remove(workspace_dir)

    os.makedirs(workspace_dir, exist_ok=True)

    workspace = Workspace(name=os.path.basename(workspace_dir), path=workspace_dir)
    sync_workspace_name(workspace)
    workspace.save()
    return workspace


def open_workspace(parent) -> Workspace | None:
    """Pick an existing workspace folder and load it."""
    path = QFileDialog.getExistingDirectory(parent, "Open Workspace")
    if not path:
        return None

    json_path = os.path.join(path, "workspace.json")
    if not os.path.isfile(json_path):
        QMessageBox.warning(
            parent,
            "Not a Workspace",
            f'No workspace.json found in "{path}".',
        )
        return None

    try:
        workspace = Workspace.load(path)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        QMessageBox.critical(
            parent,
            "Open Failed",
            f"Could not open workspace:\n{exc}",
        )
        return None

    sync_workspace_name(workspace)
    return workspace
