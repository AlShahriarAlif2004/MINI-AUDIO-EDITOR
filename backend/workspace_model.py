import os
import json
import uuid

from backend.audio_clip import AudioClip
from backend.audio_io import WavIO


class WorkspaceItem:
    """Base class for anything that can sit inside the workspace tree."""

    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent

    def to_dict(self):
        raise NotImplementedError


class Folder(WorkspaceItem):
    def __init__(self, name, parent=None):
        super().__init__(name, parent)
        self.children = []  # list[WorkspaceItem]

    def add_child(self, item):
        item.parent = self
        self.children.append(item)

    def remove_child(self, item):
        self.children.remove(item)
        item.parent = None

    def find_child_by_name(self, name):
        for child in self.children:
            if child.name == name:
                return child
        return None

    def to_dict(self):
        return {
            "type": "folder",
            "name": self.name,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data, assets_dir):
        folder = cls(data["name"])

        for child_data in data.get("children", []):
            child_type = child_data.get("type")

            if child_type == "folder":
                child = Folder.from_dict(child_data, assets_dir)
            elif child_type == "entity":
                child = Entity.from_dict(child_data, assets_dir)
            else:
                raise ValueError(f"Unknown workspace item type: {child_type!r}")

            folder.add_child(child)

        return folder


class Entity(WorkspaceItem):
    """Wraps an AudioClip plus time-division boundaries."""

    def __init__(self, name, clip, divisions=None, parent=None, id=None):
        super().__init__(name, parent)
        self.clip = clip
        self.divisions = sorted(divisions) if divisions else []
        self.id = id or uuid.uuid4().hex

    @classmethod
    def from_file(cls, path, name=None):
        clip = AudioClip.load(path)
        return cls(name or clip.name, clip, divisions=[])

    @classmethod
    def from_entity(cls, other, new_name):
        return cls(new_name, other.clip.copy(), divisions=list(other.divisions))

    def get_segments(self):
        """Return list of (start_time, end_time) tuples covering the whole clip."""
        start_time = self.clip.get_time(self.clip.start_index)
        end_time = self.clip.get_time(self.clip.end_index())
        bounds = [start_time, *self.divisions, end_time]
        return list(zip(bounds[:-1], bounds[1:]))

    def add_division(self, time):
        start_time = self.clip.get_time(self.clip.start_index)
        end_time = self.clip.get_time(self.clip.end_index())
        if start_time < time < end_time and time not in self.divisions:
            self.divisions.append(time)
            self.divisions.sort()

    def remove_division(self, time):
        if time in self.divisions:
            self.divisions.remove(time)

    def to_dict(self):
        return {
            "type": "entity",
            "name": self.name,
            "id": self.id,
            "divisions": list(self.divisions),
        }

    @classmethod
    def from_dict(cls, data, assets_dir):
        entity_id = data["id"]
        asset_path = os.path.join(assets_dir, f"{entity_id}.wav")

        if not os.path.isfile(asset_path):
            raise FileNotFoundError(
                f"Missing audio asset for entity '{data['name']}': {asset_path}"
            )

        clip = AudioClip.load(asset_path)
        clip.name = data["name"]

        return cls(
            data["name"],
            clip,
            divisions=data.get("divisions", []),
            id=entity_id,
        )


class Workspace:
    """Top-level container: a name, a root Folder, and a disk location."""

    def __init__(self, name, path):
        self.name = name
        self.path = path            # folder on disk this workspace lives in
        self.root = Folder(name="root")
        self.dirty = False

    def mark_dirty(self):
        self.dirty = True

    def _collect_entities(self):
        entities = []

        def walk(folder):
            for child in folder.children:
                if isinstance(child, Entity):
                    entities.append(child)
                elif isinstance(child, Folder):
                    walk(child)

        walk(self.root)
        return entities

    def save(self):
        if not self.path:
            raise ValueError("Workspace has no path; cannot save.")

        self.name = os.path.basename(os.path.normpath(self.path))

        os.makedirs(self.path, exist_ok=True)
        assets_dir = os.path.join(self.path, "assets")
        os.makedirs(assets_dir, exist_ok=True)

        entities = self._collect_entities()

        # write/refresh audio assets (always stored internally as WAV)
        for entity in entities:
            asset_path = os.path.join(assets_dir, f"{entity.id}.wav")
            WavIO.unload(entity.clip, asset_path)

        # remove orphaned asset files (entities deleted since last save)
        valid_filenames = {f"{entity.id}.wav" for entity in entities}
        for fname in os.listdir(assets_dir):
            if fname not in valid_filenames:
                os.remove(os.path.join(assets_dir, fname))

        tree = {
            "name": self.name,
            "root": self.root.to_dict(),
        }

        json_path = os.path.join(self.path, "workspace.json")
        with open(json_path, "w") as f:
            json.dump(tree, f, indent=2)

        self.dirty = False

    @classmethod
    def load(cls, path):
        json_path = os.path.join(path, "workspace.json")

        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"No workspace.json found in {path}")

        with open(json_path, "r") as f:
            tree = json.load(f)

        assets_dir = os.path.join(path, "assets")

        workspace = cls(name=tree["name"], path=path)
        workspace.root = Folder.from_dict(tree["root"], assets_dir)
        workspace.dirty = False

        return workspace