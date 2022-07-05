import json
import urllib
from pathlib import Path
from typing import Dict, List

from tdw.controller import Controller
from tdw.add_ons.floorplan import Floorplan
from tdw.tdw_utils import TDWUtils


class CachedFloorPlanController(Controller):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.floorplan = Floorplan()
        self.cache_path = Path(kwargs.get("cache_path", "./cache"))
        self.cache_path.mkdir(exist_ok=True)
        self.files_path = self.cache_path.joinpath("files")
        self.files_path.mkdir(exist_ok=True)

        self.cache: Dict[str, str] = {}

        if self.cache_path.joinpath("./cache.json").exists():
            self.cache = json.load(self.cache_path.joinpath("./cache.json").open("r"))

    def get_scene_init_commands(self, scene: str, layout: int, audio: bool) -> List[dict]:
        self.floorplan.init_scene(scene=scene, layout=layout)
        commands = self.floorplan.commands

        for i in range(len(commands)):
            if commands[i]["$type"] == "add_scene" or commands[i]["$type"] == 'add_object':
                remote_url = commands[i]["url"]
                if remote_url in self.cache:
                    print(f"{remote_url} in cache, retrieving")
                    local_url = Path(self.cache[remote_url])
                    commands[i]["url"] = f"file:///{local_url}"
                else:
                    print(f"{remote_url} not in cache, downloading")
                    # download remote_url to cache
                    local_url = self.files_path.joinpath(remote_url.split("/")[-1])
                    local_url.parent.mkdir(exist_ok=True)

                    # download remote_url with urllib
                    urllib.request.urlretrieve(remote_url, local_url)

                    self.cache[remote_url] = str(local_url.absolute())
                    commands[i]["url"] = f"file:///{local_url}"

        # save cache
        json.dump(self.cache, self.cache_path.joinpath("./cache.json").open("w"))

        return commands