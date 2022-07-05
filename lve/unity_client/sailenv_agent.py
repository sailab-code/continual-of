from .agent import UnityAgent
from sailenv.agent import Agent

class SailenvAgent(UnityAgent):
    def __init__(self, 
                 main_frame_active: bool = True, 
                 object_frame_active: bool = True, 
                 category_frame_active: bool = True, 
                 flow_frame_active: bool = True, 
                 depth_frame_active: bool = True, 
                 host: str = "localhost", 
                 port: int = 8080, 
                 width: int = 512, 
                 height: int = 384):
        super().__init__(
            main_frame_active=main_frame_active, 
            object_frame_active=object_frame_active, 
            category_frame_active=category_frame_active, 
            flow_frame_active=flow_frame_active, 
            depth_frame_active=depth_frame_active, 
            host=host, 
            port=port, 
            width=width, 
            height=height
        )

    @property 
    def scenes(self):
        return self.__agent.scenes

    @property
    def categories(self):
        return self.__agent.categories

    def register(self):
        self.__agent.register()

    def delete(self):
        self.__agent.delete()

    def change_scene(self, scene_name: str):
        return self.__agent.change_scene(scene_name)

    def get_frame(self):
        return self.__agent.get_frame()

    def toggle_follow(self):
        return self.__agent.toggle_follow()

    def change_main_camera_flags(self, r: int, g: int, b: int):
        return self.__agent.change_main_camera_flags(r, g, b)
