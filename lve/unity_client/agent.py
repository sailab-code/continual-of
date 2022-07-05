import abc
from enum import IntFlag

class FrameFlags(IntFlag):
    NONE = 0
    MAIN = 1
    CATEGORY = 1 << 2
    OBJECT = 1 << 3
    OPTICAL = 1 << 4
    DEPTH = 1 << 5


class UnityAgent(metaclass=abc.ABCMeta):
    """
    Abstract class for Unity agents.
    """

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
        """
        Constructor.
        :param main_frame_active: True if the virtual world should generate the main camera view
        :param object_frame_active: True if the virtual world should generate object instance supervisions
        :param category_frame_active: True if the virtual world should generate category supervisions
        :param flow_frame_active: True if the virtual world should generate optical flow data
        :param host: address on which the unity virtual world is listening
        :param port: port on which the unity virtual world is listening
        :param width: width of the stream
        :param height: height of the stream
        :param use_gzip: true if the virtual world should compress the views with gzip
        
        """
        self.main_frame_active = main_frame_active
        self.object_frame_active = object_frame_active
        self.category_frame_active = category_frame_active
        self.flow_frame_active = flow_frame_active
        self.depth_frame_active = depth_frame_active
        self.host = host
        self.port = port
        self.width = width
        self.height = height

        self.flags = 0
        self.flags |= FrameFlags.MAIN if main_frame_active else 0
        self.flags |= FrameFlags.CATEGORY if category_frame_active else 0
        self.flags |= FrameFlags.OBJECT if object_frame_active else 0
        self.flags |= FrameFlags.DEPTH if depth_frame_active else 0
        self.flags |= FrameFlags.OPTICAL if flow_frame_active else 0

    @property
    def active_frames(self):
        return {
            "main": self.main_frame_active,
            "category": self.category_frame_active,
            "object": self.object_frame_active,
            "flow": self.flow_frame_active,
            "depth": self.depth_frame_active
        }

    # define abstract property scenes
    @property
    @abc.abstractmethod
    def scenes(self):
        """
        Get a list of available scenes
        """
        pass

    # define abstract property categories
    @property
    @abc.abstractmethod
    def categories(self):
        """
        Get a dict of available categories
        """
        pass

    # define abstract register
    @abc.abstractmethod
    def register(self):
        """
        Register the agent with the virtual world.
        """
        pass

    # define abstract delete
    @abc.abstractmethod
    def delete(self):
        """
        Delete the agent from the virtual world.
        """
        pass

    # define abstract change_scene
    @abc.abstractmethod
    def change_scene(self, scene_name: str):
        """
        Change the scene of the agent.
        :param scene_name: name of the scene
        """
        pass

    # define abstract get_frame
    @abc.abstractmethod
    def get_frame(self):
        """
        Get the current frame from the virtual world.
        :return: :return: a dict of frames indexed by keys 'main', 'object', 'category', 'flow', 'depth'. It has also a "sizes" key
        containing the size in byte of the received frame
        """
        pass

    # define abstract toggle_follow
    @abc.abstractmethod
    def toggle_follow(self):
        """
        Toggle the follow mode of the agent.
        """
        pass

    # define abstract change_main_camera_flags
    # TODO: NOT SURE I SHOULD KEEP THIS IN ABSTRACT CLASS
    @abc.abstractmethod
    def change_main_camera_flags(self, r: int, g: int, b: int):
        """
        Change the camera clear flags (a solid color) on the Unity agent main camera.
        :param r: the red channel of the solid color
        :param g: the green channel of the solid color
        :param b: the blue channel of the solid color
        Note: sending negative values for at least one channel means to use the skybox
        """
        pass
