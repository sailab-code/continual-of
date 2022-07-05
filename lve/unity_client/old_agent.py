#
# SAILenv is licensed under a MIT license.
#
# You should have received a copy of the license along with this
# work. If not, see <https://en.wikipedia.org/wiki/MIT_License>.

# Import packages
import os
from random import randint
from typing import Union

import numpy as np
import socket
from enum import IntFlag
import gzip
import struct
import io
from PIL import Image


# Import src
from typing.io import BinaryIO


class FrameFlags(IntFlag):
    NONE = 0
    MAIN = 1
    CATEGORY = 1 << 2
    OBJECT = 1 << 3
    OPTICAL = 1 << 4
    DEPTH = 1 << 5


class CommandsBytes:
    FRAME = b"\x00"
    DELETE = b"\x01"
    CHANGE_SCENE = b"\x02"
    GET_CATEGORIES = b"\x03"
    GET_POSITION = b"\x04"
    SET_POSITION = b"\x05"
    GET_ROTATION = b"\x06"
    SET_ROTATION = b"\x07"
    TOGGLE_FOLLOW = b"\x08"
    SPAWN_OBJECT = b"\x09"
    DESPAWN_OBJECT = b"\x0A"
    GET_SPAWNABLE_OBJECTS_NAMES = b"\x0B"
    SEND_OBJ_ZIP = b"\x0C"
    SET_MAIN_CAMERA_CLEAR_FLAGS = b"\x0D"
    GET_LIGHT_COLOR = b"\x0E"
    SET_LIGHT_COLOR = b"\x0F"
    GET_LIGHT_INTENSITY = b"\x10"
    SET_LIGHT_INTENSITY = b"\x11"
    GET_LIGHT_INDIRECT_MULTIPLIER = b"\x12"
    SET_LIGHT_INDIRECT_MULTIPLIER = b"\x13"
    GET_LIGHT_POSITION = b"\x14"
    SET_LIGHT_POSITION = b"\x15"
    GET_LIGHT_DIRECTION = b"\x16"
    SET_LIGHT_DIRECTION = b"\x17"
    GET_LIGHT_TYPE = b"\x18"
    GET_LIGHTS_NAMES = b"\x19"
    GET_AMBIENT_LIGHT_COLOR = b"\x1A"
    SET_AMBIENT_LIGHT_COLOR = b"\x1B"


class Agent:

    sizeof_int = 4  # sizeof(int) in C#
    sizeof_float = 4  # sizeof(float) in C#
    sizeof_double = 8  # sizeof(double) in C#
    # Note: boolean values are like integers

    def __init__(self,
                 main_frame_active: bool = True,
                 object_frame_active: bool = True,
                 category_frame_active: bool = True,
                 flow_frame_active: bool = True,
                 depth_frame_active: bool = True,
                 host: str = "localhost",
                 port: int = 8080,
                 width: int = 512,
                 height: int = 384,
                 use_gzip=False):
        """
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
        self.flow_frame_active: bool = flow_frame_active
        self.category_frame_active: bool = category_frame_active
        self.object_frame_active: bool = object_frame_active
        self.main_frame_active: bool = main_frame_active
        self.depth_frame_active: bool = depth_frame_active

        self.flags = 0
        self.flags |= FrameFlags.MAIN if main_frame_active else 0
        self.flags |= FrameFlags.CATEGORY if category_frame_active else 0
        self.flags |= FrameFlags.OBJECT if object_frame_active else 0
        self.flags |= FrameFlags.DEPTH if depth_frame_active else 0
        self.flags |= FrameFlags.OPTICAL if flow_frame_active else 0

        self.id: int = -1
        self.host = host
        self.port = port
        self.width = width
        self.height = height

        self.scenes = None
        self.categories = None
        self.spawnable_objects_names = None
        self.lights_names = None

        self.spawned_objects_idstr_names_table = dict()

        # Creates a TCP socket over IPv4
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.gzip = use_gzip

        # Numbers are sent as little endian
        # TODO: check if this is also true on linux or just windows.
        self.endianness = 'little'

    @property
    def active_frames(self):
        return {
            "main": self.main_frame_active,
            "category": self.category_frame_active,
            "object": self.object_frame_active,
            "flow": self.flow_frame_active,
            "depth": self.depth_frame_active
        }

    # region Netcode

    def __send_command(self, command):
        """
        Sends a command (of type CommandBytes).
        """
        self.socket.send(command)

    def __send_resolution(self):
        """
        Convert the resolution to bytes and send it over the socket.
        """
        resolution_bytes = self.width.to_bytes(4, self.endianness) + self.height.to_bytes(4, self.endianness)
        self.socket.send(resolution_bytes)

    def __send_gzip_setting(self):
        """
        Sends gzip option (boolean).
        """
        self.__send_bool(self.gzip)

    def __send_bool(self, boolean):
        """
        Sends a boolean.
        """
        data = b"\x01" if boolean else b"\x00"
        self.socket.send(data)

    def __send_int(self, number):
        """
        Sends an integer.
        """
        data = struct.pack("i", number)
        self.socket.send(data)

    def __send_float(self, number):
        """
        Sends a float.
        """
        data = struct.pack("f", number)
        self.socket.send(data)

    def __send_double(self, number):
        """
        Sends a double.
        """
        data = struct.pack("d", number)
        self.socket.send(data)

    def __send_vector3(self, vector):
        """
        Sends a vector3 (of floats).
        """
        for i in range(0, 3):
            self.__send_float(vector[i])

    def __send_string(self, string: str, str_format="utf-8"):
        """
        Sends a string.
        :param str_format: the format of the string
        """
        string_size = len(string)
        self.socket.send(string_size.to_bytes(4, self.endianness))
        data = string.encode(str_format)
        self.socket.send(data)

    def __receive_bool(self):
        """
        Receives a boolean.
        :return: the received boolean
        """
        # Note: the size of a boolean is the same of an integer
        data = self.receive_bytes(self.sizeof_int)
        return bool.from_bytes(data, self.endianness)

    def __receive_int(self):
        """
        Receives an integer.
        :return: the received integer
        """
        data = self.receive_bytes(self.sizeof_int)
        return int.from_bytes(data, self.endianness)

    def __receive_float(self):
        """
        Receives a float.
        :return: the received float
        """
        data = self.receive_bytes(self.sizeof_float)
        number = struct.unpack("f", data)
        return number[0]  # struct.unpack always returns a tuple with one item

    def __receive_double(self):
        """
        Receives a double.
        :return: the received double
        """
        data = self.receive_bytes(self.sizeof_double)
        number = struct.unpack("d", data)
        return number[0]  # struct.unpack always returns a tuple with one item

    def __receive_vector3(self):
        """
        Receives a vector3 (of floats).
        :return: the received vector3
        """
        x = self.__receive_float()
        y = self.__receive_float()
        z = self.__receive_float()
        return x, y, z

    def __receive_string(self, str_format="utf-8"):
        """
        Receives a string.
        :return: the received string
        """
        string_size = self.__receive_int()
        data = self.receive_bytes(string_size)
        return data.decode(str_format)

    def __receive_agent_id(self):
        """
        Receives agent id (an integer).
        """
        self.id = self.__receive_int()

    def __receive_categories(self):
        """
        Sends a get categories command and receives a list of available categories (name, id).
        """
        self.__send_get_categories()

        categories_number = self.__receive_int()
        categories = dict()
        colors = dict()

        for i in range(categories_number):
            cat_id = self.__receive_int()
            cat_name = self.__receive_string()
            categories[cat_id] = cat_name
            colors[cat_id] = [
                randint(0, 255),
                randint(0, 255),
                randint(0, 255)
            ]

        self.categories = categories
        self.cat_colors = colors

    def __receive_spawnable_objects_names(self):
        """
        Sends a get spawnable objects name command and receives a list of spawnable object names (strings).
        """
        self.__send_get_spawnable_objects_names()

        spawnable_objects_number = self.__receive_int()
        prefab_names = []

        for i in range(spawnable_objects_number):
            prefab_name = self.__receive_string()
            prefab_names.append(prefab_name)

        self.spawnable_objects_names = prefab_names

    def __receive_lights_names(self):
        """
        Sends a get lights names command and receives a list of lights names (strings).
        """
        self.__send_get_lights_names()

        lights_names_number = self.__receive_int()
        lights_names = []

        for i in range(lights_names_number):
            light_name = self.__receive_string()
            lights_names.append(light_name)

        self.lights_names = lights_names

    def __receive_scenes(self):
        """
        Receives a list of available scene names.
        """
        scenes_number = self.__receive_int()
        scenes = list()

        for i in range(scenes_number):
            scenes_name = self.__receive_string()
            scenes.append(scenes_name)

        self.scenes = scenes

    def __send_get_categories(self):
        """
        Sends a get categories command.
        """
        self.__send_command(CommandsBytes.GET_CATEGORIES)

    def __send_get_spawnable_objects_names(self):
        """
        Sends a get spawnable objects command.
        """
        self.__send_command(CommandsBytes.GET_SPAWNABLE_OBJECTS_NAMES)

    def __send_get_lights_names(self):
        """
        Sends a get lights names command.
        """
        self.__send_command(CommandsBytes.GET_LIGHTS_NAMES)

    def __send_bytes(self, data):
        """
        Sends bytes in the socket.
        :param data: the bytes that must be sent
        """
        self.socket.sendall(data)

    def __send_file(self, file: BinaryIO, filename: str):
        """
        Sends a file in the socket.
        :param file: file to be sent
        """
        self.__send_string(filename)

        # get file size
        file.seek(0, io.SEEK_END)
        file_len = file.tell()
        file.seek(0)

        self.__send_int(file_len)
        self.socket.sendfile(file)

    # endregion Netcode

    # region Public commands

    def register(self):
        """
        Register the agent on the Unity server and set its id.
        """
        # Connect to the unity socket
        self.socket.connect((self.host, self.port))
        self.__send_resolution()
        self.__send_gzip_setting()
        self.__receive_agent_id()
        self.__receive_scenes()

    def delete(self):
        """
        Delete the agent on the Unity server.
        """
        self.socket.send(CommandsBytes.DELETE)

    def get_frame(self):
        """
        Get the frame from the cameras on the Unity server.
        :return: a dict of frames indexed by keys main, object, category, flow, depth. It has also a "sizes" key
        containing the size in byte of the received frame (different from the actual size of the frame when gzip
        compression is enabled)
        """
        # initialize the frame dictionary
        frame = {"sizes": {}}

        # encodes the flags in a single byte
        flags_bytes = self.flags.to_bytes(1, self.endianness)
        # adds the FRAME request byte.

        self.__send_command(CommandsBytes.FRAME + flags_bytes)

        # start reading images from socket in the following order:
        # main, category, object, optical flow, depth
        if self.main_frame_active:
            frame_bytes, received = self.receive_next_frame_view()

            frame["main"] = self.__decode_image(frame_bytes)
            frame["sizes"]["main"] = received
        else:
            frame["sizes"]["main"] = 0
            frame["main"] = None

        if self.category_frame_active:
            frame_bytes, received = self.receive_next_frame_view()

            cat_frame = self.__decode_category(frame_bytes)
            cat_frame = np.reshape(cat_frame, (self.height, self.width, 3))
            cat_frame = cat_frame[:, :, 0]
            frame["category"] = cat_frame
            frame["sizes"]["category"] = received
        else:
            frame["sizes"]["category"] = 0
            frame["category"] = None
            # frame["category_debug"] = None

        if self.object_frame_active:
            frame_bytes, received = self.receive_next_frame_view()
            frame["object"] = self.__decode_image(frame_bytes)
            frame["sizes"]["object"] = received
        else:
            frame["sizes"]["object"] = 0
            frame["object"] = None

        if self.flow_frame_active:
            frame_bytes, received = self.receive_next_frame_view()
            flow = np.frombuffer(frame_bytes, np.float32)
            frame["flow"] = self.__decode_flow(flow)
            frame["sizes"]["flow"] = received
        else:
            frame["sizes"]["flow"] = 0
            frame["flow"] = None

        if self.depth_frame_active:
            frame_bytes, received = self.receive_next_frame_view()
            frame["depth"] = self.__decode_image(frame_bytes)
            frame["sizes"]["depth"] = received
        else:
            frame["sizes"]["depth"] = 0
            frame["depth"] = None

        frame["deltaTime"] = self.__receive_float()

        return frame

    def change_scene(self, scene_name):
        """
        Sends a change scene command.
        :param scene_name: the name of the scene to load into the Unity server
        """
        self.__send_command(CommandsBytes.CHANGE_SCENE)
        self.__send_string(scene_name)

        result = self.__receive_string()
        if result != "ok":
            print(f"Cannot change scene! error = {result}")
            return

        self.__receive_categories()
        self.__receive_spawnable_objects_names()
        self.__receive_lights_names()

    def receive_bytes(self, n_bytes):
        """
        Receives exactly n_bytes from the socket.
        :param n_bytes: Number of bytes that should be received from the socket
        :return an array of n_bytes bytes
        """
        received = 0
        bytes_array = b''
        while received < n_bytes:
            chunk = self.socket.recv(n_bytes - received)
            received += len(chunk)
            bytes_array += chunk
        return bytes_array

    def receive_next_frame_view(self):
        """
        Receives the next frame from the socket.
        :return: a byte array containing the encoded frame view
        """
        frame_length = int.from_bytes(self.socket.recv(4), self.endianness)  # first we read the length of the frame
        frame_bytes = self.receive_bytes(frame_length)

        if self.gzip:
            try:
                return gzip.decompress(frame_bytes), frame_length
            except:
                print("Error decompressing gzip frame: is gzip enabled on the server?")
        else:
            return frame_bytes, frame_length

    def get_categories(self):
        """
        Gets the categories list from the socket.
        :return: the list of categories
        """
        self.__receive_categories()
        return self.categories

    def get_position(self):
        """
        Sends a get position command.
        :return: the vector3 (of floats) position of the agent
        """
        self.__send_command(CommandsBytes.GET_POSITION)
        position = self.__receive_vector3()
        return position

    def get_rotation(self):
        """
        Sends a get rotation (euler angles) command.
        :return: the vector3 (of floats) rotation of the agent
        """
        self.__send_command(CommandsBytes.GET_ROTATION)
        rotation = self.__receive_vector3()
        return rotation

    def set_position(self, position):
        """
        Sends a set position command.
        :param position: the vector3 (of floats) position of the agent
        """
        self.__send_command(CommandsBytes.SET_POSITION)
        self.__send_vector3(position)
        result = self.__receive_string()
        if result != "ok":
            print("Error setting position")

    def set_rotation(self, rotation):
        """
        Sends a set rotation (euler angles) command.
        :param rotation: the vector3 (of floats) rotation of the agent
        """
        self.__send_command(CommandsBytes.SET_ROTATION)
        self.__send_vector3(rotation)
        result = self.__receive_string()
        if result != "ok":
            print(f"Error setting rotation: {result}")

    def toggle_follow(self):
        """
        Sends a toggle follow command.
        """
        self.__send_command(CommandsBytes.TOGGLE_FOLLOW)
        result = self.__receive_string()
        if result != "ok":
            print("Error toggling follow")

    def spawn_object(self, name,
                     position=(0, 0, 0), rotation=(0, 0, 0),
                     remove_dynamics=True, scale=(1, 1, 1), use_parent=True):
        """
        Spawn an object into the Unity server by sending a spawn object command. Note: this also adds the spawned object,
        if actually spawned, to the spawned object idstr names table of the agent.
        :param name: the name of the object to spawn
        :param position: the vector3 (of floats) position where to spawn
        :param rotation: the vector3 (of floats) rotation (euler angles) at which to spawn
        :param remove_dynamics: a flag to remove or not the dynamics (gravity, movement, etc) from the spawned object
        :param scale: the vector3 (of floats) scale of the object once spawned
        :param use_parent: a flag to spawn the object a child of the default parent (as defined in the Unity server)
        :return: a string empty if the object is not spawned, containing the object unique instance id otherwise
        """
        self.__send_command(CommandsBytes.SPAWN_OBJECT)
        self.__send_string(name)
        self.__send_vector3(position)
        self.__send_vector3(rotation)
        self.__send_bool(remove_dynamics)
        self.__send_vector3(scale)
        self.__send_bool(use_parent)
        object_instance_id = self.__receive_string()
        # If the instance id is an empty string, the spawn is not successful
        if not object_instance_id:
            print("Error object " + name + " not spawned")
            return object_instance_id
        # If the object is spawned add the instance id to the table
        self.spawned_objects_idstr_names_table[object_instance_id] = name
        return object_instance_id

    def despawn_object(self, idstr):
        """
        Despawn an object identified by the given instance id (in string format), by sending a despawn object command.
        Note: this also removes the despawned object, if actually despawned, from the spawned object idstr names table
        of the agent.
        :param idstr: the instance id of the object to despawn in string format
        """
        # Convert the idstr to an integer id
        id_number = int(idstr)
        # Actually send the commands
        self.__send_command(CommandsBytes.DESPAWN_OBJECT)
        self.__send_int(id_number)
        result = self.__receive_string()
        if result != "ok":
            print("Error despawning object with id " + idstr)
            return
        # Remove the entry from the dictionary
        self.spawned_objects_idstr_names_table.pop(idstr)

    def send_obj_zip(self, file: Union[str, BinaryIO], filename=None):
        """
        Send a .zip file with a .obj model inside to the Unity server. It also returns the path of the .zip file.
        """
        self.__send_command(CommandsBytes.SEND_OBJ_ZIP)
        if isinstance(file, str):
            filename = os.path.basename(file)
            with open(file, "rb") as file:
                self.__send_file(file, filename)
        else:
            if filename is None:
                raise ValueError("filename cannot be None if an open file is provided")
            self.__send_file(file, filename)
        # After sending the object, we receive the path where the file was stored
        save_path = self.__receive_string()
        return save_path

    def change_main_camera_clear_flags(self, r: int, g: int, b: int):
        """
        Change the camera clear flags (a solid color) on the Unity agent main camera.
        :param r: the red channel of the solid color
        :param g: the green channel of the solid color
        :param b: the blue channel of the solid color
        Note: sending negative values for at least one channel means to use the skybox
        """
        self.__send_command(CommandsBytes.SET_MAIN_CAMERA_CLEAR_FLAGS)
        # Send the color
        self.__send_int(r)
        self.__send_int(g)
        self.__send_int(b)
        result = self.__receive_string()
        if result != "ok":
            print("Error sending clear flags")

    def get_light_color(self, light_name: str):
        """
        Get the (R,G,B) color tuple of the given light.
        :param light_name: the name of the light
        :return: a tuple (R,G,B) defining the color or None if the light name is not defined
        """
        self.__send_command(CommandsBytes.GET_LIGHT_COLOR)
        # Send the name
        self.__send_string(light_name)
        result = self.__receive_string()
        if result == "ok":
            # Receive the color
            r = self.__receive_int()
            g = self.__receive_int()
            b = self.__receive_int()
            return r, g, b
        print("Error getting light color")
        return None

    def set_light_color(self, light_name: str, r: int, g: int, b: int):
        """
        Set the (R,G,B) color tuple of the given light.
        :param light_name: the name of the light
        :param r: the red channel of the light color
        :param g: the green channel of the light color
        :param b: the blue channel of the light color
        """
        self.__send_command(CommandsBytes.SET_LIGHT_COLOR)
        # Send the name
        self.__send_string(light_name)
        # Send the color
        self.__send_int(r)
        self.__send_int(g)
        self.__send_int(b)
        result = self.__receive_string()
        if result != "ok":
            print("Error setting light color")

    def get_light_position(self, light_name: str):
        """
        Get the Vector3 position of the given light.
        :param light_name: the name of the light
        :return: a Vector3 defining the position or None if the light name is not defined
        """
        self.__send_command(CommandsBytes.GET_LIGHT_POSITION)
        # Send the name
        self.__send_string(light_name)
        result = self.__receive_string()
        if result == "ok":
            # Receive the position
            position = self.__receive_vector3()
            return position
        print("Error getting light position")
        return None

    def set_light_position(self, light_name: str, position):
        """
        Set the Vector3 position of the given light. It has meaning only for non-directional lights.
        :param light_name: the name of the light
        :param position: the Vector3 position of the light
        """
        self.__send_command(CommandsBytes.SET_LIGHT_POSITION)
        # Send the name
        self.__send_string(light_name)
        # Send the position
        self.__send_vector3(position)
        result = self.__receive_string()
        if result != "ok":
            print("Error setting light position")

    def get_light_direction(self, light_name: str):
        """
        Get the Vector3 direction of the given light. It has meaning only for non-point lights.
        :param light_name: the name of the light
        :return: a Vector3 defining the direction or None if the light name is not defined
        """
        self.__send_command(CommandsBytes.GET_LIGHT_DIRECTION)
        # Send the name
        self.__send_string(light_name)
        result = self.__receive_string()
        if result == "ok":
            # Receive the direction
            direction = self.__receive_vector3()
            return direction
        print("Error getting light direction")
        return None

    def set_light_direction(self, light_name: str, direction):
        """
        Set the Vector3 direction of the given light. It has meaning only for non-point lights.
        :param light_name: the name of the light
        :param direction: the Vector3 direction of the light
        """
        self.__send_command(CommandsBytes.SET_LIGHT_DIRECTION)
        # Send the name
        self.__send_string(light_name)
        # Send the direction
        self.__send_vector3(direction)
        result = self.__receive_string()
        if result != "ok":
            print("Error setting light direction")

    def get_light_intensity(self, light_name: str):
        """
        Get the float intensity of the given light.
        :param light_name: the name of the light
        :return: a float defining the intensity or None if the light name is not defined
        """
        self.__send_command(CommandsBytes.GET_LIGHT_INTENSITY)
        # Send the name
        self.__send_string(light_name)
        result = self.__receive_string()
        if result == "ok":
            # Receive the intensity
            intensity = self.__receive_float()
            return intensity
        print("Error getting light intensity")
        return None

    def set_light_intensity(self, light_name: str, intensity: float):
        """
        Set the float intensity of the given light.
        :param light_name: the name of the light
        :param intensity: the intensity float of the light
        """
        self.__send_command(CommandsBytes.SET_LIGHT_INTENSITY)
        # Send the name
        self.__send_string(light_name)
        # Send the intensity
        self.__send_float(intensity)
        result = self.__receive_string()
        if result != "ok":
            print("Error setting light intensity")

    def get_light_indirect_multiplier(self, light_name: str):
        """
        Get the float indirect multiplier of the given light.
        :param light_name: the name of the light
        :return: a float defining the indirect multiplier or None if the light name is not defined
        """
        self.__send_command(CommandsBytes.GET_LIGHT_INDIRECT_MULTIPLIER)
        # Send the name
        self.__send_string(light_name)
        result = self.__receive_string()
        if result == "ok":
            # Receive the indirect multiplier
            indirect_multiplier = self.__receive_float()
            return indirect_multiplier
        print("Error getting light indirect multiplier")
        return None

    def set_light_indirect_multiplier(self, light_name: str, indirect_multiplier: float):
        """
        Set the float indirect multiplier of the given light.
        :param light_name: the name of the light
        :param indirect_multiplier: the indirect multiplier float of the light
        """
        self.__send_command(CommandsBytes.SET_LIGHT_INDIRECT_MULTIPLIER)
        # Send the name
        self.__send_string(light_name)
        # Send the indirect multiplier
        self.__send_float(indirect_multiplier)
        result = self.__receive_string()
        if result != "ok":
            print("Error setting light indirect multiplier")

    def get_light_type(self, light_name: str):
        """
        Get the string type of the given light.
        :param light_name: the name of the light
        :return: a string defining the type or None if the light name is not defined
        """
        self.__send_command(CommandsBytes.GET_LIGHT_TYPE)
        # Send the name
        self.__send_string(light_name)
        result = self.__receive_string()
        if result == "light_not_found":
            print("Error getting light type")
            return None
        return result

    def get_ambient_light_color(self):
        """
        Get the (R,G,B) color tuple of the scene ambient light. It only works if scene ambient light is set to solid color.
        :return: a tuple (R,G,B) defining the ambient light color or None if the scene ambient light is not set to solid color.
        """
        self.__send_command(CommandsBytes.GET_AMBIENT_LIGHT_COLOR)
        # Get the result
        result = self.__receive_string()
        if result == "ok":
            # Receive the color
            r = self.__receive_int()
            g = self.__receive_int()
            b = self.__receive_int()
            return r, g, b
        print("Error getting ambient light color")
        return None

    def set_ambient_light_color(self, r: int, g: int, b: int):
        """
        Set the (R,G,B) color tuple of the scene ambient light.
        :param r: the red channel of the scene ambient light color
        :param g: the green channel of the scene ambient light color
        :param b: the blue channel of the scene ambient light color
        """
        self.__send_command(CommandsBytes.SET_AMBIENT_LIGHT_COLOR)
        # Send the color
        self.__send_int(r)
        self.__send_int(g)
        self.__send_int(b)
        result = self.__receive_string()
        if result != "ok":
            print("Error setting ambient light color")

    # endregion Public commands

    # region Data decoding and image manipulation

    @staticmethod
    def __decode_image(image_bytes) -> np.ndarray:
        """
        Decode an image from the given bytes representation to a numpy array.
        :param image_bytes: the bytes representation of an image
        :return: a PIL Image made from those bytes
        """

        bytes_buffer = io.BytesIO()
        bytes_buffer.write(image_bytes)
        pil_img = Image.open(bytes_buffer)
        return np.asarray(pil_img, dtype=np.float32) / 255

    def __decode_category(self, frame_bytes) -> np.ndarray:
        """
        Decode the category supervisions from the given base64 representation to a numpy array.
        :param input_base64: the base64 representation of categories
        :return: the numpy array containing the category supervisions
        """
        cat_frame = np.frombuffer(frame_bytes, dtype=np.uint8)
        cat_frame = np.reshape(cat_frame, (self.height, self.width, -1))
        cat_frame = np.flipud(cat_frame)
        cat_frame = np.reshape(cat_frame, (-1))
        cat_frame = np.ascontiguousarray(cat_frame)
        return cat_frame

    def __decode_flow(self, flow_frame: np.ndarray) -> np.ndarray:
        """
        Decodes the flow frame.
        :param flow_frame: frame obtained by unity. It is a float32 array.
        :return: Decoded flow frame
        """
        flow = flow_frame

        # must be flipped upside down, because Unity returns it from bottom to top.
        flow = np.reshape(flow, (self.height, self.width, -1))
        flow = np.flipud(flow)

        # restore it as contiguous array, as flipud breaks the contiguity
        flow = np.ascontiguousarray(flow)
        return flow

    # endregion