import json
import os
import shutil
from gzip import GzipFile
from pathlib import Path
from PIL import Image
import numpy as np

import cv2

from lve.utils import crop_center


def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname,img)


class CommandsBuffer():
    def __init__(self):
        self.buf = []

    def push(self, l):
        self.buf += l

    def get(self):
        commands = list(self.buf)
        self.buf = []
        return commands


class Object():
    def __init__(self, id, weight_factor=1.0):
        self.id = id
        self.force = Force()
        self.torque = Force()
        self.weight_factor = weight_factor


class Force():
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def add(self, x, y, z):
        self.x += x
        self.y += y
        self.z += z

    def get(self):
        return {'x': self.x, 'y': self.y, 'z': self.z}

    def scale(self, factor):
        self.x *= factor
        self.y *= factor
        self.z *= factor


class StreamSaver():
    def __init__(self, path, stream_name, agent, crop=False, resize=None, options={}):
        self.agent = agent
        self.crop = crop
        self.motion_buffered = None
        stream_name_opts = "_".join([opt + str(val).replace(":", "-") for opt, val in options.items()])
        stream_name = stream_name + "_" + stream_name_opts + "_" + str(np.random.randint(1000,9999))
        self.path = Path(path).joinpath(os.path.basename(stream_name))
        self.path.mkdir(exist_ok=True)
        try:
            shutil.rmtree(self.path.joinpath("frames"))
            shutil.rmtree(self.path.joinpath("motion"))
            print('Cleaned previous files')
        except:
            pass

        if crop and resize is not None:
            raise ValueError('crop and resize modes are mutually exclusive')
        self.resize = resize
        self.target_size = (agent.width, agent.height) if resize is None else resize

        self.path.joinpath("frames").mkdir(exist_ok=True, parents=True)
        with open(self.path.joinpath("frames", 'fps.json'), 'w') as outfile:
            json.dump({"fps": 25.0}, outfile)

    def save(self, i, frame, motion, delay_motion=True):
        if delay_motion:
            motion_ = np.zeros((frame.shape[0], frame.shape[1], 2)) if self.motion_buffered is None else self.motion_buffered
        else:
            motion_ = motion
        files_per_folder = 100
        num_folder = (i // files_per_folder) +1
        num_file = (i % files_per_folder) + 1

        str_folder = str(num_folder).zfill(8)
        frame_container_path = self.path.joinpath("frames", str_folder)
        motion_container_path = self.path.joinpath("motion", str_folder)
        frame_container_path.mkdir(exist_ok=True, parents=True)
        motion_container_path.mkdir(exist_ok=True, parents=True)

        str_file = str(num_file).zfill(3)

        if self.crop and (self.agent.width != frame.shape[1] or self.agent.height != frame.shape[0]):
            frame = crop_center(frame, self.target_size[0], self.target_size[1])
            motion_ = crop_center(motion_, self.target_size[0], self.target_size[1])
        elif self.resize is not None:
            frame = cv2.resize(frame, self.target_size, interpolation= cv2.INTER_AREA)
            motion_ = cv2.resize(motion_, self.target_size, interpolation=cv2.INTER_AREA) / np.array([1. * self.agent.width / self.target_size[0], 1. * self.agent.height / self.target_size[1]])

        im = Image.fromarray(frame)
        im.save(frame_container_path.joinpath(str_file + '.png'))

        with GzipFile(motion_container_path.joinpath(str_file + '.bin'), "wb") as f:
            np.save(f, motion_)

        if delay_motion:
            self.motion_buffered = motion

        self.i = i

    def finalize(self):
        with open(self.path.joinpath("frames", 'fps.json'), 'w') as outfile:
            json.dump({"fps": 25.0, "frames_count": self.i + 1}, outfile)
