import sys
import os
import cv2
from glob import glob
import numpy as np
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from gzip import GzipFile
import ffmpeg
import re
from lve.unity_client.sailenv_agent import SailenvAgent
from lve.unity_client.tdw_agent import TDWAgent

#from lve.unity_client import SocketAgent


class InputType(Enum):
    VIDEO_FILE = 0
    IMAGE_FOLDER = 1
    OUT_STREAM_FOLDER = 2
    WEB_CAM = 3
    ARRAYS = 4
    SAILENV = 5
    TDW = 6

    @staticmethod
    def readable_type(input_type):
        if input_type == InputType.VIDEO_FILE:
            input_type = "VIDEO_FILE"
        elif input_type == InputType.IMAGE_FOLDER:
            input_type = "IMAGE_FOLDER"
        elif input_type == InputType.OUT_STREAM_FOLDER:
            input_type = "OUT_STREAM_FOLDER"
        elif input_type == InputType.WEB_CAM:
            input_type = "WEB_CAM"
        elif input_type == InputType.ARRAYS:
            input_type = "ARRAYS"
        elif input_type == InputType.SAILENV:
            input_type = "SAILENV"
        elif input_type == InputType.TDW:
            input_type = "TDW"
        return input_type


class InputStream:

    def __init__(self, input_element, w=None, h=None,
                 fps=None, force_gray=False,
                 repetitions=1, max_frames=None, shuffle=False,
                 frame_op=None, foa_file=None, unity_settings=None, skip_frames=None):

        # standardizing input arguments
        if w is None:
            w = int(-1)
        if h is None:
            h = int(-1)
        if fps is None:
            fps = -1.0
        if max_frames is None:
            max_frames = int(-1)
        if skip_frames is None:
            skip_frames = int(-1)

        # features of the requested input stream
        self.input_element = input_element
        self.input, self.input_type, self.readable_input = InputStream.__get_input_features(input_element)
        self.w = int(w)
        self.h = int(h)
        self.c = None
        self.fps = float(fps)
        self.force_gray = force_gray
        self.repetitions = repetitions
        self.max_frames = max_frames
        self.frames = -1
        self.shuffle = shuffle
        self.frame_op = frame_op
        self.foa = foa_file
        self.unity_settings = unity_settings

        # checking
        if repetitions <= 0:
            raise ValueError("Invalid number of repetitions!")
        if self.h * self.w <= 0 or self.h < 0 and self.w < 0:
            self.h = -1
            self.w = -1

        # features of the original input stream
        self.w_orig = -1
        self.h_orig = -1
        self.c_orig = -1
        self.fps_orig = -1.0
        self.frames_orig = -1
        self.length_in_seconds_orig = 0.0
        self.sup_map = {}

        # other (private) stuff
        self.__files_per_folder = 100

        self.__video_capture = None
        self.__rotation_code = None
        self.__last_returned_time_in_original_video = 0.0
        self.__image_folder_files = None
        self.__last_img_in_input_index = -1
        self.__shuffled_order = None
        self.__unity_agent = None

        self.__last_returned_frame_number = 0  # the first frame returned by get_next(...) is numbered with 1
        self.__last_returned_time = 0.0  # time associated to the last frame that was returned by get_next(...), in ms

        # getting information from the video stream
        self.__getinfo()
        if skip_frames > 0:
            for i in range(skip_frames):
                self.get_next(skip_if_possible=True)

    def set_file_per_folder(self, n):
        self.__files_per_folder = n

    def get_next(self, sample_only=False, t=None, skip_if_possible=False):
        img = None  # frame to return
        of = None  # motion field to return (if available)
        supervisions = None  # supervisions to return (if available)
        foa = None  # focus of attention (x,y,vx,vy - if available)

        next_time = None  # time (in seconds) of the frame that we will get
        next_time_in_original_video = None  # time (in seconds) of the frame that we will get (in the original video)
        f = None  # file-index (when processing folders of images) of the frame to return

        # check
        if self.__last_returned_frame_number >= self.frames > 0:
            return None, None, None, None  # frame, motion, supervisions, foa
        if 0 < self.max_frames <= self.__last_returned_frame_number:
            return None, None, None, None  # frame, motion, supervisions, foa

        # opening stream (if not already opened)
        if self.input_type == InputType.VIDEO_FILE or self.input_type == InputType.WEB_CAM:
            if self.__video_capture is None or not self.__video_capture.isOpened():
                self.__video_capture = cv2.VideoCapture(self.input)
        else:
            f = self.__last_img_in_input_index + 1

        # setting time for the frame that we are going to get
        if self.input_type != InputType.WEB_CAM and self.input_type != InputType.SAILENV:  # sometimes cannot seek back
            if t is None:
                if self.fps != self.fps_orig:
                    next_time = self.__last_returned_time + (1.0 / self.fps)
                    next_time_in_original_video = self.__last_returned_time_in_original_video + (1.0 / self.fps)

                    if self.__video_capture is not None:
                        self.__video_capture.set(cv2.CAP_PROP_POS_MSEC, next_time_in_original_video * 1000.0)
                    f = int(Decimal(next_time_in_original_video * self.fps_orig).quantize(0, ROUND_HALF_UP))
            else:
                next_time = t
                next_time_in_original_video = t - math.floor(
                    t / self.length_in_seconds_orig) * self.length_in_seconds_orig
                if next_time >= self.__last_returned_time or sample_only:
                    if self.input_type == InputType.VIDEO_FILE:
                        self.__video_capture.set(cv2.CAP_PROP_POS_MSEC, next_time_in_original_video * 1000.0)
                    f = Decimal(next_time_in_original_video * self.fps_orig).quantize(0, ROUND_HALF_UP)
                else:
                    raise IOError("Cannot seek back in time!")

        # getting a new frame (video file or web-cam)
        if self.input_type == InputType.VIDEO_FILE or self.input_type == InputType.WEB_CAM:
            ret_val, img = self.__video_capture.read()

            # reached the end of video (or some weir errors occurred)
            if not ret_val:
                self.__video_capture.release()
                self.__video_capture = None

                # recursive call (next repetition)
                if self.repetitions > 1:
                    self.__last_returned_time_in_original_video = 0.0
                    return self.get_next(sample_only=sample_only, t=t)

            if img is None:
                return None, None, None, None

        # getting a new frame (folder created by the output stream)
        elif self.input_type == InputType.OUT_STREAM_FOLDER:
            if self.shuffle:
                f = self.__shuffled_order[f]

            n_folder = int(f / self.__files_per_folder) + 1
            n_file = (f + 1) - ((n_folder - 1) * self.__files_per_folder)

            folder_name = format(n_folder, '08d')
            file_name = format(n_file, '03d')
            file_name_longformat = format(n_file, '05d')

            if os.path.exists(self.input + os.sep + "frames" + os.sep + folder_name + os.sep + file_name + ".png"):
                img = cv2.imread(self.input + os.sep + "frames" + os.sep + folder_name + os.sep + file_name + ".png")
            elif os.path.exists(self.input + os.sep + "frames" + os.sep + folder_name + os.sep + file_name_longformat + ".png"):
                img = cv2.imread(self.input + os.sep + "frames" + os.sep + folder_name + os.sep + file_name_longformat + ".png")
            else:
                # reached the end of video
                self.__last_img_in_input_index = -1

                # recursive call (next repetition)
                if self.repetitions > 1:
                    self.__last_returned_time_in_original_video = 0.0
                    return self.get_next(sample_only=sample_only, t=t)

            if img is None:
                return None, None, None, None

            # loading motion (if available)
            if os.path.exists(self.input + os.sep + "motion" + os.sep + folder_name + os.sep + file_name + ".bin"):
                with GzipFile(self.input + os.sep + "motion" + os.sep + folder_name + os.sep + file_name + ".bin") as f:
                    of = np.load(f, allow_pickle=True)

            if os.path.exists(self.input + os.sep + "motion" + os.sep + folder_name + os.sep + file_name_longformat + ".bin"):
                with GzipFile(self.input + os.sep + "motion" + os.sep + folder_name + os.sep + file_name_longformat + ".bin") as f:
                    of = np.load(f, allow_pickle=True)

            # loading supervisions (if available)
            sup_file_no_extension = self.input + os.sep + "sup" + os.sep + folder_name + os.sep + file_name
            if os.path.exists(sup_file_no_extension + ".targets.bin"):
                with GzipFile(sup_file_no_extension + ".targets.bin") as f:
                    targets = np.load(f).astype(np.long)
                if os.path.exists(sup_file_no_extension + ".indices.bin"):
                    with GzipFile(sup_file_no_extension + ".indices.bin") as f:
                        indices = np.load(f).astype(np.long)
                    supervisions = (targets, indices)
                else:
                    supervisions = (targets, None)

                    # indices are needed for partially supervised frames!
                    if targets.size < self.w * self.h:
                        raise ValueError("Missing supervision indices: " + sup_file_no_extension + ".indices.bin")

        # getting a new frame (folder with image files)
        elif self.input_type == InputType.IMAGE_FOLDER:
            if f < len(self.__image_folder_files):
                if self.shuffle:
                    f = self.__shuffled_order[f]

                file_name = self.__image_folder_files[f]
                img = None if skip_if_possible else cv2.imread(file_name)
            else:
                # reached the end of video
                self.__last_img_in_input_index = -1

                # recursive call (next repetition)
                if self.repetitions > 1:
                    self.__last_returned_time_in_original_video = 0.0
                    return self.get_next(sample_only=sample_only, t=t)

            if img is None:
                return None, None, None, None

        # getting a new frame (list of arrays)
        elif self.input_type == InputType.ARRAYS:
            if f < len(self.input["frames"]):
                if self.shuffle:
                    f = self.__shuffled_order[f]

                img = self.input["frames"][f]
                if "motion" in self.input:
                    of = self.input["motion"][f]
                if "sup" in self.input:
                    supervisions = self.input["sup"][f]
            else:
                # reached the end of video
                self.__last_img_in_input_index = -1

                # recursive call (next repetition)
                if self.repetitions > 1:
                    self.__last_returned_time_in_original_video = 0.0
                    return self.get_next(sample_only=sample_only, t=t)

            if img is None:
                return None, None, None, None

        # getting a new frame (unity)
        elif self.input_type == InputType.SAILENV or self.input_type == InputType.TDW:
            try:
                frame_dict = self.__unity_agent.get_frame()
            except ConnectionError:
                self.__unity_agent.delete()
                self.__unity_agent = None
                return None, None, None, None

            #img = cv2.cvtColor((255.0 * frame_dict["main"]).astype(np.uint8), cv2.COLOR_RGB2BGR)
            img = frame_dict["main"][:,:,[2,1,0]]
            of = frame_dict["flow"]

            train_labels = frame_dict["category"]

            train_labels = train_labels.flatten()
            if train_labels is not None:
                indices = np.where(train_labels < 255)[0]
                targets = train_labels[indices]
                indices = indices.astype(np.long)
                targets = targets.astype(np.long)

                supervisions = (targets, indices)
            else:
                supervisions = None

        if self.__rotation_code is not None:
            img = cv2.rotate(img, self.__rotation_code)

            # custom transformation
        if self.frame_op is not None:
            img = self.frame_op(img)
            if of is not None:
                raise ValueError("Cannot return motion information if the frame is transformed!")
            if of is not None:
                raise ValueError("Cannot return supervision information if the frame is transformed!")
            of = None  # transforming motion information is not a good idea
            supervisions = None  # transforming annotated information is not a good idea

        # rescaling
        if self.w != self.w_orig or self.h != self.h_orig:
            img = cv2.resize(img, (self.w, self.h))
            if of is not None:
                raise ValueError("Cannot return motion information if the frame is rescaled!")
            if of is not None:
                raise ValueError("Cannot return supervision information if the frame is rescaled!")
            of = None  # rescaling motion information is not a good idea
            supervisions = None  # rescaling annotated information is not a good idea

        # converting to gray scale
        if self.force_gray and self.c_orig > 1:
            img = np.reshape(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (self.h, self.w, 1))

        # getting the FOA coordinates, if available
        if self.foa is not None and t is None:
            foa = self.foa[self.__last_returned_frame_number, :]

        # moving on the time and frame indices (if not sampling only)
        if not sample_only:
            if next_time is None:
                next_time = self.__last_returned_time + (1.0 / self.fps)  # TODO
            if next_time_in_original_video is None:
                next_time_in_original_video = self.__last_returned_time_in_original_video + (1.0 / self.fps)  # TODO

            self.__last_returned_frame_number = self.__last_returned_frame_number + 1
            self.__last_img_in_input_index = self.__last_img_in_input_index + 1
            self.__last_returned_time = next_time
            self.__last_returned_time_in_original_video = next_time_in_original_video

        return img, of, supervisions, foa

    def set_options(self, w=None, h=None, fps=None, force_gray=None, repetitions=None, max_frames=None, shuffle=None):
        if w is not None:
            self.w = w
        if h is not None:
            self.h = h
        if fps is not None:
            self.fps = float(fps)
        if force_gray is not None:
            self.force_gray = force_gray
        if max_frames is not None:
            self.max_frames = max_frames
        if repetitions is not None:
            if repetitions != self.repetitions:
                self.repetitions = repetitions
                self.frames = self.__count_frames_brute_force() * self.repetitions
        if shuffle is not None:
            self.shuffle = shuffle

    def close(self):
        if self.__video_capture is not None:
            self.__video_capture.release()
        if self.__unity_agent is not None:
            self.__unity_agent.delete()

    def reset(self):
        if self.input_type == InputType.VIDEO_FILE and self.__video_capture is not None:
            self.__video_capture.release()
            self.__video_capture = None
        self.__last_returned_frame_number = 0
        self.__last_img_in_input_index = -1
        self.__last_returned_time = 0.0
        self.__last_returned_time_in_original_video = 0.0

    def set_last_frame_number(self, frame_number):
        if self.input_type != InputType.WEB_CAM and self.input_type != InputType.SAILENV and self.input_type != InputType.TDW:
            self.reset()
            while self.__last_returned_frame_number < frame_number:
                img, _, _, _ = self.get_next()
                if img is None:
                    raise ValueError("Unable to seek to frame number " + str(frame_number))
        else:
            raise ValueError("Unable to seek to frame number " + str(frame_number))

    def set_last_frame_time(self, time_in_seconds):
        if self.input_type != InputType.WEB_CAM and self.input_type != InputType.SAILENV and self.input_type != InputType.TDW:
            self.reset()
            while self.__last_returned_time < time_in_seconds:
                img, _, _, _ = self.get_next()
                if img is None:
                    raise ValueError("Unable to seek to time " + str(time_in_seconds) + " sec.")
        else:
            raise ValueError("Unable to seek to time " + str(time_in_seconds) + " sec.")

    def get_last_frame_number(self):
        return self.__last_returned_frame_number

    def set_last_frame_number(self, x):
        self.__last_returned_frame_number = x

    def get_last_frame_time(self):
        return self.__last_returned_time

    def get_unity_agent(self):
        if self.input_type != InputType.SAILENV and self.input_type != InputType.TDW:
            raise RuntimeError("Cannot get unity agent if input_type is not UNITY")

        return self.__unity_agent

    @property
    def effective_video_frames(self):
        if self.input_type == InputType.WEB_CAM or self.input_type == InputType.SAILENV or self.input_type == InputType.TDW:
            return self.frames

        return self.frames // self.repetitions

    @property
    def current_repetition(self):
        if self.input_type == InputType.WEB_CAM or self.input_type == InputType.SAILENV or self.input_type == InputType.TDW:
            return 1

        return math.ceil(self.__last_returned_frame_number / self.effective_video_frames)

    @staticmethod
    def __get_input_features(input_element):

        # checking
        if input_element is None:
            raise ValueError("Invalid input element (None).")

        # determine the type of input: video, folder, or device (web-cam, up to 10 devices are supported)
        input_ = None
        input_type = None
        readable_input = None

        # test regex for virtual environment
        venv_match = re.match('((?P<env>tdw|sailenv)://)?(?P<url>localhost|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\:(?P<port>\d{1,5})', input_element)
        if isinstance(input_element, str) and os.path.isfile(input_element) and input_element[-4:] != '.npz':
            input_ = os.path.abspath(input_element)
            input_type = InputType.VIDEO_FILE
            readable_input = input_
        elif isinstance(input_element, str) and os.path.isdir(input_element):
            input_ = os.path.abspath(input_element)
            if input_.endswith(os.sep):
                input_ = input_[:-1]
            if os.path.isdir(input_element + os.sep + "frames" + os.sep + "00000001"):
                input_type = InputType.OUT_STREAM_FOLDER
            else:
                input_type = InputType.IMAGE_FOLDER
            readable_input = input_
        elif isinstance(input_element, str) and input_element == "0" or input_element == "1" or input_element == "2":
            input_ = int(input_element)
            input_type = InputType.WEB_CAM
            readable_input = "device_" + input_element
        elif isinstance(input_element, str) and os.path.isfile(input_element) and input_element[-4:] == '.npz':
            loaded_data = np.load(input_element)
            if isinstance(loaded_data, np.lib.npyio.NpzFile):
                if 'frames' in loaded_data:
                    input_ = {"frames": loaded_data['frames']}
                    if 'motion' in loaded_data:
                        input_["motion"] = loaded_data['motion']
                    if 'fps' in loaded_data:
                        input_["fps"] = loaded_data['fps'][0]
                input_type = InputType.ARRAYS
                readable_input = os.path.abspath(input_element)
            else:
                input_ = None
                readable_input = None
        elif isinstance(input_element, dict):
            input_ = input_element
            input_type = InputType.ARRAYS
            readable_input = "arrays"
        elif isinstance(input_element, str) and \
                venv_match is not None:

            # get the groups from regex_match
            env = venv_match.group('env')
            url = venv_match.group('url')
            port = int(venv_match.group('port'))

            if env is None or env == 'sailenv':
                input_type = InputType.SAILENV
            elif env == 'tdw':
                input_type = InputType.TDW
            else:
                raise ValueError("Invalid environment " + env) # should never happen
            
            readable_input = url + ":" + str(port)
            input_ = [url, str(port)]

        if input_ is None or input_type is None:
            raise ValueError("Invalid/Unsupported input element")

        return input_, input_type, readable_input

    def __getinfo(self):

        # checking/reading FOA file, if any
        if self.foa is not None:
            if not os.path.exists(self.foa):
                raise IOError("Cannot find the specified FOA file: ", self.foa)
            else:
                self.foa = np.loadtxt(self.foa, delimiter=",")

        # video or web cam
        if self.input_type == InputType.VIDEO_FILE or self.input_type == InputType.WEB_CAM:
            self.__check_video_rotation(self.input)

            video = cv2.VideoCapture(self.input)

            if video.isOpened():
                fps = video.get(cv2.CAP_PROP_FPS)  # float
                frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
                self.length_in_seconds_orig = video.get(cv2.CAP_PROP_POS_MSEC)
                video.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.0)

                if self.input_type == InputType.VIDEO_FILE:
                    frames = int(frames)
                else:
                    frames = sys.maxsize  # dummy

                ret_val, img = video.read()

                if ret_val:
                    h, w, c = img.shape
                    w = int(w)
                    h = int(h)
                    c = int(c)
                else:
                    raise IOError("Error while trying to grab a frame from: ", self.readable_input)
            else:
                raise IOError("Cannot open: ", self.readable_input)

            self.w_orig = w
            self.h_orig = h
            self.c_orig = c
            self.frames_orig = frames if self.input_type == InputType.VIDEO_FILE else sys.maxsize
            self.fps_orig = float(fps)

            # fixing
            if self.w == -1 and self.h == -1:
                self.w = self.w_orig
                self.h = self.h_orig
            if self.fps <= 0:
                self.fps = self.fps_orig
            self.c = 1 if self.force_gray else self.c_orig

            if self.input_type == InputType.WEB_CAM or self.input_type == InputType.SAILENV or self.input_type == InputType.TDW:
                self.frames = sys.maxsize
            else:
                self.frames = self.__count_frames_brute_force() * self.repetitions

            if self.frames_orig <= 0:
                raise ValueError("Invalid frame count: " + str(self.frames_orig))
            if self.fps_orig <= 0:
                raise ValueError("Invalid FPS count: " + str(self.fps_orig))
            if self.w_orig <= 0 or self.h_orig <= 0:
                raise ValueError("Invalid resolution: " + str(self.w_orig) + "x" + str(self.h_orig))

            if self.shuffle:
                raise ValueError("The option 'shuffle' is invalid when using a video file/device")

        # folder(s) of images created by an output stream
        elif self.input_type == InputType.OUT_STREAM_FOLDER:
            frames_count = None
            if os.path.exists(self.input + os.sep + "sup" + os.sep + "map.json"):
                ff = open(self.input + os.sep + "sup" + os.sep + "map.json")
                self.sup_map = json.load(ff)
                ff.close()

            first_file = ''
            dirs = glob(self.input + os.sep + "frames" + os.sep + "*" + os.sep)

            if dirs is not None and len(dirs) > 0:
                dirs.sort()

                n = len(dirs) - 2  # discarding '.' and '..'
                i = 1

                for d in dirs:
                    if not os.path.isdir(d):
                        continue
                    d = os.path.basename(os.path.dirname(d))
                    if d == '.' or d == '..':
                        continue

                    folder_name = format(i, '08d')
                    if folder_name != d:
                        raise ValueError("Invalid/unexpected folder: " + self.input + os.sep + "frames" + os.sep + d)

                    files = glob(self.input + os.sep + "frames" + os.sep + d + os.sep + "*.png")
                    files.sort()
                    j = 1

                    if i < n and len(files) != self.__files_per_folder:
                        raise ValueError("Invalid/unexpected number of files in: "
                                         + self.input + os.sep + d)

                    for f in files:
                        f = os.path.basename(f)
                        if format(j, '03d') + ".png" != f and format(j, '05d') + ".png" != f:
                            raise ValueError("Invalid/unexpected file '" + f + "' in: "
                                             + self.input + os.sep + d)
                        j = j + 1

                    if len(first_file) == 0:
                        files.sort()
                        first_file = files[0]
                        self.frames_orig = 0

                    self.frames_orig = self.frames_orig + len(files)

                    i = i + 1

                img = cv2.imread(first_file)
                h, w, c = img.shape

                self.w_orig = int(w)
                self.h_orig = int(h)
                self.c_orig = int(c)
                self.fps_orig = -1.0

                if self.frames_orig <= 0:
                    raise ValueError("Invalid frame count: " + str(self.frames_orig))
                if self.w_orig <= 0 or self.h_orig <= 0:
                    raise ValueError("Invalid resolution: " + str(self.w_orig) + "x" + str(self.h_orig))

                try:
                    ff = open(self.input + os.sep + "frames" + os.sep + "fps.json")
                    opts = json.load(ff)
                    self.fps_orig = float(opts['fps'])
                    if 'frames_count' in opts:
                        frames_count = int(opts['frames_count'])
                    ff.close()
                except (ValueError, IOError):
                    raise IOError("FPS file is missing/unreadable/badly-formatted!: "
                                  + self.input + os.sep + "frames" + os.sep + "fps.json")

                self.length_in_seconds_orig = float(self.frames_orig) / self.fps_orig

                # fixing
                if self.w == -1 and self.h == -1:
                    self.w = self.w_orig
                    self.h = self.h_orig
                if self.fps <= 0:
                    self.fps = self.fps_orig
                self.c = 1 if self.force_gray else self.c_orig

                if frames_count is None:
                    frames_count = self.__count_frames_brute_force()

                if self.shuffle:
                    orders = [None] * self.repetitions
                    for r in range(0, self.repetitions):
                        orders[r] = np.random.permutation(frames_count)
                    self.__shuffled_order = np.concatenate(orders)

                self.frames = frames_count * self.repetitions

            else:
                raise ValueError("No frames in: " + self.input + os.sep + "frames" + os.sep)

        # folder of images
        elif self.input_type == InputType.IMAGE_FOLDER:
            files = glob(self.input + os.sep + "*.png")
            if len(files) == 0:
                files = glob(self.input + os.sep + "*.jpg")
            if len(files) == 0:
                files = glob(self.input + os.sep + "*.jpeg")
            if len(files) == 0:
                files = glob(self.input + os.sep + "*.PNG")
            if len(files) == 0:
                files = glob(self.input + os.sep + "*.JPG")
            if len(files) == 0:
                files = glob(self.input + os.sep + "*.JPEG")
            files.sort()

            self.__image_folder_files = files
            self.frames_orig = len(files)

            if self.frames_orig == 0:
                raise ValueError("No (supported) frames found in: " + self.input + os.sep)

            img = cv2.imread(self.__image_folder_files[0])
            h, w, c = img.shape

            self.w_orig = int(w)
            self.h_orig = int(h)
            self.c_orig = int(c)
            self.fps_orig = 25.0
            self.length_in_seconds_orig = float(self.frames_orig) / self.fps_orig

            # fixing
            if self.w == -1 and self.h == -1:
                self.w = self.w_orig
                self.h = self.h_orig
            if self.fps <= 0:
                self.fps = self.fps_orig
            self.c = 1 if self.force_gray else self.c_orig

            if self.frames_orig <= 0:
                raise ValueError("Invalid frame count: " + str(self.frames_orig))
            if self.w_orig <= 0 or self.h_orig <= 0:
                raise ValueError("Invalid resolution: " + str(self.w_orig) + "x" + str(self.h_orig))

            frames_brute_force = self.__count_frames_brute_force()

            if self.shuffle:
                orders = [None] * self.repetitions
                for r in range(0, self.repetitions):
                    orders[r] = np.random.permutation(frames_brute_force)
                self.__shuffled_order = np.concatenate(orders)

            self.frames = frames_brute_force * self.repetitions

        elif self.input_type == InputType.ARRAYS:
            if len(self.input) == 0:
                raise ValueError("No frames found in the input list")

            self.frames_orig = len(self.input["frames"])
            self.fps_orig = 25.0
            if "fps" in self.input:
                self.fps_orig = float(self.input["fps"])

            img = self.input["frames"][0]
            h, w, c = img.shape

            self.w_orig = int(w)
            self.h_orig = int(h)
            self.c_orig = int(c)
            self.length_in_seconds_orig = float(self.frames_orig) / self.fps_orig

            # fixing
            if self.w == -1 and self.h == -1:
                self.w = self.w_orig
                self.h = self.h_orig
            if self.fps <= 0:
                self.fps = self.fps_orig
            self.c = 1 if self.force_gray else self.c_orig

            if self.frames_orig <= 0:
                raise ValueError("Invalid frame count: " + str(self.frames_orig))
            if self.w_orig <= 0 or self.h_orig <= 0:
                raise ValueError("Invalid resolution: " + str(self.w_orig) + "x" + str(self.h_orig))

            frames_brute_force = self.__count_frames_brute_force()

            if self.shuffle:
                orders = [None] * self.repetitions
                for r in range(0, self.repetitions):
                    orders[r] = np.random.permutation(frames_brute_force)
                self.__shuffled_order = np.concatenate(orders)

            self.frames = frames_brute_force * self.repetitions

        elif self.input_type == InputType.SAILENV:

            self.__unity_agent = SailenvAgent(host=self.input[0],
                                       port=int(self.input[1]),
                                       width=self.w,
                                       height=self.h,
                                       depth_frame_active=self.unity_settings["depth_frame_active"],
                                       flow_frame_active=self.unity_settings["flow_frame_active"],
                                       object_frame_active=self.unity_settings["object_frame_active"],
                                       main_frame_active=self.unity_settings["main_frame_active"],
                                       category_frame_active=self.unity_settings["category_frame_active"],
                                       use_gzip=self.unity_settings["use_gzip"])

            self.w_orig = self.w
            self.h_orig = self.h
            self.c_orig = 3  # dummy
            self.frames_orig = sys.maxsize
            self.fps_orig = float(25)  # dummy

            # fixing
            if self.w == -1 and self.h == -1:
                self.w = self.w_orig
                self.h = self.h_orig
            if self.fps <= 0:
                self.fps = self.fps_orig
            self.c = 1 if self.force_gray else self.c_orig

            try:
                self.__unity_agent.register()

                print(f"Available scenes: {self.__unity_agent.scenes}")

                scene_setting = self.unity_settings['scene']
                if isinstance(scene_setting, str):
                    scene = scene_setting
                elif isinstance(scene_setting, int):
                    scene = self.__unity_agent.scenes[self.unity_settings["scene"]]
                else:
                    raise ValueError("Invalid scene setting")

                print(f"Changing scene to {scene}")
                self.__unity_agent.change_scene(scene)
            except ConnectionError:
                self.__unity_agent = None
                raise IOError("Error while trying to communicate with the SAILENV server: ", self.readable_input)

            self.frames = sys.maxsize
            self.sup_map = {v: k for k,v in self.__unity_agent.categories.items()}

            if self.fps_orig <= 0:
                raise ValueError("Invalid FPS count: " + str(self.fps_orig))
            if self.w_orig <= 0 or self.h_orig <= 0:
                raise ValueError("Invalid resolution: " + str(self.w_orig) + "x" + str(self.h_orig))

            if self.shuffle:
                raise ValueError("The option 'shuffle' is invalid when using the UNITY client")
        
        elif self.input_type == InputType.TDW:
            self.__unity_agent = TDWAgent(
                main_frame_active=self.unity_settings["main_frame_active"],
                depth_frame_active=self.unity_settings["depth_frame_active"],
                flow_frame_active=self.unity_settings["flow_frame_active"],
                object_frame_active=self.unity_settings["object_frame_active"],
                category_frame_active=self.unity_settings["category_frame_active"],
                controller=self.unity_settings.get("controller", None), # if not provided, the agent will create its controller
                width=self.w,
                height=self.h,
            )

            self.w_orig = self.w
            self.h_orig = self.h
            self.c_orig = 3  # dummy
            self.frames_orig = sys.maxsize
            self.fps_orig = float(25)  # dummy

            # fixing
            if self.w == -1 and self.h == -1:
                self.w = self.w_orig
                self.h = self.h_orig
            if self.fps <= 0:
                self.fps = self.fps_orig
            self.c = 1 if self.force_gray else self.c_orig

            try:
                self.__unity_agent.register()
                print(f"Available scenes: {self.__unity_agent.scenes}")
                scene_setting = self.unity_settings['scene']
                if isinstance(scene_setting, str):
                    if scene_setting not in self.__unity_agent.scenes:
                        raise ValueError(f"Invalid scene: {scene_setting}")
                    scene = scene_setting
                elif isinstance(scene_setting, int):
                    scene = self.__unity_agent.scenes[self.unity_settings["scene"]]
                else:
                    raise ValueError("Invalid scene:")

                print(f"Changing scene to {scene}")
                self.__unity_agent.change_scene(scene)
            except:
                self.__unity_agent = None

                raise IOError("Error while trying to communicate with the TDW server: ", self.readable_input)

            self.frames = sys.maxsize
            self.sup_map = {} 
            
            # TODO: try to get categories from the TDW server
            # self.sup_map = {v: k for k,v in self.__unity_agent.categories.items()}

            if self.fps_orig <= 0:
                raise ValueError("Invalid FPS count: " + str(self.fps_orig))
            if self.w_orig <= 0 or self.h_orig <= 0:
                raise ValueError("Invalid resolution: " + str(self.w_orig) + "x" + str(self.h_orig))

            if self.shuffle:
                raise ValueError("The option 'shuffle' is invalid when using the UNITY client")

    def __count_frames_brute_force(self):
        backup_max_frames = self.max_frames
        backup_shuffle = self.shuffle
        backup_frames = self.frames
        backup_repetitions = self.repetitions

        self.frames = -1
        self.repetitions = 1
        self.set_options(max_frames=-1, shuffle=False)

        n = 0
        while True:
            img, _, _, _ = self.get_next()
            if img is None:
                break
            else:
                n += 1

        self.reset()
        self.set_options(max_frames=backup_max_frames, shuffle=backup_shuffle)
        self.repetitions = backup_repetitions
        self.frames = backup_frames
        return n

    def __check_video_rotation(self, video_file):
        if self.input_type == InputType.VIDEO_FILE:
            meta_dict = ffmpeg.probe(video_file)

            rotate_code = None
            if 'tags' in meta_dict['streams'][0] and 'rotate' in meta_dict['streams'][0]['tags']:
                if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
                    rotate_code = cv2.ROTATE_90_CLOCKWISE
                elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
                    rotate_code = cv2.ROTATE_180
                elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
                    rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
            self.__rotation_code = rotate_code

    def print_info(self):
        print("[Original Stream]")
        print("- Input:      " + self.readable_input)
        print("- Input Type: " + InputType.readable_type(self.input_type))
        print("- Resolution: " + str(self.w_orig) + "x" + str(self.h_orig))
        print("- Channels:   " + str(self.c_orig))
        print("- FPS:        " + str(self.fps_orig))
        print("- Frames:     " + str(self.frames_orig))
        print("")
        print("[Requested Stream]")
        print("- Resolution: " + str(self.w) + "x" + str(self.h))
        print("- Channels:   " + str(self.c))
        print("- FPS:        " + str(self.fps))
        print("- Frames:     " + str(self.frames) + " (involving " + str(self.repetitions)
              + " repetitions of the video)")
        print("- Max Frames: " + (str(self.max_frames) if self.max_frames > 0 else "Not specified"))
