import os
import cv2
import shutil
import numpy as np
from gzip import GzipFile
import json
from enum import Enum
from torch.utils.tensorboard import SummaryWriter


class OutputType(Enum):
    BINARY = 0
    JSON = 1
    TEXT = 2
    IMAGE = 3
    PRIVATE = 4

    @staticmethod
    def readable_type(output_type):
        if output_type == OutputType.BINARY:
            output_type = "BINARY"
        elif output_type == OutputType.JSON:
            output_type = "JSON"
        elif output_type == OutputType.TEXT:
            output_type = "TEXT"
        elif output_type == OutputType.IMAGE:
            output_type = "IMAGE"
        elif output_type == OutputType.PRIVATE:
            output_type = "PRIVATE"
        return output_type

    @staticmethod
    def from_string(string_output_type):
        if string_output_type == "BINARY":
            output_type = OutputType.BINARY
        elif string_output_type == "JSON":
            output_type = OutputType.JSON
        elif string_output_type == "TEXT":
            output_type = OutputType.TEXT
        elif string_output_type == "IMAGE":
            output_type = OutputType.IMAGE
        elif string_output_type == "PRIVATE":
            output_type = OutputType.PRIVATE
        return output_type


class OutputStream:

    def __init__(self, folder, fps, virtual_save=False, save_per_frame_data=True, purge_existing_data=True,
                 tensorboard=True, gzip_bin=True, save_interval=1):
        self.folder = os.path.abspath(folder)
        if self.folder.endswith(os.sep):
            self.folder = self.folder[:-1]
            
        self.__files_per_folder = 100
        self.__gzip_bin = gzip_bin
        self.__last_saved_frame_number = 0
        self.__output_elements = {}
        self.__purged_existing_data = purge_existing_data
        self.__save_interval = save_interval

        if purge_existing_data and os.path.exists(self.folder):
            shutil.rmtree(self.folder)

        if not (virtual_save and not tensorboard):
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)

        self.virtual_save = virtual_save
        self.save_per_frame_data = save_per_frame_data
        self.tensorboard = SummaryWriter(log_dir=self.folder + os.sep + "tensorboard") if tensorboard else None

        self.register_output_element("frames", data_type=OutputType.IMAGE, per_frame=True)
        self.register_output_element("frames.fps", data_type=OutputType.JSON, per_frame=False)
        if not self.virtual_save and self.save_per_frame_data:
            self.save_element("frames.fps", {'fps': float(fps)})

    def close(self):
        if self.tensorboard is not None:
            self.tensorboard.close()

    def register_output_elements(self, element_to_attrib_dict):
        for element_name in element_to_attrib_dict:
            if not element_name.endswith("__header"):
                attrib = element_to_attrib_dict[element_name]
                self.register_output_element(element_name, attrib['data_type'], attrib['per_frame'])

        # repeating the same cycle again to register headers (this must be done when the cycle above has ended!)
        for element_name in element_to_attrib_dict:
            if element_name.endswith("__header"):
                attrib = element_to_attrib_dict[element_name]
                self.register_output_element(element_name, attrib, False)

    def register_output_element(self, element_name, data_type=OutputType.BINARY, per_frame=True):
        if element_name in self.__output_elements:
            raise ValueError("Already existing output element: " + str(element_name))

        self.__output_elements[element_name] = {'data_type': data_type, 'data': [], 'per_frame': per_frame}

        if not self.virtual_save:
            if element_name.endswith("__header"):
                element_name_real = element_name[0:-8]
                if self.get_data_type_of(element_name_real) == OutputType.TEXT and \
                        (isinstance(data_type, list) or isinstance(data_type, str)):
                    self.save_element(element_name_real, data_type,
                                      include_frame_number_when_saving_non_per_frame_text=False)
                else:
                    raise ValueError("The special suffix __header is used with a non-text output element or "
                                     "without providing the data that compose the header (" + element_name_real + ")")
            else:
                if self.save_per_frame_data or per_frame == False:
                    p = element_name.find(".")
                    if p > 0:
                        if not os.path.exists(self.folder + os.sep + element_name[0:p]):
                            os.makedirs(self.folder + os.sep + element_name[0:p])
                    else:
                        if not os.path.exists(self.folder + os.sep + element_name):
                            os.makedirs(self.folder + os.sep + element_name)

        if element_name.endswith("__header"):
            self.__output_elements.pop(element_name)

    def clear_data_of_output_elements(self):
        for el in self.__output_elements.values():
            el["data"] = None

    def get_output_elements(self, data_type=None):
        if data_type is None:
            return self.__output_elements
        else:
            ret = {}
            for el in self.__output_elements:
                if self.__output_elements[el]['data_type'] == data_type:
                    ret[el] = self.__output_elements[el]
            return ret

    def get_data_type_of(self, element):
        return self.__output_elements[element]['data_type']

    def get_max_number_of_files_per_subfolder(self):
        return self.__files_per_folder

    def get_data_types(self):
        return {k: OutputType.readable_type(self.__output_elements[k]['data_type']) for k in self.__output_elements}

    def save_elements(self, elements, prev_frame=False):
        if elements is not None:
            for element_name in elements:
                self.save_element(element_name, elements[element_name], prev_frame=prev_frame)

    def save_element(self, element_name, element,
                     include_frame_number_when_saving_non_per_frame_text=True, prev_frame=False):

        if element is None:
            return

        if not prev_frame:
            frame_number = self.__last_saved_frame_number
        else:
            frame_number = self.__last_saved_frame_number - 1

        # saving tensorboard data (prefix "tb.")
        if element_name[0:3] == "tb.":
            if self.tensorboard is not None:
                if not isinstance(element, dict):
                    self.tensorboard.add_scalar(element_name[3:], element, frame_number + 1)
                else:
                    self.tensorboard.add_scalars(element_name[3:], element, frame_number + 1)
            return

        # checking if element is registered
        if element_name not in self.__output_elements:
            raise ValueError("Unknown output element " + element_name +
                             " (remember to register outputs before starting to save them!)")

        # saving references to memory first
        el = self.__output_elements[element_name]
        el['data'] = element
        data_type = el['data_type']
        per_frame = el['per_frame']

        # checking coherence with data types
        if data_type == OutputType.PRIVATE:
            return
        if isinstance(element, np.ndarray) and not (data_type == OutputType.BINARY or data_type == OutputType.IMAGE):
            raise ValueError("Numpy array must be saved with data types: OutputType.BINARY or OutputType.IMAGE")
        if isinstance(element, str) and not data_type == OutputType.TEXT:
            raise ValueError("Strings must be saved with data types: OutputType.TEXT")
        if isinstance(element, list) and (not data_type == OutputType.TEXT and not isinstance(element[0], dict)):
            raise ValueError("Lists must be saved with data types: OutputType.TEXT")
        if isinstance(element, dict) and not data_type == OutputType.JSON:
            raise ValueError("Dictionaries must be saved with data types: OutputType.JSON")

        if self.virtual_save:
            return

        if not self.save_per_frame_data and per_frame == True:
            return

        if self.__last_saved_frame_number % self.__save_interval != 0:
            return

        # getting destination file names and paths
        full_file_name, full_folder_name = \
            OutputStream.get_full_file_name_of(element_name,
                                               frame_number if per_frame else None,
                                               data_type,
                                               self.folder,
                                               self.__files_per_folder)

        # creating the internal folders, if needed
        if full_folder_name is not None:
            if not os.path.isdir(full_folder_name):
                os.makedirs(full_folder_name)

        # saving
        if data_type == OutputType.IMAGE:
            cv2.imwrite(full_file_name, element)
        elif data_type == OutputType.BINARY:
            if self.__gzip_bin:
                with GzipFile(full_file_name, 'wb') as file:
                    np.save(file, element)
            else:
                with open(full_file_name, 'wb') as file:
                    np.save(file, element)
        elif data_type == OutputType.JSON:
            f = open(full_file_name, 'w')
            if f is None or not f or f.closed:
                raise IOError("Cannot access: " + full_file_name)
            json.dump(element, f, indent=4)
            f.close()
        elif data_type == OutputType.TEXT:
            if isinstance(element, list):
                element = ','.join(map(str, element))
            elif not isinstance(element, str):
                raise ValueError("Unknown/unsupported output data (str or list expected). Cannot save it.")

            with open(full_file_name, "a") as file:
                if not per_frame and include_frame_number_when_saving_non_per_frame_text:
                    file.write(str(frame_number + 1) + "," + element + "\n")
                else:
                    file.write(element + "\n")
        else:
            raise ValueError("Unknown/unsupported output data type. Cannot save it.")

    @staticmethod
    def get_full_file_name_of(element_name, frame_number, data_type, folder, files_per_folder):
        per_frame = False if frame_number is None else True

        # getting the right folder and file ID
        if per_frame:
            f = frame_number
            n_folder = int(f / files_per_folder) + 1
            n_file = (f + 1) - ((n_folder - 1) * files_per_folder)

            folder_name = format(n_folder, '08d')
            file_name = format(n_file, '03d')

        p = element_name.find(".")
        if p > 0:
            if per_frame:
                file_name = file_name + "." + element_name[p+1:]
                root_folder_name = element_name[0:p]
            else:
                file_name = element_name[p+1:]
                root_folder_name = element_name[0:p]
        else:
            if per_frame:
                root_folder_name = element_name
            else:
                root_folder_name = element_name
                file_name = element_name

        if per_frame:
            full_folder_name = folder + os.sep + root_folder_name + os.sep + folder_name
        else:
            full_folder_name = None

        # final names and their folders
        if per_frame:
            if data_type == OutputType.IMAGE:
                return full_folder_name + os.sep + file_name + ".png", full_folder_name
            elif data_type == OutputType.BINARY:
                return full_folder_name + os.sep + file_name + ".bin", full_folder_name
            elif data_type == OutputType.JSON:
                return full_folder_name + os.sep + file_name + ".json", full_folder_name
            elif data_type == OutputType.TEXT:
                return full_folder_name + os.sep + file_name + ".txt", full_folder_name
            else:
                raise ValueError("Unknown/unsupported output data type.")
        else:
            if data_type == OutputType.IMAGE:
                return folder + os.sep + root_folder_name + os.sep + file_name + ".png", None
            elif data_type == OutputType.BINARY:
                return folder + os.sep + root_folder_name + os.sep + file_name + ".bin", None
            elif data_type == OutputType.JSON:
                return folder + os.sep + root_folder_name + os.sep + file_name + ".json", None
            elif data_type == OutputType.TEXT:
                return folder + os.sep + root_folder_name + os.sep + file_name + ".txt", None
            else:
                raise ValueError("Unknown/unsupported output data type.")

    def save_done(self):
        self.__last_saved_frame_number = self.__last_saved_frame_number + 1  # moving forward
        if self.tensorboard is not None:
            self.tensorboard.flush()

    def set_last_frame_number(self, last_frame_number):
        self.__last_saved_frame_number = last_frame_number

    def get_last_frame_number(self):
        return self.__last_saved_frame_number

    def is_newly_created(self):
        return self.__purged_existing_data

    def is_gzipping_binaries(self):
        return self.__gzip_bin

    def print_info(self):
        print("[Output Stream]")
        print("- Folder:   " + self.folder)
        output_elements = self.get_output_elements()
        s = ""
        i = 0
        for ko in output_elements:
            output_type = OutputType.readable_type(output_elements[ko]['data_type'])
            if i > 0:
                s += ", "
            s += ko + ":" + output_type
            i += 1
        print("- Elements: " + s)
