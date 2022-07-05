import copy


class Worker:
    """The abstract class that describes a generic computational unit that is responsible of processing a given frame.

    If you want to implement your own worker, then you have to extend this class and implement the abstract methods.
    This is because the worker is manipulated by an external class (lve.VProcessor) that will call its methods.

    Each worker can handle commands that are provided from the outside (for example, a new supervision).

    Each worker describes the output data that it can produce (get_output), and it can add new output whenever it
    becomes available (add_output). The output data is intended to be store in an lve.OutputStream, even if the worker
    has no direct access to it.

    Each worker inherits basic functionalities in terms of handling a supervision.
    In particular, it stores the map that transforms the class name in a numerical index (get_target), and vice-versa
    (get_class_name), and it can store the number of received supervised examples (increment_supervision_count).

    See the list of methods for further details.

    Attributes:
        w (int): frame width.
        h (int): frame height.
        c (int): number of channels.
        fps (float): frame rate.
        options (dict): dictionary of options that customize this computational unit and its internal modules.
        frame_is_gray_scale (bool): flag that indicates whether the number of input channels is one.
        heavy_output_data_needed (bool): flag that indicates whether this worker should also produce those output
            elements that might require some additional computations to be produced (for example, these outputs might
            be needed when a visualization tool is interacting with this worker, while they are not needed when no
            visualization is happening).

    Args:
        w (int): frame width.
        h (int): frame height.
        c (int): number of channels.
        fps (float): frame rate.
        options (dict): dictionary of options that customize this computational unit and its internal modules.
    """
    def __init__(self, w, h, c, fps, options):
        self.w = w
        self.h = h
        self.c = c
        self.fps = fps
        self.frame_is_gray_scale = self.c == 1
        self.options = options
        self.heavy_output_data_needed = True

        self.__command_buffer = {}
        self.__command_handlers = {}
        self.__batched_output_data = []
        self.__sup_map = {}
        self.__sup_count = {}
        self.__sup_map_inv = {}
        self.__sup_count_checkpoint = None

    def process_frame(self, frame, of=None, supervisions=None, foa=None):
        """The main method of the worker: it processes the given frame and it stores the produced output.

        The input arguments are lists, since the data can be provided in batches.
        By default, batch size is 1, so there will be lists with only 1 element.

        Args:
            frame (list of np.array, unsigned int8): input frames (each frame is h x w x c).
            of (list of np.array, float32, optional): precomputed motion vectors (each is h x w x 2).
            supervisions (list of whatever): supervisions or other frame-related annotations.
            foa (list of mono-dim np.array, float32, optional): x,y,vx,vy,saccade-flag of the focus of attention.
        """
        raise NotImplementedError("To be implemented!")

    def update_model_parameters(self):
        """This method updates the internal model parameters.
        """
        raise NotImplementedError("To be implemented!")

    def load(self, model_folder):
        """Load the model parameters from the model folder.

        Args:
            model_folder (str): path of the model folder.
        """
        raise NotImplementedError("To be implemented!")

    def save(self, model_folder):
        """Save the model parameters to disk.

        Remember to create the model folder, if needed.

        Args:
            model_folder (str): path of the model folder.
        """
        raise NotImplementedError("To be implemented!")

    def get_output_types(self):
        """Return all the types of the elements that this worker will output, to be saved in an lve.OutputStream.

        Returns:
            Dictionary with all the details of the elements that this worker will output, see the following example.
            For each output element, the type of such element must be specified (see lve.OutputType), and also if such
            output is intended to be stored in a separate file per frame (per_frame = True) or not (per_frame = False).

            Example:
                {name_of_output_1: {'data_type': lve.OutputType.SELECT_TYPE, 'per_frame': True/False},
                name_of_output_2: {'data_type': lve.OutputType.SELECT_TYPE, 'per_frame': True/False}}
        """
        raise NotImplementedError("To be implemented!")

    def print_info(self):
        """This print-function will be called from outside right after having created the worker."""
        raise NotImplementedError("To be implemented!")

    def add_outputs(self, outputs, batch_index=0):
        """Add the model outputs to the queue of available outputs.

        Args:
            outputs (dict): dictionary of pairs (output_element_name, element) of the output elements to be saved.
            batch_index (int, optional): when storing batched output data, the batch index is specified (default: 0).
        """
        for element_name, element in outputs.items():
            self.add_output(element_name, element, batch_index)

    def add_output(self, element_name, element, batch_index=0):
        """Add the model outputs to the queue of available outputs.

        Args:
            element_name (str): name of the output element to be saved.
            element (whatever): output element to be saved.
            batch_index (int, optional): when storing batched output data, the batch index is specified (default: 0).
        """
        while len(self.__batched_output_data) <= batch_index:
            self.__batched_output_data.append({})
        output_data = self.__batched_output_data[batch_index]
        output_data[element_name] = element

    def get_output(self, batch_index=0):
        """Return the dictionary of the output elements that are available.

        When an output element is returned, it is also removed from the dictionary.

        Args:
            batch_index (int, optional): when getting batched output data, the batch index is specified (default: 0).

        Returns:
            Dictionary with all the the output elements that are available.
        """
        if len(self.__batched_output_data) >= batch_index + 1:
            output_data = self.__batched_output_data[batch_index]
            output_data_copy = output_data.copy()  # shallow copy
            for element_name in output_data:
                output_data[element_name] = None
            return output_data_copy
        else:
            return None

    def set_h_w(self, h, w):
        self.h = h
        self.w = w

    def set_heavy_output_data_needed(self, flag):
        """Set heavy_output_data_needed, to indicate if also those heavy-to-be-computed outputs are now needed.

        Args:
            flag (bool): flat that indicates the value to be set to heavy_output_data_needed.
        """
        self.heavy_output_data_needed = flag

    def register_commands(self, command_names_handlers_dic):
        """Register new commands that can be handled by this worker.

        Args:
            command_names_handlers_dic (dict): dictionary with the names (str) of the commands and their handlers.
        """
        for command_name, command_handler in command_names_handlers_dic.items():
            self.register_command(command_name, command_handler)

    def register_command(self, command_name, command_handler):
        """Register a new command that can be handled by this worker.

        Args:
            command_name (str): the name of the command.
            command_handler (fcn): the function that handles the command.
        """
        if command_name in self.__command_handlers:
            raise ValueError("Command was already registered! (" + command_name + ")")
        self.__command_buffer[command_name] = []
        self.__command_handlers[command_name] = command_handler

    def handle_commands(self, batch_index=0):
        """Check for buffered commands and handle them.

        The commands that are sent to the worker are buffered first, then they are all handled when this method is
        called.

        Args:
            batch_index (int, optional): when handling batched data, the batch index can be specified (default: 0).

        Returns:
            Boolean flag to indicate if at least one command was handled.
        """
        handled_something = False
        for command_name, command_values in self.__command_buffer.items():
            for command_value in command_values:
                self.__command_handlers[command_name](command_value, batch_index=batch_index)
                handled_something = True
            self.__command_buffer[command_name] = []
        return handled_something

    def send_command(self, command_name, command_value):
        """Send a command to this working unit.

        Args:
            command_name (str): the name of the command.
            command_value (whatever): the value associated to the command.
        """
        if command_name not in self.__command_handlers:
            raise ValueError("Unknown (not registered) command received: " + command_name)
        self.__command_buffer[command_name].append(command_value)

    def get_supervision_map(self):
        """Get the map class name -> class index (target).

        Returns:
            Dictionary class name -> class index.
        """
        return self.__sup_map

    def get_supervision_count(self):
        """Get the map class name -> number of received supervisions.

        Returns:
            Dictionary class name -> number of received supervisions.
        """
        return self.__sup_count

    def get_min_supervision_per_class_count(self):
        """Get the number of supervisions of the class that received the smallest number of supervisions.

        Returns:
            Min number of received supervisions.
        """
        min_sup = None
        for v in self.__sup_count.values():
            if min_sup is None:
                min_sup = v
            else:
                if v < min_sup:
                    min_sup = v
        return min_sup if min_sup is not None else 0

    def get_class_name(self, target):
        """Get the class name associated to the provided index (target).

        Args:
            target (int): the class index.

        Returns:
            Class name (str) or None.
        """
        return self.__sup_map_inv[target] if target in self.__sup_map_inv else None

    def get_target(self, class_name, max_classes):
        """Get the class index (target) associated to the provided class name.

        This function also adds the class name (if never seen before) to the internal map that relates each class name
        to its numerical index.

        Args:
            class_name (str): name of the class.
            max_classes (int): maximum number of classes that can be managed by the worker.

        Returns:
            The index (target) of the provided class name, or None if the class is new and the worker cannot handle
            new classes.
            A boolean flag that indicates whether the class name was never-seen-before and added to the internal list.
        """
        if class_name in self.__sup_map:
            target = self.__sup_map[class_name]
            new_class_added = False
        else:
            m = -1
            for t in self.__sup_map.values():
                m = max(m, t)
            target = m + 1

            if target >= max_classes:
                return None, None

            self.__sup_map[class_name] = target
            self.__sup_count[class_name] = 0
            self.__sup_map_inv[target] = class_name
            new_class_added = True
        return target, new_class_added

    def increment_supervision_count(self, target, num_sup=1):
        """Increment the internal supervision counter for a given class index (target).

        Args:
            target (int): the class index.
            num_sup (int, optional): the number of supervisions (default: 1).
        """
        class_name = self.get_class_name(target)
        count = self.__sup_count[class_name]
        count += num_sup
        self.__sup_count[class_name] = count

    def augment_supervision_map(self, sup_map, max_classes, counts=None):
        """Augment the internal map class name -> class index (target) by merging it with the provided map.

        Args:
            sup_map (dict): map from class name to class index that you want to add to the already existing map.
            max_classes (int): maximum number of classes that can be managed by the worker.
            counts (int, optional): map from class name to numer of supervisions to add to the already existing counts
                (default: None)
        """
        if sup_map is None:
            return

        for k in sup_map.keys():
            if k not in self.__sup_map:
                if len(self.__sup_map) < max_classes:
                    self.__sup_map[k] = sup_map[k]
                else:
                    raise ValueError("Cannot further extend the supervision map, too many classes! "
                                     "(max=" + str(max_classes) + ")")
            else:
                if self.__sup_map[k] != sup_map[k]:
                    raise ValueError("Mismatching class maps! Expected class: "
                                     + k + "->" + str(self.__sup_map[k]) + ", Found class: "
                                     + k + "->" + str(sup_map[k]))
        self.__sup_map_inv = {}
        for class_name in self.__sup_map.keys(): self.__sup_map_inv[self.__sup_map[class_name]] = class_name

        if counts is not None:
            for k in sup_map.keys():
                if k not in self.__sup_count:
                    self.__sup_count[k] = counts[k]
                else:
                    self.__sup_count[k] += counts[k]
        else:
            for k in self.__sup_map.keys():
                if k not in self.__sup_count:
                    self.__sup_count[k] = 0

    def create_supervision_count_checkpoint(self):
        self.__sup_count_checkpoint = copy.deepcopy(self.__sup_count)

    def restore_supervision_count_checkpoint(self):
        self.__sup_count.update(self.__sup_count_checkpoint)
