import threading
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket
import numpy as np
from gzip import GzipFile
import io
import json
import lve


class VisualizationServer:

    def __init__(self,
                 port=8080,
                 html_root=os.path.dirname(os.path.realpath(lve.__file__)) + os.sep + "web",
                 output_folder=None,
                 model_folder=None,
                 v_processor=None):

        assert output_folder is not None and model_folder is not None, "Missing arguments (VisualizationServer)"

        # getting output folder attributes
        if v_processor is None:
            file_name = model_folder + os.sep + "options.json"
            with open(file_name, "r") as f:
                options = json.load(f)
        else:
            options = v_processor.options

        files_per_folder = options["output_folder_files_per_subfolder"]
        gzipped_bin = options["output_folder_gzipped_bin"]

        data_types = {}
        all_stats_json = []

        for element_name in options["output_folder_data_types"]:
            data_types[element_name] = \
                lve.OutputType.from_string(options["output_folder_data_types"][element_name])

        for element_name in data_types:
            if data_types[element_name] == lve.OutputType.JSON:
                all_stats_json.append(element_name)

        output_folder_attributes = {'files_per_folder': files_per_folder,
                                    'gzipped_bin': gzipped_bin,
                                    'data_types': data_types,
                                    'all_stats_json': all_stats_json}

        # creating handler
        handler = make_handler_class(html_root, output_folder, model_folder, output_folder_attributes, v_processor)

        # opening server (only if port >= 0)
        if port >= 0:
            self.server = HTTPServer(('', port), handler)
            self.ip = socket.gethostbyname(socket.gethostname())
            self.port = self.server.server_port
            self.data_root = output_folder
            self.model_root = model_folder
            threading.Thread(target=self.open).start()
        else:
            self.server = None
            self.ip = socket.gethostbyname(socket.gethostname())
            self.port = -1
            self.data_root = output_folder
            self.model_root = model_folder

    def open(self):
        self.server.serve_forever()

    def close(self):
        if self.server is not None:
            self.server.socket.close()
            self.server.shutdown()

    def print_info(self):
        print('[Visualization Server]')
        print('- IP:         ' + str(self.ip))
        print('- Port:       ' + str(self.port))
        print('- Data Root:  ' + self.data_root)
        print('- Model Root: ' + self.model_root)
        print("- URL:        http://" + str(self.ip) + ":" + str(self.port))

def make_handler_class(html_root="web",
                       output_folder="output",
                       model_folder="model",
                       output_folder_attributes=None,
                       v_processor=None):

    class Handler(BaseHTTPRequestHandler, object):

        def __init__(self, *args, **kwargs):
            self.html_root = os.path.abspath(html_root)
            self.output_folder = os.path.abspath(output_folder)
            self.model_folder = os.path.abspath(model_folder)
            self.v_processor = v_processor
            self.data_types = output_folder_attributes["data_types"]
            self.all_stats_json = output_folder_attributes["all_stats_json"]
            self.files_per_folder = output_folder_attributes["files_per_folder"]
            self.gzipped_bin = output_folder_attributes["gzipped_bin"]
            self.path = None

            super(Handler, self).__init__(*args, **kwargs)

        def log_message(self, format_, *args):
            return

        def do_GET(self):
            args = None

            if self.path == "/":
                self.path = "index.html"

            if '?' in self.path:
                self.path, tmp = self.path.split('?', 1)
                args = urllib.parse.parse_qs(tmp)

            try:
                send_static_data = False
                send_binary_data = False

                if self.path.endswith(".html"):
                    mime_type = 'text/html'
                    send_static_data = True
                elif self.path.endswith(".htm"):
                    mime_type = 'text/html'
                    send_static_data = True
                elif self.path.endswith(".jpg"):
                    mime_type = 'image/jpg'
                    send_static_data = True
                elif self.path.endswith(".png"):
                    mime_type = 'image/png'
                    send_static_data = True
                elif self.path.endswith(".gif"):
                    mime_type = 'image/gif'
                    send_static_data = True
                elif self.path.endswith(".ico"):
                    mime_type = 'image/png'
                    send_static_data = True
                elif self.path.endswith(".js"):
                    mime_type = 'application/javascript'
                    send_static_data = True
                elif self.path.endswith(".css"):
                    mime_type = 'text/css'
                    send_static_data = True
                else:
                    mime_type = 'application/octet-stream'
                    send_binary_data = True

                if send_static_data:
                    f = open(self.html_root + os.sep + self.path, 'rb')
                    self.send_response(200)
                    self.send_header('Content-Type', mime_type)
                    self.end_headers()
                    self.wfile.write(f.read())
                    f.close()
                elif send_binary_data:
                    data, is_gzipped = self.__get_data(self.path, args)
                    self.send_response(200)
                    self.send_header('Content-Type', mime_type)
                    self.send_header("Content-Length", len(data))
                    if is_gzipped:
                        self.send_header("Content-Encoding", 'gzip')
                    self.end_headers()
                    self.wfile.write(data)
                return

            except (IOError, OSError):
                try:
                    self.send_error(404, 'File Not Found: %s' % self.path)
                except (ValueError, IOError, OSError):
                    pass

        def __get_data(self, request, args):

            # variables associated with data sent by the web interface
            # (these are all the possible and only data this server will ever get)
            frame_number = None
            sync = False
            opt_name = None
            opt_value = None

            # variables associated with data that will be sent to the web interface
            data = None
            is_gzipped = None

            # getting data sent by the web interface
            if args is not None:

                # frame number
                if 'frame' in args and args['frame'] is not None:
                    if args['frame'][0] is not None:
                        try:
                            frame_number = int(args['frame'][0])
                        except ValueError:
                            frame_number = 1
                        if frame_number <= 0:
                            frame_number = 1

                # sync flag
                if 'sync' in args and args['sync'] is not None:
                    if args['sync'][0] is not None:
                        try:
                            sync = int(args['sync'][0]) == 1
                        except ValueError:
                            sync = False

                # option to be changed or command
                if 'opt_name' in args and args['opt_name'] is not None:
                    if args['opt_name'][0] is not None:
                        opt_name = args['opt_name'][0]
                        if 'opt_value' in args and args['opt_value'] is not None:
                            try:
                                opt_value = json.loads(args['opt_value'][0])
                            except ValueError:
                                opt_value = args['opt_value'][0]

            # requesting stream status details
            if request == "/last_frame_number":
                if v_processor is not None:
                    data = str.encode(str(v_processor.output_stream.get_last_frame_number()))
                else:
                    with open(self.model_folder + os.sep + "status_info.json", "r") as f:
                        status_info = json.load(f)
                    data = str.encode(str(status_info["output_stream.last_frame_number"]))
                is_gzipped = False

            # requesting model options and details
            elif request == "/options":
                with open(self.model_folder + os.sep + "options.json", "r") as f:
                    loaded_options = json.load(f)
                data = str.encode(json.dumps(loaded_options))
                is_gzipped = False

            # requesting supervision details
            if request == "/supervision_map_and_counts":
                if v_processor is not None:
                    sup_map_count = {"supervision_map": v_processor.worker.get_supervision_map(),
                                     "supervision_count": v_processor.worker.get_supervision_count()}
                    data = str.encode(json.dumps(sup_map_count))
                else:
                    file_name, _ = lve.OutputStream.get_full_file_name_of("sup.map", None,
                                                                          lve.OutputType.JSON,
                                                                          self.output_folder,
                                                                          None)

                    try:
                        with open(file_name, "rb") as f:
                            sup_map = json.load(f)
                    except FileNotFoundError:
                        sup_map = None
                        pass

                    with open(self.model_folder + os.sep + "worker" + os.sep + "params.json", "r") as f:
                        params = json.load(f)
                        sup_map = sup_map if sup_map is not None else params["supervision_count"]
                        sup_count = params["supervision_count"]

                        if len(sup_count) < len(sup_map):
                            for k in sup_map.keys():
                                if k not in sup_count:
                                    sup_count[k] = 0

                        sup_map_count = \
                            {"supervision_map": sup_map,
                             "supervision_count": sup_count}
                        data = str.encode(json.dumps(sup_map_count))
                is_gzipped = False

            # requesting multiple output elements at once, by packing them into a single JSON (prefix: custom)
            elif request == "/all_stats_json":
                if sync and v_processor is not None:
                    data = {}
                    found_something = False

                    for element_name in self.all_stats_json:
                        element = self.v_processor.remote_get_data_to_visualize(element_name)
                        if element is not None:
                            found_something = True
                            data[element_name] = element
                else:
                    data = {}
                    found_something = False
                    for element_name in self.all_stats_json:
                        file_name, _ = lve.OutputStream.get_full_file_name_of(element_name, frame_number-1,
                                                                              lve.OutputType.JSON,
                                                                              self.output_folder,
                                                                              self.files_per_folder)
                        try:
                            element = lve.utils.load_json(file_name)
                        except FileNotFoundError:
                            element = None

                        if element is not None:
                            found_something = True
                            data[element_name] = element

                if found_something:
                    data = Handler.__gzip(data)
                else:
                    data = None
                is_gzipped = True

            # requesting a vprocessor-related operation
            elif request == "/vprocessor_allow_processing":
                if self.v_processor is not None:
                    self.v_processor.remote_allow_processing()
                    data = str.encode(str(1))
                else:
                    data = str.encode(str(0))
                is_gzipped = False

            # requesting a vprocessor-related operation
            elif request == "/vprocessor_allow_processing_next_frame_only":
                if self.v_processor is not None:
                    ret = self.v_processor.remote_allow_processing_next_frame_only()
                    data = str.encode(str(ret))
                else:
                    data = str.encode(str(0))
                    is_gzipped = False

            # requesting a vprocessor-related operation
            elif request == "/vprocessor_disable_processing_asap":
                if self.v_processor is not None:
                    self.v_processor.remote_disable_processing_asap()
                    data = str.encode(str(1))
                else:
                    data = str.encode(str(0))
                    is_gzipped = False

            # requesting a vprocessor-related operation
            elif request == "/vprocessor_is_processing_allowed":
                if self.v_processor is not None:
                    ret = self.v_processor.remote_is_processing_allowed()
                    if ret:
                        ret = "1"
                    else:
                        ret = "0"
                else:
                    ret = "-1"
                data = str.encode(ret)
                is_gzipped = False

            # requesting a vprocessor-related operation
            elif request == "/vprocessor_command" and opt_name is not None and opt_value is not None:
                if self.v_processor is not None:
                    self.v_processor.remote_command(opt_name, opt_value)
                    data = str.encode("1")
                else:
                    data = str.encode("0")
                is_gzipped = False

            # requesting a vprocessor-related operation
            elif request == "/vprocessor_change_option":
                if self.v_processor is not None and opt_name is not None and opt_value is not None:
                    if self.v_processor.remote_change_option(opt_name, opt_value):
                        data = str.encode("1")
                    else:
                        data = str.encode("0")
                else:
                    data = str.encode("0")
                is_gzipped = False

            # requesting an output element (some examples: /frames, /motion, /features.3, /others.foa, ...)
            elif request[0] == "/":
                element_name = request[1:]

                # hack: forcing blurred frames to be sent as plain frames, if available
                if element_name == "frames":
                    if "blurred" in self.data_types:
                        element_name = "blurred"

                if sync and v_processor is not None:
                    data = Handler.__gzip(self.v_processor.remote_get_data_to_visualize(element_name))
                    is_gzipped = True
                else:
                    try:
                        data_type = self.data_types[element_name]
                        file_name, _ = lve.OutputStream.get_full_file_name_of(element_name, frame_number-1, data_type,
                                                                              self.output_folder,
                                                                              self.files_per_folder)
                    except KeyError:
                        data_type = None
                        file_name = "__dummy__"
                        pass

                    try:
                        with open(file_name, "rb") as f:
                            data = f.read()  # reading from disk

                        # g-zipped flag and data encoding (when needed)
                        if data_type == lve.OutputType.IMAGE:
                            is_gzipped = False
                        elif data_type == lve.OutputType.BINARY:
                            if not self.gzipped_bin:
                                data = Handler.__gzip(data)
                            is_gzipped = True
                        elif data_type == lve.OutputType.JSON:
                            data = str.encode(data.decode('utf-8'))
                            is_gzipped = False
                        else:
                            raise ValueError("Unsupported data type!")
                    except FileNotFoundError:
                        pass

            # unsupported data request
            if data is None:
                raise IOError()

            return data, is_gzipped

        @staticmethod
        def __gzip(numpy_data_or_json):
            if numpy_data_or_json is not None:
                f = io.BytesIO()
                writer = GzipFile(fileobj=f, mode="wb")
                if isinstance(numpy_data_or_json, np.ndarray):
                    np.save(writer, numpy_data_or_json)
                else:
                    json_str = json.dumps(numpy_data_or_json)
                    writer.write(json_str.encode('utf-8'))
                writer.close()
                f.seek(0)
                data = f.read()
                f.close()
                return data
            else:
                return None

    return Handler

