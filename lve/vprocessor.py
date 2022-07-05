import os
import sys
import signal
import numpy as np
from functools import partial
import lve
import time
from threading import Event
import torch
import torch.autograd.profiler as profiler
import subprocess
import wandb

from lve.metrics_hs import HsMetricsContainer
from lve.utils import visualize_flows

class VProcessor:

    def __init__(self, input_stream, output_stream, worker, model_folder, visualization_port=0,
                 resume=False, stop_condition=None, wandb=False, save_every=1000, print_every=25, save_callback=None):
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.worker = worker
        self.model_folder = os.path.abspath(model_folder)
        self.saved_at = None
        self.save_every = save_every
        self.save_callback = save_callback
        self.print_every = print_every

        # creating model folder
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

        # locks and other remote-control-related fields
        self.__wandb = wandb
        self.__event_visualization_is_happening = Event()
        self.__event_processing_is_running = Event()
        self.__worker_options_to_change = None
        self.__event_visualization_is_happening.set()
        self.__event_processing_is_running.set()

        # registering output elements that will be saved by the worker
        self.output_stream.register_output_elements(self.worker.get_output_types())

        # metrics (registering also metric-related output elements)
        if 'metrics' in self.worker.options and self.worker.options['metrics'] is not None:
            if 'hs_evaluation' in self.worker.options['metrics']:
                self.metrics_container = HsMetricsContainer(
                    options=self.worker.options['metrics'],
                    output_stream=self.output_stream)
        else:
            self.metrics_container = None

        # stream options
        self.options = {'input': input_stream.readable_input,
                        'input_type': lve.InputType.readable_type(input_stream.input_type),
                        'w': input_stream.w,
                        'h': input_stream.h,
                        'c': input_stream.c,
                        'fps': input_stream.fps,
                        'frames': input_stream.frames,
                        'repetitions': input_stream.repetitions,
                        'max_frames': input_stream.max_frames,
                        'output_folder': output_stream.folder,
                        'output_folder_gzipped_bin': self.output_stream.is_gzipping_binaries(),
                        'output_folder_files_per_subfolder':
                            self.output_stream.get_max_number_of_files_per_subfolder(),
                        'output_folder_data_types': self.output_stream.get_data_types()
                        }

        # opening a visualization service
        self.visual_server = lve.VisualizationServer(visualization_port,
                                                     output_folder=output_stream.folder,
                                                     model_folder=self.model_folder,
                                                     v_processor=self)

        # running tensorboard (on the port right next the one of the visualization server)
        if self.output_stream.tensorboard:
            tensorboad_port = visualization_port + 1
            subprocess.Popen(["tensorboard --logdir=" + self.output_stream.folder + os.sep + "tensorboard" +
                              " --host 0.0.0.0 --port=" + str(tensorboad_port)], shell=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            tensorboad_port = -1

        # updating model options
        self.options.update({'tensorboard_port': str(tensorboad_port),
                             'visualization_server_ip': str(self.visual_server.ip),
                             'visualization_server_port': str(self.visual_server.port),
                             'visualization_server_url': "http://" + str(self.visual_server.ip) +
                                                         ":" + str(self.visual_server.port)})

        self.options["worker"] = self.worker.options

        # turning on output generation "on request"
        if self.output_stream.virtual_save:
            self.worker.set_heavy_output_data_needed(False)

        # printing info
        self.input_stream.print_info()
        print("")
        self.output_stream.print_info()
        print("")
        self.visual_server.print_info()
        print("")

        # resuming (if needed)
        if resume:
            print("Resuming...")
            self.__resumed = True
            self.load()
        else:
            print("No resume...")
            self.__resumed = False

        # saving references
        self.__stop_condition = stop_condition

        # finally, saving options to disk (keep this at the end of the method)
        self.__save_options()

    def __del__(self):
        if hasattr(self, 'visual_server'): self.visual_server.close()

    def process_video(self, log_dict=None):
        check_ctrlc = True
        elapsed_time = 0.0

        # handling CTRL-C
        def interruption(status_array_vprocessor, signal, frame):
            status_array = status_array_vprocessor[0]
            vprocessor = status_array_vprocessor[1]
            vprocessor.remote_allow_processing()
            if status_array[0] < 4:
                status_array[0] = status_array[0] + 1
                print("*** CTRL-C detected, waiting for current processing to finish...")
            else:
                print("*** Several CTRL-C detected, forcing exit.")
                os._exit(1)

        status = [1]
        if check_ctrlc:
            signal.signal(signal.SIGINT, partial(interruption, [status, self]))

        # counters and info
        batch_size = self.options["worker"]["batch_size"] if "batch_size" in self.options["worker"] else 1
        steps = 0
        tot_frames_str = "/" + str(self.input_stream.frames) if self.input_stream.frames != sys.maxsize else ""
        cur_frame_number = self.input_stream.get_last_frame_number() + 1
        step_time = np.zeros(batch_size)
        step_io_time = np.zeros(batch_size)
        step_io_time_2 = np.zeros(batch_size)
        fps = 0.0
        batch_img = [None] * batch_size
        batch_index = 0

        # main loop over frames
        while status[0] == 1:

            # saving (every 1000 steps)
            if steps % self.save_every == 0 and (steps > 0 or (not self.__resumed)):
                print("Saving model...")
                self.save(steps)

            if steps % self.print_every == 0:
                print("Processing frame " + str(cur_frame_number) + tot_frames_str +
                      " {prev_time: " + "{0:.3f}".format(step_time[batch_index]) + ", prev_proctime: " +
                      "{0:.3f}".format(step_time[batch_index] -
                                       step_io_time[batch_index] -
                                       step_io_time_2[batch_index]) + ", " +
                      "avg_fps: " + "{0:.3f}".format(fps) + "}")

            start_step_time = time.time()

            # getting next frame(s), eventually packing them into a batched-tensor
            batch_index = steps % batch_size
            if batch_index == 0:
                batch_of = [None] * batch_size
                batch_supervisions = [None] * batch_size
                batch_foa = [None] * batch_size
                got_something = False
                got_sup = False

                for i in range(0, batch_size):
                    img, of, supervisions, foa = self.input_stream.get_next()

                    # if reached the end of the stream...
                    if img is None or self.__stop_condition is not None and self.__stop_condition():
                        if i > 0:
                            batch_img = batch_img[0:i]
                            batch_of = batch_of[0:i]
                            batch_supervisions = batch_supervisions[0:i]
                            batch_foa = batch_foa[0, i]
                        break
                    else:
                        got_something = True
                        batch_img[i] = img
                        batch_of[i] = of  # it can be None
                        batch_supervisions[i] = supervisions  # it can be None
                        batch_foa[i] = foa  # it can be None

                        if supervisions is not None:
                            got_sup = True

                # purging
                if not got_sup:
                    batch_supervisions = None
                    unfiltered_batch_supervisions = None
                else:
                    unfiltered_batch_supervisions = batch_supervisions

                step_io_time[batch_index] = time.time() - start_step_time

                # stop condition
                if not got_something:
                    print("End of stream!")
                    break

                # preparing the output stream
                self.output_stream.clear_data_of_output_elements()

                # processing frame (forward)
                self.worker.process_frame(batch_img, of=batch_of, supervisions=batch_supervisions, foa=batch_foa)

                # printing
                if steps % self.print_every == 0:
                    self.worker.print_info()

                # saving output
                start_io_time_2 = time.time()

                self.output_stream.save_element("frames", batch_img[0])
                self.output_stream.save_elements(self.worker.get_output(batch_index=0))
                self.output_stream.save_done()
                step_io_time_2[batch_index] = time.time() - start_io_time_2
            else:
                step_io_time[batch_index] = time.time() - start_step_time

                # stop condition
                if batch_index >= len(batch_img):
                    print("End of stream!")
                    break

                # saving output
                start_io_time_2 = time.time()

                self.output_stream.save_element("frames", batch_img[batch_index])
                self.output_stream.save_elements(self.worker.get_output(batch_index=batch_index))
                self.output_stream.save_done()
                step_io_time_2[batch_index] = time.time() - start_io_time_2

            output_elements = self.output_stream.get_output_elements()

            # updating worker parameters (backward)
            if batch_index == batch_size - 1:
                updated = self.worker.update_model_parameters()
                self.worker.update_stats({'updated': updated})

            if 'predicted_motion' in output_elements and output_elements["predicted_motion"][
                "data"] is not None and \
                    self.worker.options["metrics"] is not None:
                self.metrics_container.update(
                    updated=updated,
                    hs_invariance_term=output_elements['stats.worker']['data']['loss_invariance'],
                    hs_smoothness_term=output_elements['stats.worker']['data']['loss_smoothness'],
                    hs_loss=output_elements['stats.worker']['data']['hs_loss'],
                    photo_and_smooth_loss=output_elements['stats.worker']['data'][
                        'photo_and_smooth_loss'],
                    photo_term=output_elements['stats.worker']['data']['loss_photometric'],
                    flow_std=output_elements['stats.worker']['data']['flow_std'],
                    recon_acc=output_elements['stats.worker']['data']['recon_acc'],
                    motion_mask=output_elements['motion_mask']['data'],
                    predicted_motion_mask=output_elements['predicted_motion_mask']['data']
                )
                self.metrics_container.compute()
                # self.metrics.print_info() #

            # eventually logging on a python list some output elements that could be needed
            if log_dict is not None:
                if log_dict['element'] in output_elements:
                    elem = output_elements[log_dict['element']]['data']
                    if isinstance(elem, list):
                        log_dict['logged'].append(elem.copy())
                    elif isinstance(elem, dict):
                        log_dict['logged'].append(elem.copy())
                    elif isinstance(elem, str):
                        log_dict['logged'].append(str(elem))
                    if log_dict['log_last_only'] and len(log_dict['logged']) > 1:
                        del log_dict['logged'][0]

            # locking and other remote-control-related procedures
            self.__event_processing_is_running.set()
            self.__event_visualization_is_happening.wait()
            self.__event_processing_is_running.clear()

            # handling received commands (and updating the output data, if some commands had effects on such data)
            if self.worker.handle_commands(batch_index=batch_index):
                self.output_stream.save_elements(self.worker.get_output(batch_index=batch_index), prev_frame=True)
            self.__handle_hot_option_changes()

            # updating system status
            cur_frame_number = cur_frame_number + 1
            steps = steps + 1
            end_of_step_time = time.time()
            step_time[batch_index] = end_of_step_time - start_step_time

            if batch_index == batch_size - 1:
                if steps == batch_size:
                    elapsed_time = np.sum(step_time)
                elapsed_time = elapsed_time * 0.95 + np.sum(step_time) * 0.05
                fps = batch_size / elapsed_time
            elif steps == 1:
                fps = 1.0 / step_time[0]

        # last save
        self.save(steps)
        print("Done! (model saved)")

        # quit the visualization service
        self.__event_visualization_is_happening.set()
        self.__event_processing_is_running.set()
        self.visual_server.close()

    def load(self):

        # loading options
        loaded_options = lve.utils.load_json(self.model_folder + os.sep + "options.json")

        # checking if inout stream options have changed
        input_steam_opts = ['input', 'w', 'h', 'c', 'fps', 'frames', 'repetitions', 'max_frames']

        input_stream_changed = False
        for io_opt in input_steam_opts:
            if self.options[io_opt] != loaded_options[io_opt]:
                input_stream_changed = True
                print("WARNING: input stream configuration has changed! current " + io_opt + ": " +
                      str(self.options[io_opt]) + ", loaded " + io_opt + ": " + str(loaded_options[io_opt]))

        # loading status info
        status_info = lve.utils.load_json(self.model_folder + os.sep + "status_info.json")

        # setting up the input stream
        if not input_stream_changed:
            self.input_stream.set_last_frame_number(status_info['input_stream.last_frame_number'])
            assert abs(self.input_stream.get_last_frame_time() - status_info['input_stream.last_frame_time']) < 0.01

        # setting up the output stream
        if not self.output_stream.is_newly_created():
            self.output_stream.set_last_frame_number(status_info['output_stream.last_frame_number'])


        # setting up the main worker
        if self.worker.options['metrics'] is not None:
            self.worker.load(self.model_folder)

        # loading metrics
        self.metrics_container.load(self.model_folder, self.worker.options['metrics'])

    def save(self, steps):
        if self.saved_at is not None and steps - self.saved_at <= 1:
            print('*** Skipping saving, saved', steps - self.saved_at, 'steps ago')
            return False

        status_info = {'input_stream.last_frame_number': self.input_stream.get_last_frame_number(),
                       'input_stream.last_frame_time': self.input_stream.get_last_frame_time(),
                       'output_stream.last_frame_number': self.output_stream.get_last_frame_number(),
                       'number_of_frames_processed_during_last_run': steps}

        # saving stream/processing status
        lve.utils.save_json(self.model_folder + os.sep + "status_info.json", status_info)

        # saving worker
        self.worker.save(self.model_folder)

        stats_worker = self.output_stream.get_output_elements()['stats.worker']

        if self.metrics_container is not None:
            stats_metrics = self.output_stream.get_output_elements()['stats.metrics']
        else:
            stats_metrics = {'data': None}

        if stats_worker['data'] is not None and len(stats_worker['data']) > 0 and np.isnan(stats_worker['data']['loss']):
            print('Loss is nan') # CHECK
            wandb.run.finish(exit_code=1)
            os._exit(1)

        if self.__wandb and stats_metrics['data'] is not None and len(stats_metrics['data']) > 0 \
                and stats_worker['data'] is not None and len(stats_worker['data']) > 0:
            rm_fnames = []

            if self.save_callback is None:
                if 'predicted_motion' in self.output_stream.get_output_elements():
                    warped, prev_frame, frame, flow, motion, motion_mask, predicted_motion_mask = self.worker.get_warped_frame()
                    wandb_dict = {
                        'photo_and_smooth_w': stats_metrics['data']["whole_frame"]["window"]["photo_and_smooth"],
                        'hs_w': stats_metrics['data']["whole_frame"]["window"]["hs"],
                        'hs_invariance_w': stats_metrics['data']["whole_frame"]["window"]["hs_invariance"],
                        'hs_smoothness_w': stats_metrics['data']["whole_frame"]["window"]["hs_smoothness"],
                        'warped': wandb.Image(warped[0, 0]),
                        'prev_frame': wandb.Image(prev_frame[0, 0]),
                        'frame': wandb.Image(frame[0, 0]),
                        'largest_predicted_flow_x': stats_worker['data']["largest_predicted_flow_x"],
                        'largest_predicted_flow_y': stats_worker['data']["largest_predicted_flow_y"],
                        'error_rate_w': stats_metrics['data']["whole_frame"]["window"]["error_rate"],
                        'l2_dist_w': stats_metrics['data']["whole_frame"]["window"]["l2_dist"],
                        'l2_dist_moving_w': stats_metrics['data']["whole_frame"]["window"]["l2_dist_moving"],
                        'l2_dist_still_w': stats_metrics['data']["whole_frame"]["window"]["l2_dist_still"],
                        'eq_iterations_w': stats_metrics['data']["whole_frame"]["window"]["eq_iterations"],
                        'recon_acc_w': stats_metrics['data']["whole_frame"]["window"]["recon_acc"],
                        'moving_f1_w': stats_metrics['data']["whole_frame"]["window"]["moving_f1"],
                        'moving_acc_w': stats_metrics['data']["whole_frame"]["window"]["moving_acc"],
                        'moving_cm_w': stats_metrics['data']["whole_frame"]["window"]["moving_cm"],
                        'moving_precision_w': stats_metrics['data']["whole_frame"]["window"]["moving_precision"],
                        'moving_recall_w': stats_metrics['data']["whole_frame"]["window"]["moving_recall"]
                    }
                    flow_dic, rm_fnames = visualize_flows(flow, prefix='flow')
                    wandb_dict.update(flow_dic)
                    motion_dic, rm_fnames_ = visualize_flows(motion, prefix='ground_truth')
                    wandb_dict.update(motion_dic)

                    wandb.log(wandb_dict)
                    for x in rm_fnames + rm_fnames_:
                        os.remove(x)
            else:
                self.save_callback(stats_metrics, stats_worker)

        # saving metrics
        if 'metrics' in self.worker.options and self.worker.options['metrics'] is not None:
            self.metrics_container.save(self.model_folder)

        self.saved_at = steps

    # called by the web-server to handle requests from the visualization client
    def remote_change_option(self, name, value_str):
        print("Received worker option change request: " + str(name) + " -> " + str(value_str))
        names = name.split(".")
        opt = self.options["worker"]
        for n in names:
            if n not in opt:
                print("ERROR: Unknown option: " + name)
                return
            else:
                if n != names[-1]:
                    opt = opt[n]
                else:
                    current_value = opt[n]
        try:
            if isinstance(current_value, int):
                casted_value = int(value_str)
            elif isinstance(current_value, float):
                casted_value = float(value_str)
            elif isinstance(current_value, bool):
                casted_value = bool(value_str)
            elif isinstance(current_value, str):
                casted_value = str(value_str)
            elif isinstance(current_value, list):
                values_str_array = current_value.strip()[1:-1].split(",")
                if isinstance(current_value[0], int):
                    casted_value = [int(i) for i in values_str_array]
                elif isinstance(current_value[0], float):
                    casted_value = [float(i) for i in values_str_array]
                elif isinstance(current_value[0], str):
                    casted_value = [str(i) for i in values_str_array]
            else:
                print("ERROR: Skipping option change request due to unhandled type: " + name + " -> " + value_str)
                return False
        except ValueError:
            print("ERROR: Skipping option change request due not-matching type: " + name + " -> " + value_str)
            return False

        if self.__worker_options_to_change is None:
            self.__worker_options_to_change = {}
        self.__worker_options_to_change[name] = {"fields": names, "value": casted_value}
        return True

    # called by the web-server to handle requests from the visualization client
    def remote_command(self, command_name, command_value):
        print("Received command: " + command_name + " -> " + str(command_value))
        self.worker.send_command(command_name, command_value)

    # called by the web-server to handle requests from the visualization client
    def remote_allow_processing_next_frame_only(self):
        if self.output_stream.virtual_save:
            self.worker.set_heavy_output_data_needed(True)

        self.__event_visualization_is_happening.set()  # allow the system to process next frames...
        self.__event_visualization_is_happening.clear()  # ...and immediately block it again
        return self.output_stream.get_last_frame_number() + 1

    # called by the web-server to handle requests from the visualization client
    def remote_allow_processing(self):
        if self.output_stream.virtual_save:
            self.worker.set_heavy_output_data_needed(False)

        self.__event_visualization_is_happening.set()  # always allow to process next frames

    # called by the web-server to handle requests from the visualization client
    def remote_is_processing_allowed(self):
        return self.__event_visualization_is_happening.is_set()

    # called by the web-server to handle requests from the visualization client
    def remote_disable_processing_asap(self):
        if self.output_stream.virtual_save:
            self.worker.set_heavy_output_data_needed(True)

        self.__event_visualization_is_happening.clear()  # block attempts to process next frame

    # called by the web-server to handle requests from the visualization client
    def remote_get_data_to_visualize(self, data_identifier):
        self.__event_visualization_is_happening.clear()  # block attempts to process next frame
        self.__event_processing_is_running.wait()  # wait until processing of the current frame has ended
        try:
            return self.output_stream.get_output_elements()[data_identifier]["data"]
        except KeyError:
            return None

    def __handle_hot_option_changes(self):
        something_changed = False
        if self.__worker_options_to_change is not None:
            for name in self.__worker_options_to_change:
                opt_w = self.worker.options
                names = self.__worker_options_to_change[name]["fields"]
                for n in names:
                    if n != names[-1]:
                        opt_w = opt_w[n]
                    else:

                        # patching the case of boolean
                        if isinstance(opt_w[n], bool):
                            if self.__worker_options_to_change[name]["value"] != 0:
                                self.__worker_options_to_change[name]["value"] = True
                            else:
                                self.__worker_options_to_change[name]["value"] = False

                        if opt_w[n] != self.__worker_options_to_change[name]["value"]:
                            opt_w[n] = self.__worker_options_to_change[name]["value"]
                            something_changed = True

            if something_changed:
                self.__save_options()
        self.__worker_options_to_change = None

    def __save_options(self):

        # filtering options: keeping only key that do not start with "_"
        options_filtered = {}
        queue = [self.options]
        queue_f = [options_filtered]

        while len(queue) > 0:
            opt = queue.pop()
            opt_f = queue_f.pop()
            if isinstance(opt, dict):
                for k in opt:
                    if k[0] != '_':
                        if isinstance(opt[k], dict):
                            opt_f[k] = {}
                            queue.append(opt[k])
                            queue_f.append(opt_f[k])
                        else:
                            opt_f[k] = opt[k]  # copying

        lve.utils.save_json(self.model_folder + os.sep + "options.json", options_filtered)
