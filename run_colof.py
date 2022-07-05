import time
from math import floor

import torchsummary

import lve
import os
import cv2
import numpy as np
import wandb

import argparse
import torch
import matplotlib.pyplot as plt

from lve.colof import WandbSaveCallback, FULL_EXP_TYPE, MODEL_SELECTION_EXP_TYPE, ColofInputStream, default_params, \
    print_stats
from settings import *

def run_exp(args_cmd):
    DEFAULT_MODEL_FOLDER = 'model_folder'
    if args_cmd.arch == 'sota-smurf':
        from lve.worker_smurf import WorkerSmurf
    elif args_cmd.arch == 'sota-raft' or args_cmd.arch == 'sota-raft-small':
        from lve.worker_raft import WorkerRaft

    STREAM_MOVIE = "data/1917-cut"
    STREAM_A = "data/stream_a_9424" #3569 #9424
    STREAM_B = "data/stream_b_8714"
    STREAM_C = "data/stream_c_1235"

    stream_props = {
        STREAM_A: {'frames_count': 90000},
        STREAM_B: {'frames_count': 90000},
        STREAM_C: {'frames_count': 90000},
        STREAM_MOVIE: {'frames_count': 153760}
    }

    # Remember that Unity-generated streams have y-axis flow inverted
    stream_pref = {
        STREAM_A: {'ground_truth_motion_threshold': 0.5, 'output_motion_threshold': 0.5, 'fix_flow_v': True, 'fix_flow_u': False},
        STREAM_B: {'ground_truth_motion_threshold': 0.5, 'output_motion_threshold': 0.5, 'fix_flow_v': True, 'fix_flow_u': False},
        STREAM_C: {'ground_truth_motion_threshold': -1.0, 'output_motion_threshold': -1.0, 'fix_flow_v':True, 'fix_flow_u': False},
        STREAM_MOVIE: {'ground_truth_motion_threshold': -1.0, 'output_motion_threshold': -1.0, 'fix_flow_v': False, 'fix_flow_u': False},
    }

    if args_cmd.experience == "a":
        dataset = STREAM_A
        other_settings = {
            'frozen_all_stream': {'dataset': STREAM_A, 'start_window': 0, 'duration_window': -1},
            'short_term_forgetting_trial': {'dataset': STREAM_A, 'start_window': 59, 'duration_window': 1},
            'long_term_forgetting_trial': {'dataset': STREAM_A, 'start_window': 0, 'duration_window': 1},
            'stream_b_trial': {'dataset': STREAM_B, 'start_window': 0, 'duration_window': 1},
            'stream_c_trial': {'dataset': STREAM_C, 'start_window': 0, 'duration_window': 1},
            'stream_movie_trial': {'dataset': STREAM_MOVIE, 'start_window': 0, 'duration_window': 1},
            'retuning_all_stream': {'dataset': STREAM_A, 'start_window': 0, 'duration_window': -1}
        }
    elif args_cmd.experience == "b":
        dataset = STREAM_B
        other_settings = {
            'frozen_all_stream': {'dataset': STREAM_B, 'start_window': 0, 'duration_window': -1},
            'short_term_forgetting_trial': {'dataset': STREAM_B, 'start_window': 59, 'duration_window': 1},
            'long_term_forgetting_trial': {'dataset': STREAM_B, 'start_window': 0, 'duration_window': 1},
            'stream_a_trial': {'dataset': STREAM_A, 'start_window': 0, 'duration_window': 1},
            'stream_c_trial': {'dataset': STREAM_C, 'start_window': 0, 'duration_window': 1},
            'stream_movie_trial': {'dataset': STREAM_MOVIE, 'start_window': 0, 'duration_window': 1},
            'retuning_all_stream': {'dataset': STREAM_B, 'start_window': 0, 'duration_window': -1}
        }
    elif args_cmd.experience == "c":
        dataset = STREAM_C
        other_settings = {
            'frozen_all_stream': {'dataset': STREAM_C, 'start_window': 0, 'duration_window': -1},
            'short_term_forgetting_trial': {'dataset': STREAM_C, 'start_window': 59, 'duration_window': 1},
            'long_term_forgetting_trial': {'dataset': STREAM_C, 'start_window': 0, 'duration_window': 1},
            'stream_a_trial': {'dataset': STREAM_A, 'start_window': 0, 'duration_window': 1},
            'stream_b_trial': {'dataset': STREAM_B, 'start_window': 0, 'duration_window': 1},
            'stream_movie_trial': {'dataset': STREAM_MOVIE, 'start_window': 0, 'duration_window': 1},
            'retuning_all_stream': {'dataset': STREAM_C, 'start_window': 0, 'duration_window': -1}
        }
    elif args_cmd.experience == "movie":
        dataset = STREAM_MOVIE
        other_settings = {
            'frozen_all_stream': {'dataset': STREAM_MOVIE, 'start_window': 0, 'duration_window': -1},
            'short_term_forgetting_trial': {'dataset': STREAM_MOVIE, 'start_window': 101.5, 'duration_window': 1},
            'long_term_forgetting_trial': {'dataset': STREAM_MOVIE, 'start_window': 0, 'duration_window': 1},
            'medium_term_forgetting_trial': {'dataset': STREAM_MOVIE, 'start_window': 50, 'duration_window': 1},
            'stream_a_trial': {'dataset': STREAM_A, 'start_window': 0, 'duration_window': 1},
            'stream_b_trial': {'dataset': STREAM_B, 'start_window': 0, 'duration_window': 1},
            'stream_c_trial': {'dataset': STREAM_C, 'start_window': 0, 'duration_window': 1},
            'retuning_all_stream': {'dataset': STREAM_MOVIE, 'start_window': 0, 'duration_window': -1}
        }
    elif args_cmd.experience == "cat":
        dataset = STREAM_A
        other_settings = {
            'cat_b_tuning': {'dataset': STREAM_B, 'start_window': 0, 'duration_window': -1},
            'cat_c_tuning': {'dataset': STREAM_C, 'start_window': 0, 'duration_window': -1},
            'cat_movie_tuning': {'dataset': STREAM_MOVIE, 'start_window': 0, 'duration_window': -1},
            'frozen_full_a': {'dataset': STREAM_A, 'start_window': 0, 'duration_window': -1},
            'frozen_full_b': {'dataset': STREAM_B, 'start_window': 0, 'duration_window': -1},
            'frozen_full_c': {'dataset': STREAM_C, 'start_window': 0, 'duration_window': -1},
            'frozen_full_movie': {'dataset': STREAM_MOVIE, 'start_window': 0, 'duration_window': -1}
        }
    else:
        raise NotImplementedError()

    if args_cmd.custom_playback is not None:
        if 'retuning_all_stream' in other_settings: del other_settings['retuning_all_stream']
        other_settings['frozen_all_stream_unp'] = dict(other_settings['frozen_all_stream'])

    # resuming
    resume = False

    fps = 25
    single_window_duration = 60 # in seconds
    window_frames = int(single_window_duration * fps)
    training_windows = default_params['training_windows']

    if args_cmd.exp_type == FULL_EXP_TYPE:
        max_frames = None
        max_training_frames = None
        training_windows = None
    elif args_cmd.exp_type == MODEL_SELECTION_EXP_TYPE:
        max_frames = (training_windows + 1) * window_frames
        max_training_frames = training_windows * window_frames
    else:
        raise NotImplementedError()

    start_time = time.time()
    ins = ColofInputStream(dataset, w=-1, h=-1, fps=None, max_frames=max_frames,
                         repetitions=1, force_gray=args_cmd.force_gray == "yes", foa_file=None,
                         unity_settings=None, custom_playback=args_cmd.custom_playback)
    # ins = lve.InputStream(dataset, w=-1, h=-1, fps=None, max_frames=max_frames, #skip_frames=15000,
    #                       repetitions=1, force_gray=args_cmd.force_gray == "yes", foa_file=None,
    #                       unity_settings=None)

    if args_cmd.custom_playback is not None:
        custom_playback = ins.custom_playback

    if args_cmd.exp_type == FULL_EXP_TYPE:
        windows = (ins.effective_video_frames if ins.max_frames < 0 else min(ins.max_frames, ins.effective_video_frames)) // window_frames
        log_windows_every = 1
    elif args_cmd.exp_type == MODEL_SELECTION_EXP_TYPE:
        windows = training_windows
        log_windows_every = 5

    log_windows = [0] + np.arange(-1, windows, log_windows_every).tolist()[1:-1] + [windows - 1, windows]

    output_settings = {
        'folder': args_cmd.output_folder,
        'fps': ins.fps,
        'virtual_save': args_cmd.save_output=="no",
        'tensorboard': False,
        'save_per_frame_data': True,
        'purge_existing_data': not resume
    }

    if 'cuda' in args_cmd.device and torch.cuda.is_available():
        device = args_cmd.device
    else:
        device = "cpu"

    infer_bw = default_params['infer_bw']
    deterministic = default_params['deterministic']

    #### OPTIONS
    general_options = {"device": device,  # "cuda:0",  # cpu, cuda:0, cuda:1, ...
                       "seed": args_cmd.seed,  # if smaller than zero, current time is used
                       "single_window_duration": single_window_duration,
                       "fix_flow_v": stream_pref[dataset]['fix_flow_v'],
                       "fix_flow_u": stream_pref[dataset]['fix_flow_u'],
                       'force_gray': args_cmd.force_gray,
                       'experience': args_cmd.experience,
                       'custom_playback': args_cmd.custom_playback,
                       'align_corners': True,
                       'deterministic': deterministic == 'yes'
                       }
    if args_cmd.exp_type == MODEL_SELECTION_EXP_TYPE:
        general_options['training_windows'] = training_windows
    else:
        general_options['other_domains'] = other_settings

    color_channels = (1 if args_cmd.force_gray=='yes' else 3)
    c = 2 * color_channels if args_cmd.net_flow_input_type == "implicit" else 3 * color_channels


    net_options = {'c': c,
                   'color_channels': color_channels,
                   'fps': ins.fps,
                   'weight_decay': args_cmd.weight_decay,
                   'step_size': args_cmd.step_size,  # a negative value triggers Adam
                   'output_dim': 2,
                   'num_what': 2,
                   'lambda_s': args_cmd.lambda_s,
                   'charb_eps': args_cmd.charb_eps,
                   'charb_alpha': args_cmd.charb_alpha,
                   'freeze': True if "sota" in args_cmd.arch else args_cmd.freeze == "yes",
                   'subsampling_updates': args_cmd.subsampling_updates,
                   'architecture': args_cmd.arch,
                   'training_max_frames': max_training_frames,
                   'training_loss': args_cmd.training_loss,
                   'infer_bw': infer_bw == 'yes',
                   'occlusions': 'no',
                   'recon_linf_thresh': [float(x) for x in args_cmd.recon_linf_thresh.split(',')],
                   'net_flow_input_type': args_cmd.net_flow_input_type,  # "explicit" , "implicit"
                   'compute_ihs': False,
                   'ground_truth_motion_threshold': stream_pref[dataset]['ground_truth_motion_threshold'],
                   'output_motion_threshold': stream_pref[dataset]['output_motion_threshold']
                   }
    if args_cmd.arch == 'none-ihs':
        net_options['iter_ihs'] = args_cmd.iter_ihs
        net_options['warm_ihs'] = args_cmd.warm_ihs == 'yes'

    metrics_options = {'window': window_frames, 'hs_evaluation': True, 'recon_acc_thresh': net_options['recon_linf_thresh']}

    # creating worker
    if args_cmd.arch == 'sota-smurf':
        worker = WorkerSmurf(ins.w, ins.h, ins.c, ins.fps, ins, options={
            **general_options,
            "net": net_options,
            "metrics": metrics_options
        })
    elif args_cmd.arch == 'sota-raft' or args_cmd.arch == 'sota-raft-small':
        worker = WorkerRaft(ins.w, ins.h, ins.c, ins.fps, ins, options={
            **general_options,
            "net": net_options,
            "metrics": metrics_options
        })
    else:
        worker = lve.WorkerHS(ins.w, ins.h, ins.c, ins.fps, ins, options={
            **general_options,
            "net": net_options,
            "metrics": metrics_options
        })

        if args_cmd.verbose == 'yes':
            channel_factor = 2 if args_cmd.net_flow_input_type == 'implicit' else 3
            torchsummary.summary(worker.net.model, (ins.c * channel_factor, ins.h, ins.w))

    if args_cmd.load is not None:
        worker.load(args_cmd.load)

    wandb_save_callback = WandbSaveCallback(worker, log_windows=log_windows)

    if args_cmd.exp_type == FULL_EXP_TYPE:
        setting_name = 'tuning_all_stream'
        wandb_save_callback.save_running = True
        wandb_save_callback.prefix = setting_name+'/'
    else:
        setting_name = 'model_selection'

    # logger
    log_dict = {'element': 'stats.metrics', 'log_last_only': True, 'logged': []}

    log_opts = {'': general_options,
                'net': net_options
                }
    total_options = {}
    for prefix, dic in log_opts.items():
        for key, val in dic.items():
            total_options[prefix + "_" + key] = val
    print(total_options)

    # processing stream
    target_port = args_cmd.port

    if wandb_mode == 'disabled':
        w_run = wandb.init(mode='disabled')
    else:
        w_run = wandb.init(project=wandb_project, entity=wandb_entity, config=total_options)

    port = target_port
    print('Starting VProcessor with visualizer at port ' + str(port) + '..')
    outs = lve.OutputStream(**output_settings)
    fldr = "model_folder" + os.sep + w_run.name + "_" + w_run.id if args_cmd.save == 'yes' else DEFAULT_MODEL_FOLDER

    lve.VProcessor(ins, outs, worker, fldr, visualization_port=port, resume=resume, wandb=False, save_every=window_frames, save_callback=wandb_save_callback).process_video(
        log_dict=log_dict)

    elapsed_time = time.time() - start_time
    print_stats(log_dict, fldr=fldr, setting=setting_name, show_f1=dataset in [STREAM_A, STREAM_B])

    # closing streams
    ins.close()
    outs.close()

    print("")
    print("Elapsed: " + str(elapsed_time) + " seconds")

    if args_cmd.exp_type == MODEL_SELECTION_EXP_TYPE:
        history_ = wandb_save_callback.buffer
        if len(history_) > 2:
            wandb.run.summary["moving_f1_selection:tuning"] = history_[-2]['moving_f1_w']
            wandb.run.summary["moving_f1_selection:freezed"] = history_[-1]['moving_f1_w']
            wandb.run.summary["moving_f1_selection:avg"] = np.mean([history_[-1]['moving_f1_w'], history_[-2]['moving_f1_w']])

            wandb.run.summary["flow_std_selection:tuning"] = history_[-2]['flow_std_w']
            wandb.run.summary["flow_std_selection:freezed"] = history_[-1]['flow_std_w']
            wandb.run.summary["flow_std_selection:avg"] = np.mean([history_[-1]['flow_std_w'], history_[-2]['flow_std_w']])

            wandb.run.summary["hs_smoothness_selection:tuning"] = history_[-2]['hs_smoothness_w']
            wandb.run.summary["hs_smoothness_selection:freezed"] = history_[-1]['hs_smoothness_w']
            wandb.run.summary["hs_smoothness_selection:avg"] = np.mean([history_[-1]['hs_smoothness_w'], history_[-2]['hs_smoothness_w']])

            wandb.run.summary["photo_and_smooth_selection:tuning"] = history_[-2]['photo_and_smooth_w']
            wandb.run.summary["photo_and_smooth_selection:freezed"] = history_[-1]['photo_and_smooth_w']
            wandb.run.summary["photo_and_smooth_selection:avg"] = np.mean([history_[-1]['photo_and_smooth_w'], history_[-2]['photo_and_smooth_w']])
            for t in metrics_options['recon_acc_thresh']:
                wandb.run.summary["recon_acc_selection:tuning."+str(t)] = history_[-2]['recon_acc_w'][t]
                wandb.run.summary["recon_acc_selection:freezed." + str(t)] = history_[-1]['recon_acc_w'][t]
                wandb.run.summary["recon_acc_selection:avg." + str(t)] = np.mean([history_[-1]['recon_acc_w'][t], history_[-2]['recon_acc_w'][t]])

    if args_cmd.exp_type == FULL_EXP_TYPE:
        table_metrics = ['moving_f1', 'recon_acc', 'photo_and_smooth_w', 'photo_w']
        table_data = {k: [] for k in table_metrics}

        def append_to_tables(default_mode='window'):
            for m in table_metrics:
                if m.endswith("_w") or m.endswith("_r"):
                    m_full = m
                else:
                    m_full = m + ('_r' if default_mode == 'running' else '_w')
                if len(wandb_save_callback.buffer) > 0:
                    table_data[m].append([wandb_save_callback.setting, wandb_save_callback.buffer[-1][m_full]])

        append_to_tables('window')


        # RUnning other settings
        for setting_name, setting_properties in other_settings.items():
            # save time avoiding running again with frozen weight, since tuning corresponds with frozen on baselines
            if ('sota' in args_cmd.arch or args_cmd.arch == 'none' or args_cmd.arch == 'none-ihs') \
                    and (setting_name == 'frozen_all_stream' or setting_name == 'retuning_all_stream'): continue
            if 'tuning' in setting_name:
                worker.options['net']['freeze'] = False
            else:
                worker.options['net']['freeze'] = True
            print('-- running setting', setting_name, setting_properties)
            worker.options['net']['ground_truth_motion_threshold'] = stream_pref[setting_properties['dataset']]['ground_truth_motion_threshold']
            worker.options['net']['output_motion_threshold'] = stream_pref[setting_properties['dataset']]['output_motion_threshold']
            worker.options['fix_flow_v'] = stream_pref[setting_properties['dataset']]['fix_flow_v']
            worker.options['fix_flow_u'] = stream_pref[setting_properties['dataset']]['fix_flow_u']
            wandb_save_callback = WandbSaveCallback(worker)
            wandb_save_callback.save_running = True
            wandb_save_callback.prefix = setting_name + "/"
            start_frames = floor(setting_properties['start_window'] * window_frames)
            end_frames = start_frames + setting_properties['duration_window'] * window_frames if setting_properties['duration_window'] >= 0 else None
            if args_cmd.custom_playback is not None:
                if 'frozen_all_stream_unp' == setting_name:
                    orig_stream_duration = stream_props[setting_properties['dataset']]['frames_count']
                    moving_frames = 0
                    total_frames = 0
                    for i in range(orig_stream_duration // sum(custom_playback)):
                        moving_frames += custom_playback[0]
                        total_frames += sum(custom_playback)
                    if orig_stream_duration > total_frames:
                        if total_frames + custom_playback[0] > orig_stream_duration:
                            moving_frames += orig_stream_duration - total_frames
                        else:
                            moving_frames += custom_playback
                    ins = lve.InputStream(setting_properties['dataset'], w=-1, h=-1, fps=None, max_frames=moving_frames,
                                          skip_frames=0,
                                          repetitions=1, force_gray=args_cmd.force_gray == "yes", foa_file=None,
                                          unity_settings=None)
                else:
                    ins = ColofInputStream(setting_properties['dataset'], w=-1, h=-1, fps=None, max_frames=end_frames , skip_frames=start_frames,
                                       repetitions=1, force_gray=args_cmd.force_gray == "yes", foa_file=None,
                                       unity_settings=None, custom_playback=args_cmd.custom_playback)
            else:
                ins = lve.InputStream(setting_properties['dataset'], w=-1, h=-1, fps=None, max_frames=end_frames, skip_frames=start_frames,
                                      repetitions=1, force_gray=args_cmd.force_gray == "yes", foa_file=None,
                                      unity_settings=None)
            worker.ins = ins
            if hasattr(worker, 'subsampling_policy') and worker.subsampling_policy is not None:
                worker.subsampling_policy.warmup = 0
            worker.reset()
            worker.set_h_w(h=ins.h, w=ins.w)
            outs = lve.OutputStream(**output_settings)
            lve.VProcessor(ins, outs, worker, DEFAULT_MODEL_FOLDER, visualization_port=port, resume=False, wandb=True, save_every=window_frames,
                           save_callback=wandb_save_callback).process_video(log_dict=log_dict)
            print_stats(log_dict, fldr=fldr, setting=setting_name, show_f1= setting_properties['dataset'] in [STREAM_A, STREAM_B])
            append_to_tables('running')
            outs.close()
            ins.close()

        table_dict = {}
        for m in table_metrics:
            table = wandb.Table(data=table_data[m], columns=["Setting", "Value"])
            bar_plot = wandb.plot.bar(table, "Setting", "Value", title=m)
            table_dict[m+'_settings'] = bar_plot
        wandb.log(table_dict)
        run_url = wandb.run.url
        return run_url


def main():
    parser = argparse.ArgumentParser(description='Continual Unsupervised Learning for Optical Flow Estimation with Deep Networks -- experiments')
    parser.add_argument('--step_size', type=float, default=default_params['step_size'], help='learning rate of neural network optimizer')
    parser.add_argument('--weight_decay', type=float, default=default_params['weight_decay'], help='weight decay of neural network optimizer')
    parser.add_argument('--lambda_s', type=float, default=default_params['lambda_s'], help='weight of the smoothness penalty in the loss function')
    parser.add_argument('--charb_eps', type=float, default=default_params['charb_eps'], help='epsilon parameter in Charbonnier distance')
    parser.add_argument('--charb_alpha', type=float, default=default_params['charb_alpha'], help='alpha parameter in Charbonnier distance')
    parser.add_argument('--recon_linf_thresh', type=str, default=default_params['recon_linf_thresh'], help='threshold for reconstruction accuracy computation (comma separated if you want to have multiple thresholds')
    parser.add_argument('--subsampling_updates', type=str, default=default_params['subsampling_updates'], help='update subsampling policies: integer [decimation factor: 0 for no subsampling], avgflow(q, r, warmup_frames, force_update_every_n_frames), avgflowhistory(l, warmup_frames, force_update_every_n_frames), avgdiff(q, warmup_frames) ')
    parser.add_argument('--force_gray', type=str, default=default_params['force_gray'], choices=["yes", "no"], help='convert stream data to grayscale')
    parser.add_argument('--net_flow_input_type', type=str, default=default_params['net_flow_input_type'], choices=["implicit", "explicit"],
                        help='flag to indicate whether the network input is the two frames or their estimated derivatives')
    parser.add_argument('--device', type=str, default=default_params['device'], help='computation device (cpu, cuda..)')
    parser.add_argument('--iter_ihs', type=int, default=default_params['iter_ihs'], help='HS iteration limit, only to be set for HS')
    parser.add_argument('--warm_ihs', type=str, default=default_params['warm_ihs'], help='HS warm start option, only to be set for HS')
    parser.add_argument('--save', type=str, default=default_params['save'], choices=["yes","no"], help="flag to create a separate folder for the current experiment, 'yes' to persist the model")
    parser.add_argument('--save_output', type=str, default=default_params['save_output'], choices=["yes","no"], help="flag to save the flow predictions")
    parser.add_argument('--freeze', type=str, default=default_params['freeze'], choices=["yes","no"], help="flag to skip all the weights update operations")
    parser.add_argument('--load', type=str, default=default_params['load'], help='path for pre-loading models')
    parser.add_argument('--exp_type', type=str, default=default_params['exp_type'], choices=[FULL_EXP_TYPE, MODEL_SELECTION_EXP_TYPE], help="flag to indicate whether the current run is for model selection (short portion of stream) or full experiment")
    parser.add_argument('--training_loss', type=str, default=default_params['training_loss'], choices=["photo_and_smooth", "hs"], help='neural network loss function (photometric+smoothness or HS-like loss')
    parser.add_argument('--port', type=int, default=default_params['port'], help='visualizer port')
    parser.add_argument('--arch', type=str, default=default_params['arch'],
                        choices=["none", "none-ihs",
                                 "resunetof", "ndconvof", "dilndconvof", "flownets",
                                 "sota-flownets", "sota-smurf", "sota-raft", "sota-raft-small"], help='neural architecture for flow prediction')
    parser.add_argument('--experience', type=str, default="a", choices=["a", "b", "c", "movie", "cat"], help='stream selection')
    parser.add_argument('--output_folder', type=str, default=default_params['output_folder'], help='additional output folder')
    parser.add_argument('--custom_playback', type=str, default=default_params['custom_playback'], help="flag to alter the stream playback: 'x:y' means x minutes of standard playback and y minutes interleaved with y minutes of pause, repeated")
    parser.add_argument('--verbose', type=str, default=default_params['verbose'], choices=["yes","no"])
    parser.add_argument('--seed', type=int, default=1234, help='seed for neural network weights initialization')
    args_cmd = parser.parse_args()
    run_exp(args_cmd)

if __name__ == '__main__':
    main()