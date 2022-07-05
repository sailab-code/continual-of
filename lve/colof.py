import abc
import collections
import json
import os
import time

import wandb
import numpy as np
import torch
from torch import nn

from lve import InputStream
from lve.utils import visualize_flows, plot_standard_heatmap

MODEL_SELECTION_EXP_TYPE = 'model-selection'
FULL_EXP_TYPE = 'full'
default_params = {
    'training_windows': 15,
    'step_size': -0.001,
    'weight_decay': 0,
    'lambda_s': 0.00005,
    'charb_eps': 0.001,
    'charb_alpha': 0.5,
    'recon_linf_thresh': "0.050,0.025",
    'subsampling_updates': "0",
    'force_gray': "no",
    'device': "cuda",
    'compute_ihs': "no",
    'iter_ihs': 0,
    'warm_ihs': "no",
    'save': 'no',
    'save_output': 'no',
    'freeze': 'no',
    'load': None,
    'exp_type': MODEL_SELECTION_EXP_TYPE,
    'training_loss': "photo_and_smooth",
    'infer_bw': 'no',
    'occlusions': 'no',
    'port': 0,
    'arch': 'larger_standard',
    'net_flow_input_type': 'implicit',
    'output_folder': 'output_folder',
    'custom_playback': None,
    'deterministic': 'no',
    'seed': 1234,
    'verbose': 'no'
}


class WandbSaveCallback:
    def __init__(self, worker, save_running=False, fix_flow_v=False, log_windows=None):
        self.worker = worker
        self.prefix = ''
        self.save_running = save_running
        self.fix_flow_v = fix_flow_v
        self.buffer_keys = ['moving_f1_w', 'moving_f1_r', 'recon_acc_r', 'recon_acc_w', 'photo_and_smooth_w', 'photo_w', 'hs_smoothness_w', 'flow_std_w']
        self.buffer = []
        self.log_windows = log_windows
        self.i = 0

    @property
    def setting(self):
        return self.prefix.replace("/","").replace("_trial", "")

    def __call__(self, stats_metrics, stats_worker):
        save_images = True if self.log_windows is None or self.i in self.log_windows else False
        rm_fnames = []

        image_dict = self.worker.get_loggable_image_dict()

        moving_labels = {0: 'still', 1: 'moving'}
        wandb_dict = {
            'updated_w': stats_metrics['data']["whole_frame"]["window"]["updated"],
            'photo_w': stats_metrics['data']["whole_frame"]["window"]["photo"],
            'photo_and_smooth_w': stats_metrics['data']["whole_frame"]["window"]["photo_and_smooth"],
            'hs_w': stats_metrics['data']["whole_frame"]["window"]["hs"],
            'hs_invariance_w': stats_metrics['data']["whole_frame"]["window"]["hs_invariance"],
            'hs_smoothness_w': stats_metrics['data']["whole_frame"]["window"]["hs_smoothness"],
            'largest_predicted_flow_x': stats_worker['data']["largest_predicted_flow_x"],
            'largest_predicted_flow_y': stats_worker['data']["largest_predicted_flow_y"],
            'flow_std_w': stats_metrics['data']["whole_frame"]["window"]["flow_std"],
            'recon_acc_w': stats_metrics['data']["whole_frame"]["window"]["recon_acc"],
            'moving_f1_w': stats_metrics['data']["whole_frame"]["window"]["moving_f1"],
            'moving_acc_w': stats_metrics['data']["whole_frame"]["window"]["moving_acc"],
            'moving_cm_w': stats_metrics['data']["whole_frame"]["window"]["moving_cm"],
            'moving_precision_w': stats_metrics['data']["whole_frame"]["window"]["moving_precision"],
            'moving_recall_w': stats_metrics['data']["whole_frame"]["window"]["moving_recall"]
        }
        w, b = self.worker.net.compute_weights_norm()
        wandb_dict['weights_norm'] = w
        wandb_dict['bias_norm'] = b
        if 'updated' in stats_worker['data']: wandb_dict['updated'] = int(stats_worker['data']['updated'])
        if save_images:
            wandb_dict.update({
                'warped': wandb.Image(torch.mean(image_dict['warped'][0, :], dim=0)),
                'prev_frame': wandb.Image(torch.mean(image_dict['prev_frame'][0, :], dim=0)),
                'frame': wandb.Image(torch.mean(image_dict['frame'][0, :], dim=0), masks={
                    "prediction":
                        {
                            "mask_data": image_dict['predicted_motion_mask'][0,0].cpu().numpy().astype(int),
                            "class_labels": moving_labels
                        },
                    "ground_truth":
                        {
                            "mask_data": image_dict['motion_mask'][0,0].cpu().numpy().astype(int),
                            "class_labels": moving_labels
                        }
                })
            })
            flow_dic, rm_fnames_ = visualize_flows(image_dict['flow'], prefix='flow')
            wandb_dict.update(flow_dic)
            rm_fnames += rm_fnames_

            motion_dic, rm_fnames_ = visualize_flows(image_dict['motion'], prefix='ground_truth')
            wandb_dict.update(motion_dic)
            rm_fnames += rm_fnames_

            diff_warping_dic, rm_fnames_ = plot_standard_heatmap(image_dict['diff_warping'], name='diff_warping')
            wandb_dict.update(diff_warping_dic)
            rm_fnames += rm_fnames_

        if self.save_running:
            wandb_dict.update({
                'photo_r': stats_metrics['data']["whole_frame"]["running"]["photo"],
                'recon_acc_r': stats_metrics['data']["whole_frame"]["running"]["recon_acc"],
                'moving_f1_r': stats_metrics['data']["whole_frame"]["running"]["moving_f1"],
                'moving_acc_r': stats_metrics['data']["whole_frame"]["running"]["moving_acc"],
                'moving_precision_r': stats_metrics['data']["whole_frame"]["running"]["moving_precision"],
                'moving_recall_r': stats_metrics['data']["whole_frame"]["running"]["moving_recall"],
                'moving_cm_r': stats_metrics['data']["whole_frame"]["running"]["moving_cm"],
                'flow_std_r': stats_metrics['data']["whole_frame"]["running"]["flow_std"]
            })

        if self.prefix != '':
            wandb_dict = {self.prefix+k: v for k,v in wandb_dict.items()}
        wandb.log(wandb_dict)

        for x in rm_fnames:
            os.remove(x)

        self.buffer.append({k: wandb_dict[self.prefix+k] for k in self.buffer_keys if self.prefix+k in wandb_dict})
        self.i += 1

class SubsamplingPolicy(metaclass=abc.ABCMeta):

    def __init__(self, options):
        self.options = options
        self.i = 0

    @staticmethod
    def create_from_options(options):
        if options is None or str(options) == "0": return None
        options = str(options)
        pcs = options.split(":")
        if len(pcs) == 1 and pcs[0].replace('.','',1).isdigit(): return StaticSubsamplingPolicy(float(options))
        elif pcs[0] == 'avgflow':
            if len(pcs) == 4:
                return AvgPredictedFlowSubsamplingPolicy(float(pcs[1]), float(pcs[2]), int(pcs[3]))
            elif len(pcs) == 5:
                return AvgPredictedFlowSubsamplingPolicy(float(pcs[1]), float(pcs[2]), int(pcs[3]), int(pcs[4]))
        elif pcs[0] == 'avgflowhistory':
            if len(pcs) == 3:
                return AvgPredictedFlowHistorySubsamplingPolicy(float(pcs[1]), int(pcs[2]))
            elif len(pcs) == 4:
                return AvgPredictedFlowHistorySubsamplingPolicy(float(pcs[1]), int(pcs[2]), int(pcs[3]))
        elif pcs[0] == "avgdiff":
            return AvgFrameDiffSubsamplingPolicy(float(pcs[1]), int(pcs[2]))
        else:
            raise ValueError("Invalid subsampling policy type")

    @abc.abstractmethod
    def is_active(self, frame, prev_frame, flow, i):
        pass

class StaticSubsamplingPolicy(SubsamplingPolicy):
    def __init__(self, n):
        self.n = n

    def is_active(self, frame, prev_frame, flow, i):
        return i % self.n == 0

class AvgPredictedFlowSubsamplingPolicy(SubsamplingPolicy):
    def __init__(self, threshold, ratio, warmup, force_update_every=150):
        self.threshold = threshold
        self.l = collections.deque(maxlen=1)
        self.warmup = warmup
        self.ratio = ratio
        self.force_update_every = force_update_every

    def is_active(self, frame, prev_frame, flow, i):
        if i < self.warmup or \
                (self.force_update_every > 0 and (i % self.force_update_every) == 0):
            return True
        moving_rate = (flow.norm(dim=1) > self.threshold).float().mean().item()
        #print(moving_rate)
        self.l.append(moving_rate)
        return bool(np.mean(self.l) > self.ratio)


class AvgPredictedFlowHistorySubsamplingPolicy(SubsamplingPolicy):
    def __init__(self, threshold, warmup, force_update_every=150):
        self.threshold = threshold
        self.warmup = warmup
        self.force_update_every = force_update_every
        self.old_avg_flow = None

    def is_active(self, frame, prev_frame, flow, i):
        avg_flow = flow.mean(dim=[2, 3])
        if i < self.warmup or \
                (self.force_update_every > 0 and (i % self.force_update_every) == 0) or \
                self.old_avg_flow is None:
            active_flag = True
        else:
            old_avg_flow_norm = torch.sqrt(torch.sum(self.old_avg_flow ** 2))
            dev = torch.sqrt(((avg_flow - self.old_avg_flow) ** 2).sum())
            dev_ratio = dev / old_avg_flow_norm
            active_flag = dev_ratio.item() > self.threshold
            # print('dev: {:}, old_avg_flow_norm: {:}, dev_ratio: {:}, active:{:}'. format(
            #       dev.item(), old_avg_flow_norm.item(), dev_ratio.item(), active_flag)
            # )

        if active_flag:
            self.old_avg_flow = avg_flow
        return active_flag

class AvgFrameDiffSubsamplingPolicy(SubsamplingPolicy):
    def __init__(self, threshold, warmup):
        self.threshold = threshold
        self.l = collections.deque(maxlen=1)
        self.warmup = warmup

    def is_active(self, frame, prev_frame, flow, i):
        if i < self.warmup: return True
        avg_diff = (frame - prev_frame).norm(dim=1).mean().item()
        #print(avg_diff)
        self.l.append(avg_diff)
        return bool(np.mean(self.l) > self.threshold)

class DummyNet(nn.Module):
    def __init__(self, options):
        super(DummyNet, self).__init__()
        self.options = options
        self.dummy = torch.nn.Parameter(torch.randn(()))
        self.hs_filter = torch.Tensor([[1/12, 1/6, 1/12],
                                       [1/6,    0, 1/6],
                                       [1/12, 1/6, 1/12]]).to(self.dummy.device)

    def forward(self, x):
        n, c, h, w = x.shape
        x = torch.zeros((n, 2, h, w), device=self.dummy.device)

        return x, None

def print_stats(log_dict, fldr, setting, show_f1=False):
    print('-- Final stats for', setting)
    stats = log_dict['logged'][-1]['whole_frame']['running']
    stats['recon_acc'] = {'threshold:' + str(k): v for k, v in stats['recon_acc'].items()}
    final_stats = {'recon_acc': stats['recon_acc']}

    if show_f1:
        final_stats['moving_f1'] = stats['moving_f1']
        print("[running avg] Moving-F1:", final_stats['moving_f1'])

    print("[running avg] Reconstruction accuracy:", final_stats['recon_acc'])

    # dump metrics dict to file
    with open(os.path.join(fldr, 'results_'+setting+'.json'), 'w') as fp:
        json.dump(final_stats, fp)

class IHSWrapper(nn.Module):
    def __init__(self, options):
        super(IHSWrapper, self).__init__()
        self.options = options
        self.dummy = torch.nn.Parameter(torch.randn(()))
        self.hs_filter = torch.Tensor([[1/12, 1/6, 1/12],
                                       [1/6,    0, 1/6],
                                       [1/12, 1/6, 1/12]]).to(self.dummy.device)

    def forward(self, x, old_estimate):
        n, c, h, w = x.shape
        assert self.options['net_flow_input_type'] == 'explicit'
        assert c == 3
        I_t = x[:, 0]
        I_x = x[:, 1]
        I_y = x[:, 2]
        x = self.compute_ihs(I_t, I_x, I_y, self.options['lambda_s'], niter=self.options['iter_ihs'], old_estimate=old_estimate)

        return x, None

    def compute_ihs(self, I_t, I_x, I_y, lambda_s, old_estimate=None, niter=1500):
        n, h, w = I_t.shape
        device = self.dummy.device
        output = torch.zeros(n,2,h,w).to(device)
        for i in range(n):
            if self.options['warm_ihs'] and old_estimate is not None:
                u = old_estimate[i, 0].view(1,1,h,w)
                v = old_estimate[i, 1].view(1,1,h,w)
            else:
                u = torch.zeros(1, 1, h, w).to(device)
                v = torch.zeros(1, 1, h, w).to(device)
            f = self.hs_filter.view((1, 1, 3, 3)).to(device)
            for _ in range(niter):
                u_bar = torch.nn.functional.conv2d(u, f, stride=1, padding=1)
                v_bar = torch.nn.functional.conv2d(v, f, stride=1, padding=1)
                der = (I_x * u_bar + I_y * v_bar + I_t) / (lambda_s ** 2 + I_x ** 2 + I_y ** 2)
                # print(torch.mean(der))
                u = u_bar - I_x * der
                v = v_bar - I_y * der
            output[i] = torch.cat((u, v), dim=1)[0]
        return output


class ColofInputStream(InputStream):
    def __init__(self, input_element, w=None, h=None,
                 fps=None, force_gray=False,
                 repetitions=1, max_frames=None, shuffle=False,
                 frame_op=None, foa_file=None, unity_settings=None, skip_frames=None, custom_playback=None):
        self.custom_playback = custom_playback
        if custom_playback is not None:
            if ":" in custom_playback:
                play, pause = custom_playback.split(":")
                self.custom_playback = (int(25 * float(play) * 60), int(25 * float(pause) * 60))
        self.buffer = {}
        self.i = 0
        super().__init__(input_element, w, h,
                 fps, force_gray,
                 repetitions, max_frames, shuffle,
                 frame_op, foa_file, unity_settings, skip_frames)

    def is_advancing(self):
        if type(self.custom_playback) is tuple and len(self.custom_playback) == 2:
            play, static = self.custom_playback
            t = sum(self.custom_playback)
            x = self.get_last_frame_number() % t
            if x < play:
                return True
            else:
                return False
        return True


    def get_next(self, skip_if_possible=False):
        __last_returned_frame_number = super().get_last_frame_number()
        if __last_returned_frame_number >= self.frames > 0:
            return None, None, None, None  # frame, motion, supervisions, foa
        if 0 < self.max_frames <= __last_returned_frame_number:
            return None, None, None, None  # frame, motion, supervisions, foa

        if self.is_advancing():
            img, of, supervisions, foa = super().get_next(skip_if_possible)
            self.buffer['img'], self.buffer['of'], self.buffer['supervisions'], self.buffer['foa'] = img, of, supervisions, foa
        else:
            if self.buffer['of'] is not None: self.buffer['of'] *= 0.0
            super().set_last_frame_number(1 + super().get_last_frame_number())
        return self.buffer['img'], self.buffer['of'], self.buffer['supervisions'], self.buffer['foa']