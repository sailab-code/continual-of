import os
import pathlib

import numpy as np
from random import randint, uniform, randrange
import lve
import torch
import cv2
import time
import copy
import sys

from lve import Worker
from lve.colof import SubsamplingPolicy
from lve.utils import backward_warp
from collections import OrderedDict
from lve.net_hs import compute_motion_mask, compute_reconstruction_acc
from settings import smurf_parent_directory

sys.path.append(smurf_parent_directory)

from smurf import smurf_flags  # pylint:disable=unused-import
from smurf import smurf_plotting
from smurf import smurf_net
from smurf import smurf_evaluator
import tensorflow as tf

class WorkerSmurf(Worker):
    def __init__(self, w, h, c, fps, ins, options):
        super().__init__(w, h, c, fps, options)  # do not forget this

        self.optical_flow = lve.OpticalFlowCV(backward=False)
        self.smurf_height = 296
        self.smurf_width = 696
        self.smurf = None
        self.ins = ins
        self.__frame = None
        self.__old_frame = None
        self.__old_frame_np = None
        self.__prev_frame = None
        self.__flow = None
        self.__frame_embeddings = None
        self.__motion_needed = True

        # if the device name ends with 'b' (e.g., 'cpub'), the torch benchmark mode is activated (usually keep it off)
        if options["device"][-1] == 'b':
            self.device = torch.device(options["device"][0:-1])
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device(options["device"])

        options_net = dict(options['net'])
        options_net['architecture'] = 'none'
        self.net_options = options_net

        physical_devices = tf.config.list_physical_devices('GPU')
        for i in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[i], True)
        self.net = lve.NetHS(options_net, self.device).to(self.device)

        self.__stats = OrderedDict([('loss', -1.), ('loss_smoothness', -1.), ('loss_invariance', -1.)])  # placeholders

    def update_stats(self, d):
        self.__stats.update(d)

    def reset(self):
        self.__frame = None
        self.__old_frame = None
        self.__old_frame_np = None
        self.__prev_frame = None
        self.__flow = None
        self.__motion = None

    def build_network(self, ckpt):
        self.smurf = smurf_net.SMURFNet(
            checkpoint_dir=ckpt,
            optimizer='adam',
            dropout_rate=0.,
            feature_architecture="raft",
            flow_architecture="raft",
            size=(1, self.smurf_height, self.smurf_width),
            occlusion_estimation='wang',
            smoothness_at_level=2,
            use_float16=True,
        )

    def call_smurf(self, frame, old_frame):
        flow_forward = self.smurf.infer(
            old_frame, frame, input_height=296, input_width=696,
            infer_occlusion=False, infer_bw=False)

        ff = flow_forward.numpy() # 256*256*2

        flow = torch.tensor(ff).to(self.device).permute(2, 0, 1).unsqueeze(0) # it puts channel in front 2x256x256
        flow = torch.flip(flow, dims=(1,)) # it flips the uv dim (smurf)

        flow_np = np.flip(ff, axis=2)
        return flow, flow_np

    def process_frame(self, frame, of=None, supervisions=None, foa=None):
        frame_np = frame[0]

        old_frame_np = self.__old_frame_np if self.__old_frame_np is not None else frame_np
        flow, flow_np = self.call_smurf(frame_np, old_frame_np)
        flow_std = (torch.std(flow, dim=(2, 3), keepdim=True)[0].squeeze() ** 2).sum().sqrt()

        frame, motion = \
            self.__compute_missing_data_and_convert(frame, of, foa, supervisions,
                                                    fix_motion_v='fix_flow_v' in self.options and self.options['fix_flow_v'],
                                                    fix_motion_u='fix_flow_u' in self.options and self.options['fix_flow_u']
                                                    )
        old_frame = self.__old_frame if self.__old_frame is not None else frame

        _, b_prime, image_dx, image_dy = self.net(frame, old_frame)

        warped = backward_warp(frame=frame, displacement=flow)
        computed_unsup_metrics = self.net.compute_unsup_loss(frame,
                                                             old_frame,
                                                             flow,
                                                             b_prime,
                                                             image_dx,
                                                             image_dy,
                                                             warped)

        motion_mask = compute_motion_mask(flow=motion, threshold=self.net_options['ground_truth_motion_threshold'])
        predicted_motion_mask = compute_motion_mask(flow=flow, threshold=self.net_options['output_motion_threshold'])
        diff_motion_mask = torch.logical_xor(motion_mask, predicted_motion_mask)
        diff_warping, recon_acc = compute_reconstruction_acc(warped_frame=warped, old_frame=old_frame,
                                               motion_mask=motion_mask, thresholds=self.net.options['recon_linf_thresh'])

        if self.net_options['training_loss'] == 'photo_and_smooth':
            self.__unsup_loss = computed_unsup_metrics['photo_and_smooth_loss']
        elif self.net_options['training_loss'] == 'hs':
            self.__unsup_loss = computed_unsup_metrics['hs_loss']
        else:
            raise ValueError('no valid training loss')

        self.__flow = flow
        self.__motion = motion
        self.__warped = warped
        self.__diff_warping = diff_warping

        if self.net.options['compute_ihs']:
            ihs_flow = self.compute_hs_flow_iteratively(frame[0, 0], old_frame[0, 0], I_t=b_prime, I_x=image_dx,
                                                        I_y=image_dy, lambda_s=self.net.options['lambda_s'])
        else:
            ihs_flow = torch.zeros_like(flow)

        self.__stats.update({"loss": self.__unsup_loss.item(),
                             "hs_loss": computed_unsup_metrics['hs_loss'].item(),
                             "photo_and_smooth_loss": computed_unsup_metrics['photo_and_smooth_loss'].item(),
                             "loss_smoothness": computed_unsup_metrics['smoothness_term'].item(),
                             "loss_invariance": computed_unsup_metrics['invariance_term'].item(),
                             "loss_photometric": computed_unsup_metrics['photo_term'].item(),
                             "recon_acc": recon_acc,
                             "flow_std": flow_std.item(),
                             "largest_flow_x": torch.max(torch.abs(motion[0, 0, :, :])).item(),
                             "largest_flow_y": torch.max(torch.abs(motion[0, 1, :, :])).item(),
                             "largest_predicted_flow_x": torch.max(torch.abs(flow[0, 0, :, :])).item(),
                             "largest_predicted_flow_y": torch.max(torch.abs(flow[0, 1, :, :])).item(),
                             "largest_ihs_flow_x": torch.max(torch.abs(ihs_flow[0, 0, :, :])).item() if
                             self.net.options['compute_ihs'] else 0.0,
                             "largest_ihs_flow_y": torch.max(torch.abs(ihs_flow[0, 1, :, :])).item() if
                             self.net.options['compute_ihs'] else 0.0
                             })


        motion_mask_np = motion_mask.detach().float().squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
        predicted_motion_mask_np = predicted_motion_mask.detach().float().squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
        diff_motion_mask_np = diff_motion_mask.detach().float().squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
        self.__predicted_motion_mask = predicted_motion_mask
        self.__motion_mask = motion_mask

        self.add_outputs({"motion": of[0],
                          "predicted_motion": flow_np,
                          # "fake_motion": fake_flow_np,
                          "ihs_motion": ihs_flow.detach().cpu().numpy(),
                          "warped": warped.detach().cpu().numpy(),  # - self.__old_frame.detach().cpu().numpy(),
                          "image_dx": image_dx.detach().cpu().numpy(),
                          "image_dy": image_dy.detach().cpu().numpy(),
                          "b_prime": b_prime.detach().cpu().numpy(),
                          "flow_x": flow.detach().cpu().numpy()[0, 0, :, :],
                          "flow_y": flow.detach().cpu().numpy()[0, 1, :, :],
                          "motion_mask": motion_mask_np,
                          "predicted_motion_mask": predicted_motion_mask_np,
                          "diff_motion_mask": diff_motion_mask_np,
                          "stats.worker": self.__stats,  # dictionary
                          "logs.worker": list(self.__stats.values()),  # CSV log
                          "tb.worker": self.__stats}, batch_index=0)  # tensorboard

        # storing data to be used in the next frame or needed to handle a supervision given through the visualizer
        self.__prev_frame = self.__old_frame
        self.__old_frame = frame
        self.__old_frame_np = frame_np

    def get_loggable_image_dict(self):
        d = {
            'warped': self.__warped,
            'diff_warping': self.__diff_warping,
            'prev_frame': self.__prev_frame,
            'frame': self.__old_frame, # YES: it has already been updated
            'flow': self.__flow,
            'motion': self.__motion,
            'motion_mask': self.__motion_mask,
            'predicted_motion_mask': self.__predicted_motion_mask
        }
        return d

    def get_warped_frame(self):
        return backward_warp(frame=self.__old_frame, displacement=self.__flow), self.__prev_frame, self.__old_frame, \
               self.__flow, self.__motion, self.__motion_mask, self.__predicted_motion_mask

    def update_model_parameters(self):
        return False

    def load(self, model_folder):
        self.build_network(model_folder)
        self.smurf.update_checkpoint_dir(model_folder)
        self.smurf.restore()

    def save(self, model_folder):
        pass

    def get_output_types(self):
        output_types = {  # the output element "frames" is already registered by default
            "motion": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "warped": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "ihs_motion": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "predicted_motion": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "fake_motion": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "motion_mask": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "predicted_motion_mask": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "diff_motion_mask": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "flow_x": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "flow_y": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "image_dx": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "image_dy": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "b_prime": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "stats.worker": {'data_type': lve.OutputType.JSON, 'per_frame': True},
            "logs.worker": {'data_type': lve.OutputType.TEXT, 'per_frame': False},
            "logs.worker__header": ['frame'] + list(self.__stats.keys())  # first line of CSV
        }
        return output_types

    def print_info(self):
        s = "   worker {"
        i = 0
        for k, v in self.__stats.items():
            if isinstance(v, dict): continue
            s += (k + (": {0:.3e}".format(v) if abs(v) >= 1000 else ": {0:.3f}".format(v)))
            if (i + 1) % 7 == 0:
                if i < len(self.__stats) - 1:
                    s += ",\n           "
                else:
                    s += "}"
            else:
                if i < len(self.__stats) - 1:
                    s += ", "
                else:
                    s += "}"
            i += 1

        print(s)


    def __compute_missing_data_and_convert(self, batch_frames_np_uint8, batch_motion_np_float32,
                                           batch_foa_np_float32, batch_sup_np, fix_motion_v, fix_motion_u):
        # assumption: data are stored in batches of size 1, i.e., one frame at each time instant
        assert len(batch_frames_np_uint8) == 1, "We are assuming to deal with batches of size 1, " \
                                                "and it does not seem to be the case here!"

        # convert to tensor
        frame_np_uint8 = batch_frames_np_uint8[0]
        frame = lve.utils.np_uint8_to_torch_float_01(frame_np_uint8, device=self.device)

        # grayscale-instance of the input frame
        if not self.frame_is_gray_scale:
            frame_gray_np_uint8 = cv2.cvtColor(frame_np_uint8, cv2.COLOR_BGR2GRAY).reshape(self.h, self.w, 1)
            frame_gray = lve.utils.np_uint8_to_torch_float_01(frame_gray_np_uint8, device=self.device)
        else:
            frame_gray_np_uint8 = frame_np_uint8
            frame_gray = frame

        # optical flow
        if batch_motion_np_float32 is None or batch_motion_np_float32[0] is None:
            if self.__motion_needed:
                motion_np_float32 = self.optical_flow(frame_gray_np_uint8)  # it returns np.float32, h x w x 2
            else:
                motion_np_float32 = np.zeros((self.h, self.w, 2), dtype=np.float32)

            if fix_motion_v: motion_np_float32[..., 1] *= -1
            if fix_motion_u: motion_np_float32[..., 0] *= -1
            motion = lve.utils.np_float32_to_torch_float(motion_np_float32, device=self.device)  # 1 x 2 x h x w

            if batch_motion_np_float32 is not None:
                batch_motion_np_float32[0] = motion_np_float32  # updating, it might be used out of this function
        else:
            motion_np_float32 = batch_motion_np_float32[0]  # h x w x 2

            if fix_motion_v: motion_np_float32[..., 1] *= -1
            if fix_motion_u: motion_np_float32[..., 0] *= -1
            motion = lve.utils.np_float32_to_torch_float(motion_np_float32, device=self.device)  # 1 x 2 x h x w

        return frame, motion
