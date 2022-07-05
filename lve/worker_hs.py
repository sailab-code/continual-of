import os
import numpy as np
from random import randint, uniform, randrange
import lve
import torch
import cv2
import time

from lve.colof import SubsamplingPolicy
from lve.utils import backward_warp
from collections import OrderedDict
from lve.net_hs import compute_motion_mask, compute_reconstruction_acc


class WorkerHS(lve.Worker):

    def __init__(self, w, h, c, fps, ins, options):
        super().__init__(w, h, c, fps, options)  # do not forget this

        # if the device name ends with 'b' (e.g., 'cpub'), the torch benchmark mode is activated (usually keep it off)
        if options["device"][-1] == 'b':
            self.device = torch.device(options["device"][0:-1])
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device(options["device"])

        # enforcing a deterministic behaviour, when possible
        if 'deterministic' in options and options['deterministic']:
            torch.backends.cudnn.deterministic = True
            torch.set_deterministic(True)

        # setting up seeds for random number generators
        seed = int(time.time()) if options["seed"] < 0 else int(options["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)

        # registering supported commands (commands that are triggered by the external visualizer/interface)

        # saving a shortcut to the neural network options
        self.net_options = self.options["net"]
        self.net_options["w"] = self.w
        self.net_options["h"] = self.h

        # defining processors
        self.ins = ins
        self.optical_flow = lve.OpticalFlowCV(backward=False)

        self.net = lve.NetHS(self.net_options, self.device, self).to(self.device)
        print('Initialized neural network with weights norm:', self.net.compute_weights_norm())

        # neural network optimizer
        self.__lr = self.net_options["step_size"]
        if self.__lr < 0.:  # hack
            if  self.net_options['weight_decay'] > 0:
                self.net_optimizer = torch.optim.AdamW(self.net.parameters(),
                                                      lr=-self.__lr, weight_decay = self.net_options['weight_decay'])
            else:
                self.net_optimizer = torch.optim.Adam(self.net.parameters(), lr=-self.__lr)
        else:
            self.net_optimizer = torch.optim.SGD(self.net.parameters(), lr=self.__lr,
                                                 weight_decay = self.net_options['weight_decay'])
        self.__freeze = self.net_options["freeze"]
        if self.__freeze:
            self.net.requires_grad_(False)

        self.subsampling_policy = SubsamplingPolicy.create_from_options(
            self.net_options['subsampling_updates']
        )

        # misc
        self.__frame = None
        self.__old_frame = None
        self.__prev_frame = None
        self.__flow = None
        self.__frame_embeddings = None
        self.__predicted_motion_mask = None
        self.__motion_mask = None
        self.__motion = None
        self.__warped = None
        self.__diff_warping = None
        self.__motion_needed = True

        self.__unsup_loss = torch.tensor(-1.).to(self.device)
        self.__stats = OrderedDict([('loss', -1.), ('loss_smoothness', -1.), ('loss_invariance', -1.)])  # placeholders

    def reset(self):
        self.__frame = None
        self.__old_frame = None
        self.__prev_frame = None
        self.__flow = None
        self.__motion = None
        self.__warped = None

    def compute_hs_flow_iteratively(self, frame, old_frame, I_t, I_x, I_y, lambda_s, niter=200):
        u = torch.autograd.Variable(torch.zeros(1, 1, frame.shape[0], frame.shape[1]), requires_grad=True).to(
            self.device)
        v = torch.autograd.Variable(torch.zeros(1, 1, frame.shape[0], frame.shape[1]), requires_grad=True).to(
            self.device)
        f = self.net.hs_filter.view((1, 1, 3, 3))
        for _ in range(niter):
            u_bar = torch.nn.functional.conv2d(u, f, stride=1, padding=1)
            v_bar = torch.nn.functional.conv2d(v, f, stride=1, padding=1)
            der = (I_x * u_bar + I_y * v_bar + I_t) / (lambda_s ** 2 + I_x ** 2 + I_y ** 2)
            # print(torch.mean(der))
            u = u_bar - I_x * der
            v = v_bar - I_y * der
        return torch.cat((u, v), dim=1)

    def process_frame(self, frame, of=None, supervisions=None, foa=None):

        # data returned by the call below and their types are:
        # frame: input frame (torch tensor, 1 x c x h x w, better keep the dummy batch dimension here),
        # motion: optical flow (torch tensor, 1 x 2 x h x w, better keep the dummy batch dimension here),
        frame, motion = \
            self.__compute_missing_data_and_convert(frame, of, foa, supervisions,
                                                    fix_motion_v='fix_flow_v' in self.options and self.options['fix_flow_v'],
                                                    fix_motion_u='fix_flow_u' in self.options and self.options['fix_flow_u'])

        old_frame = self.__old_frame if self.__old_frame is not None else frame

        flow, b_prime, image_dx, image_dy = self.net(frame, old_frame)
        flow_std = (torch.std(flow, dim=(2,3), keepdim=True)[0].squeeze() ** 2).sum().sqrt()

        if 'infer_bw' in self.net_options and self.net_options['infer_bw']:
            backward_flow, _, _, _ = self.net(frame=old_frame, old_frame=frame)
        if 'occlusions' in self.net_options and self.net_options['occlusions'] == 'brox':
            # Resampled backward flow at forward flow locations.
            backward_flow_resampled = backward_warp(frame=backward_flow, displacement=flow)

            # Compute occlusions based on forward-backward consistency.
            fb_sq_diff = torch.sum((flow + backward_flow_resampled) ** 2, dim=1, keepdim=True)
            fb_sum_sq = torch.sum(
                flow ** 2 + backward_flow_resampled ** 2, dim=1, keepdim=True)
            occlusion_mask = (fb_sq_diff > 0.01 * fb_sum_sq + 0.5)

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
            self.__unsup_loss =  computed_unsup_metrics['hs_loss']
        else:
            raise ValueError('no valid training loss')

        self.__flow = flow
        self.__motion = motion

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

        flow_np = flow.detach().squeeze().permute(1, 2, 0).cpu().numpy()

        motion_mask_np = motion_mask.detach().float().squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
        predicted_motion_mask_np = predicted_motion_mask.detach().float().squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
        diff_motion_mask_np = diff_motion_mask.detach().float().squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
        self.__predicted_motion_mask = predicted_motion_mask
        self.__motion_mask = motion_mask
        self.__warped = warped
        self.__diff_warping = diff_warping

        outputs_dic = {"motion": of[0],
                          "predicted_motion": flow_np,
                          "ihs_motion": ihs_flow.detach().cpu().numpy(),
                          "warped": warped.detach().cpu().numpy(), # - self.__old_frame.detach().cpu().numpy(),
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
                          "tb.worker": self.__stats}

        if 'infer_bw' in self.net_options and self.net_options['infer_bw']:
            backward_flow_np = backward_flow.detach().squeeze().permute(1, 2, 0).cpu().numpy()
            outputs_dic.update({"predicted_backward_motion": backward_flow_np})

        if 'occlusions' in self.net_options and self.net_options['occlusions'] != 'no':
            occlusion_mask_np = occlusion_mask.detach().float().squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
            outputs_dic.update({"occlusion_mask": occlusion_mask_np})

        self.add_outputs(outputs_dic, batch_index=0)  # tensorboard

        # storing data to be used in the next frame or needed to handle a supervision given through the visualizer
        self.__prev_frame = self.__old_frame
        self.__old_frame = frame

    def update_stats(self, d):
        self.__stats.update(d)

    def update_model_parameters(self):
        update_flag = False
        # check if the freeze option was changed while the agent was running (and react to such change)
        if self.__freeze != self.net_options['freeze']:
            self.__freeze = self.net_options['freeze']
            if self.__freeze:
                self.net.requires_grad_(False)
            else:
                self.net.requires_grad_(True)  # we still have not computed gradients, better 'return'
            return False

        # if frozen, nothing to do here
        if self.__freeze:
            return False

        # completing the loss function
        loss = self.__unsup_loss

        if loss > 0 and self.net_options['architecture'] not in ['none', 'none-ihs']:
            update_flag = self.subsampling_policy is None or self.subsampling_policy.is_active(self.__old_frame, self.__prev_frame, self.__flow, self.ins.get_last_frame_number())
            if update_flag:
                # computing gradients
                loss.backward()

                # update step
                self.net_optimizer.step()
            self.net.zero_grad()

        # check if learning rate was changed (hot)
        if self.__lr != self.net_options['step_size']:
            self.__lr = self.net_options['step_size']
            if self.__lr < 0.:
                if self.net_options['weight_decay'] > 0:
                    self.net_optimizer = torch.optim.AdamW(self.net.parameters(), lr=-self.__lr,
                                                           weight_decay=self.net_options['weight_decay'])
                else:
                    self.net_optimizer = torch.optim.Adam(self.net.parameters(),
                                                          lr=-self.__lr)
            else:
                self.net_optimizer = torch.optim.SGD(self.net.parameters(), lr=self.__lr,
                                                     weight_decay=self.net_options['weight_decay'])

        return update_flag

    def load(self, model_folder):
        worker_model_folder = model_folder + os.sep

        # loading neural network weights
        network_data = torch.load(worker_model_folder + "net.pth", map_location=self.device)
        if self.net_options['architecture'] == 'sota-flownets':
            print(network_data['arch'])
            self.net.model.layers.load_state_dict(network_data['state_dict'])
            self.net_options['div_flow'] = network_data['div_flow'] if 'div_flow' in network_data.keys() else 20.0
        else:
            self.net.load_state_dict(network_data)

    def save(self, model_folder):
        worker_model_folder = model_folder + os.sep
        if not os.path.exists(worker_model_folder):
            os.makedirs(worker_model_folder)
        print(worker_model_folder)
        # saving neural network weights
        torch.save(self.net.state_dict(), worker_model_folder + "net.pth")

        # saving worker-status related tensors
        torch.save({}, worker_model_folder + "worker.pth")

        # saving other parameters
        lve.utils.save_json(worker_model_folder + "worker.json",
                            {})

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

    def get_output_types(self):
        output_types = {  # the output element "frames" is already registered by default
            "motion": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "warped": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "ihs_motion": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "predicted_motion": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "predicted_backward_motion": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "fake_motion": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "occlusion_mask": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
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
