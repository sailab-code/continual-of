import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from lve.colof import IHSWrapper, DummyNet
from lve.networks import NetworkFactory

def compute_photometric_loss(f1, f2, charb_eps, charb_alpha):
    return ((f1-f2)**2+charb_eps**2).pow(charb_alpha).mean()

def compute_motion_mask(flow, threshold=0.5):
    mask = torch.norm(flow, p=float('inf'), dim=1, keepdim=True) > threshold
    return mask

def compute_reconstruction_acc(warped_frame, old_frame, motion_mask, thresholds):
    if not warped_frame.shape[0] == 1: raise AssertionError
    diff = torch.norm(old_frame - warped_frame, p=float('inf'), dim=1, keepdim=True)
    diff_masked = diff[motion_mask]
    accs = {threshold:
                1.0 if torch.numel(diff_masked) == 0 else
                torch.mean((diff_masked < threshold).float()).item()
            for threshold in thresholds}

    return diff, accs


class NetHS(nn.Module):

    def __init__(self, options, device, worker=None):
        super(NetHS, self).__init__()

        # keeping track of the network options
        self.options = options
        self.device = device

        # encoding splitting
        self.output_dim = options['output_dim']
        self.net_flow_input_type = options["net_flow_input_type"]

        # model
        if options['architecture'] == 'none':
            self.model = DummyNet(options)
        elif options['architecture'] == 'none-ihs':
            self.model = IHSWrapper(options)
        else:
            self.model = NetworkFactory.createEncoder(options)

        self.sobel_dx_kernel = torch.Tensor([[1 / 2, 0, -1 / 2],
                                             [1, 0, -1],
                                             [1/2, 0, -1/2]]).to(device)
        self.sobel_dy_kernel = torch.Tensor([[1 / 2, 1, 1 / 2],
                                             [0, 0, 0],
                                             [-1/2, -1, -1/2]]).to(device)

        self.hs_dx_kernel = torch.Tensor([[0, 0, 0],
                                          [0, -1 / 4, 1 / 4],
                                          [0, -1 / 4, 1 / 4]]).to(device)

        self.hs_dy_kernel = torch.Tensor([[0, 0, 0],
                                          [0, -1 / 4, -1 / 4],
                                          [0, 1 / 4, 1 / 4]]).to(device)

        self.hs_dt_kernel = torch.Tensor([[0, 0, 0],
                                          [0, 1 / 4, 1 / 4],
                                          [0, 1 / 4, 1 / 4]]).to(device)


        self.hs_filter = torch.Tensor([[1/12, 1/6, 1/12],
                                       [1/6,    0, 1/6],
                                       [1/12, 1/6, 1/12]]).to(device)

        if self.options['color_channels'] == 3:
            self.hs_dt_kernel_f = self.hs_dt_kernel.view((1, 1, 3, 3)).expand(3, -1, -1, -1)
            self.hs_dx_kernel_f = self.hs_dx_kernel.view((1, 1, 3, 3)).expand(3, -1, -1, -1)
            self.hs_dy_kernel_f = self.hs_dy_kernel.view((1, 1, 3, 3)).expand(3, -1, -1, -1)
            self.sobel_dx_kernel_f = self.sobel_dx_kernel.view((1, 1, 3, 3)).expand(2, -1, -1, -1)
            self.sobel_dy_kernel_f = self.sobel_dy_kernel.view((1, 1, 3, 3)).expand(2, -1, -1, -1)
        elif self.options['color_channels'] == 1:
            self.hs_dt_kernel_f = self.hs_dt_kernel.view((1, 1, 3, 3))
            self.hs_dx_kernel_f = self.hs_dx_kernel.view((1, 1, 3, 3))
            self.hs_dy_kernel_f = self.hs_dy_kernel.view((1, 1, 3, 3))
            self.sobel_dx_kernel_f = self.sobel_dx_kernel.view((1, 1, 3, 3))
            self.sobel_dy_kernel_f = self.sobel_dy_kernel.view((1, 1, 3, 3))
        else:
            raise NotImplementedError

        self.hs_dx_kernel_uv = self.hs_dx_kernel.view((1, 1, 3, 3)).expand(2, -1, -1, -1)
        self.hs_dy_kernel_uv = self.hs_dy_kernel.view((1, 1, 3, 3)).expand(2, -1, -1, -1)
        self.sobel_filter_x_uv = self.sobel_dx_kernel.view((1, 1, 3, 3)).expand(2, -1, -1, -1)
        self.sobel_filter_y_uv = self.sobel_dy_kernel.view((1, 1, 3, 3)).expand(2, -1, -1, -1)

        self.worker = worker

    def gradient(self, x, x_=None, type="sobel"):
        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)
        batch_dims = x.shape[0]
        channel_dims = x.shape[1]

        if type == "shift":
            left = x
            right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
            top = x
            bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

            dx, dy = right - left, bottom - top
            # dx will always have zeros in the last column, right-left
            # dy will always have zeros in the last row,    bottom-top
            dx[:, :, :, -1] = 0
            dy[:, :, -1, :] = 0

        elif type == "sobel":
            if channel_dims == 3 or channel_dims == 1: # on frames
                dx = F.conv2d(x, self.sobel_dx_kernel_f, stride=1, padding=1, groups=channel_dims) # TODO check is poosible padding="same"
                dy = F.conv2d(x, self.sobel_dy_kernel_f, stride=1, padding=1, groups=channel_dims)
            elif channel_dims == 2: # on flow
                dx = F.conv2d(x, self.sobel_dx_kernel_uv, stride=1, padding=1, groups=channel_dims)
                dy = F.conv2d(x, self.sobel_dy_kernel_uv, stride=1, padding=1, groups=channel_dims)
            else:
                raise NotImplementedError

        elif type == "hs":
            if channel_dims == 3 or channel_dims == 1: # on frames
                a = self.hs_dx_kernel_f
                b = self.hs_dy_kernel_f
                dx = F.conv2d(x, a, stride=1, padding=1, groups=channel_dims) + \
                     F.conv2d(x_, a, stride=1, padding=1, groups=channel_dims)
                dy = F.conv2d(x, b, stride=1, padding=1, groups=channel_dims) + \
                     F.conv2d(x_, b, stride=1, padding=1, groups=channel_dims)
            elif channel_dims == 2: # on flow
                a = self.hs_dx_kernel_uv
                b = self.hs_dy_kernel_uv
                dx = F.conv2d(x, a, stride=1, padding=1, groups=channel_dims) * 2
                dy = F.conv2d(x, b, stride=1, padding=1, groups=channel_dims) * 2
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return dx, dy

    def time_derivative(self, frame, old_frame, type='difference'):
        batch_dims = frame.shape[0]
        channel_dims = frame.shape[1]

        # it is always computed on frames
        if type == 'difference':
            dt = frame - old_frame
        elif type == 'hs':
            dt = F.conv2d(frame, self.hs_dt_kernel_f, stride=1, padding=1, groups=channel_dims) - \
                 F.conv2d(old_frame, self.hs_dt_kernel_f, stride=1, padding=1, groups=channel_dims)
        return dt

    def forward(self, frame, old_frame):
        # compute gradients
        b_prime = self.time_derivative(frame, old_frame, type='hs')
        dx, dy = self.gradient(frame, x_=old_frame, type='hs')
        # encoding data
        if self.net_flow_input_type == "implicit":
            if self.options['architecture'] == 'sota-flownets':
                # in the case of pretrained flownets
                # preprocessing as described in https://github.com/ClementPinard/FlowNetPytorch/blob/6b409990e17139941685560a4802d73732d76e00/run_inference.py
                assert frame.shape[1] == 3
                offset = torch.tensor([[0.411,0.432,0.45]], device=frame.device)
                offset = offset[:, :, None, None]
                flownet_input = torch.cat([old_frame-offset, frame-offset], 1)
                output, _ = self.model(flownet_input)
                output *= self.options['div_flow']
            else:
                cat_input = torch.cat([frame, old_frame], 1)
                output, _ = self.model(cat_input)
        elif self.net_flow_input_type == "explicit":
            cat_input = torch.cat([b_prime, dx, dy], 1)
            if self.options['architecture'] == 'none-ihs':
                old_flow = None if self.worker is None else self.worker.get_loggable_image_dict()['flow']
                output, _ = self.model(cat_input, old_estimate = old_flow)
            else:
                output, _ = self.model(cat_input)
        else:
            raise NotImplementedError
        return output, b_prime, dx, dy

    def compute_unsup_loss(self, frame, old_frame, flow, b_prime, image_dx, image_dy, warped=None):
        lambda_s = self.options['lambda_s']
        flow_dx, flow_dy = self.gradient(flow, type='hs')

        p1 = image_dx * flow[:, 0, :, :].unsqueeze(1) + image_dy * flow[:, 1, :, :].unsqueeze(1)
        invariance_term = 0.5 * torch.mean((p1 + b_prime) ** 2)
        smoothness_term = 0.5 * torch.mean(flow_dx ** 2 + flow_dy ** 2)
        hs_loss = invariance_term + lambda_s * smoothness_term
        hs_sum_loss = invariance_term + smoothness_term
        photo_term = torch.tensor((0.0)) if warped is None else compute_photometric_loss(old_frame, warped,
                                                                         charb_eps = self.options['charb_eps'],
                                                                         charb_alpha = self.options['charb_alpha'])
        photo_and_smooth_loss = photo_term + lambda_s * smoothness_term

        unsup_loss_dic = {'hs_loss': hs_loss, 'photo_and_smooth_loss': photo_and_smooth_loss,
                          'smoothness_term': smoothness_term, 'invariance_term': invariance_term,
                          'photo_term': photo_term, 'hs_sum_loss': hs_sum_loss}
        return unsup_loss_dic


    def zero_grad(self):
        for param in self.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    if param.grad.grad_fn is not None:
                        param.grad.detach_()
                    else:
                        param.grad.requires_grad_(False)
                    param.grad.zero_()

    def compute_weights_norm(self):
        w = 0.0
        b = 0.0
        for param in self.parameters():
            if param.ndim == 1:
                b += torch.sum(param**2)
            else:
                w += torch.sum(param**2)
        if torch.is_tensor(b): b = b.item()
        if torch.is_tensor(w): w = w.item()
        return w, b

    def print_parameters(self):
        params = list(self.parameters())
        print("Number of tensor params: " + str(len(params)))
        for i in range(0, len(params)):
            p = params[i]
            print("   Tensor size: " + str(p.size()) + " (req. grad = " + str(p.requires_grad) + ")")
