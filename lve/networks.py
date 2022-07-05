from torch import nn as nn
import torch.nn.functional as F
import torch
from torch.nn.init import kaiming_normal_, constant_
import lve.flownet_util as flownet_util
from torchvision import transforms, datasets, models

import os
import torchvision
import cv2


#from torchsummary import summary

class NetworkFactory:
    @staticmethod
    def createEncoder(options):
        if options["architecture"] == "standard":
            return Encoder(options)
        elif options["architecture"] == "flownets" or options["architecture"] == "sota-flownets":
                return FlowNetSPredictor(options)
        elif options["architecture"] == "resunetof":
            return ResUnetOfPredictor(options)
        elif options["architecture"] == "ndconvof":
            return NdConvOfPredictor(options)
        elif options["architecture"] == "dilndconvof":
            return DilNdConvOfPredictor(options)
        elif options["architecture"] == "identity":
            return IdentityEncoder(options)
        else:
            raise AttributeError(f"Architecture {options['architecture']} unknown.")


class Encoder(nn.Module):

    def __init__(self, options):
        super(Encoder, self).__init__()
        self.options = options
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=options['c'], out_channels=8, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=options['num_what'], kernel_size=7, padding=3,
                      bias=True),
            nn.Sigmoid(),
        ])

    def forward(self, x):
        # list containing all the neural activations after the application of the non-linearity
        activations = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if "activation" in str(type(self.layers[i])):  # activation modules have this string...watch out
                activations.append(x)
        return x, activations


class IdentityEncoder(nn.Module):

    def __init__(self, options):
        super(IdentityEncoder, self).__init__()
        self.options = options

    def forward(self, x):
        # list containing all the neural activations after the application of the non-linearity
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        return x, None


class NdConvOfPredictor(nn.Module):

    def __init__(self, options):
        super(NdConvOfPredictor, self).__init__()
        self.options = options
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=options['c'], out_channels=32, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=5, padding=2, bias=True),
        )

    def forward(self, x):
        return self.layers(x), None


class DilNdConvOfPredictor(nn.Module):
    def __init__(self, options):
        super(DilNdConvOfPredictor, self).__init__()
        self.options = options
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=options['c'], out_channels=32, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=4, bias=True, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, padding=4, bias=True, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=4, bias=True, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=5, padding=4, bias=True, dilation=2),
        )

    def forward(self, x):
        return self.layers(x), None



#####
base_model = models.resnet18(pretrained=False)


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


def recursive_avoid_bn(base_model):
    for id, (name, child_model) in enumerate(base_model.named_children()):
        if isinstance(child_model, nn.BatchNorm2d):
            setattr(base_model, name, nn.Identity())

    for name, immediate_child_module in base_model.named_children():
        recursive_avoid_bn(immediate_child_module)

def recursive_avoid_maxpool(base_model):
    for id, (name, child_model) in enumerate(base_model.named_children()):
        if isinstance(child_model, nn.MaxPool2d):
            setattr(base_model, name, nn.AvgPool2d(kernel_size=child_model.kernel_size,
                                                   stride=child_model.stride,
                                                   padding=child_model.padding,
                                                   ceil_mode=child_model.ceil_mode))

    for name, immediate_child_module in base_model.named_children():
        recursive_avoid_maxpool(immediate_child_module)

def fix_first_layer(x, c):
    orig_conv = x[0]
    x[0] = torch.nn.Conv2d(in_channels=c, out_channels=orig_conv.out_channels,
                                           kernel_size=orig_conv.kernel_size, stride=orig_conv.stride, groups=orig_conv.groups,
                                           dilation=orig_conv.dilation, padding_mode=orig_conv.padding_mode,
                                           padding=orig_conv.padding, bias=orig_conv.bias)
    return x


def recursive_reduce_num_filters(base_model, factor=2):
        for id, (name, child_model) in enumerate(base_model.named_children()):
            if isinstance(child_model, nn.Conv2d):
                orig_conv = child_model
                new_conv = torch.nn.Conv2d(in_channels=int(orig_conv.in_channels // factor),
                                                            out_channels=int(orig_conv.out_channels // factor),
                                                            kernel_size=orig_conv.kernel_size, stride=orig_conv.stride, groups=orig_conv.groups,
                                                            dilation=orig_conv.dilation, padding_mode=orig_conv.padding_mode,
                                                            padding=orig_conv.padding, bias=orig_conv.bias)
                setattr(base_model, name, new_conv)

        for name, immediate_child_module in base_model.named_children():
            recursive_reduce_num_filters(immediate_child_module, factor)

# from https://github.com/usuyama/pytorch-unet/blob/master/pytorch_resnet18_unet.ipynb
class ResNetUNet(nn.Module):

    def __init__(self, n_class, batch_norm, c=3, maxpool=True):
        super().__init__()

        if not batch_norm:
            recursive_avoid_bn(base_model)
        if not maxpool:
            recursive_avoid_maxpool(base_model)

        self.c = c
        self.base_layers = list(base_model.children())
        self.base_layers = fix_first_layer(self.base_layers, c)

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(c, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        # repeat grey channel for rgb
        if input.shape[1] == 1 and self.c == 3:
            input = input.repeat(1, 3, 1, 1)

        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

class ResNetUNet2(nn.Module):

    def __init__(self, n_out, batch_norm, c=3, maxpool=True):
        super().__init__()

        if not batch_norm:
            recursive_avoid_bn(base_model)
        if not maxpool:
            recursive_avoid_maxpool(base_model)

        factor = 2
        recursive_reduce_num_filters(base_model, factor)

        self.c = c
        self.base_layers = list(base_model.children())
        self.base_layers = fix_first_layer(self.base_layers, c)

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64 // factor, 64 // factor, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64 // factor, 64 // factor, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128 // factor, 128 // factor, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256 // factor, 256 // factor, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512 // factor, 512 // factor, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu((256 + 512) // factor, 512 // factor, 3, 1)
        self.conv_up2 = convrelu((128 + 512) // factor, 256 // factor, 3, 1)
        self.conv_up1 = convrelu((64 + 256) // factor, 256 // factor, 3, 1)
        self.conv_up0 = convrelu((64 + 256) // factor, 128 // factor, 3, 1)

        self.conv_original_size0 = convrelu(c, 64 // factor, 3, 1)
        self.conv_original_size1 = convrelu(64 // factor, 64 // factor, 3, 1)
        self.conv_original_size2 = convrelu((64 + 128) // factor, 64 // factor, 3, 1)

        self.conv_last = nn.Conv2d(64 // factor, n_out, 1)

    def forward(self, input):
        # repeat grey channel for rgb
        if input.shape[1] == 1 and self.c == 3:
            input = input.repeat(1, 3, 1, 1)

        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

class ResUnetOfPredictor(nn.Module):
    def __init__(self, options):
        super(ResUnetOfPredictor, self).__init__()
        self.options = options
        self.layers = ResNetUNet2(n_out=2, batch_norm=False, c=options['c'],
                                 maxpool=options['maxpool'] if 'maxpool' in options else True)

    def forward(self, x):
        return self.layers(x), None




class FlowNetS(nn.Module):
    expansion = 1

    def __init__(self,c,batchNorm=True):
        super(FlowNetS,self).__init__()
        self.c = c
        self.batchNorm = batchNorm
        self.conv1   = flownet_util.conv(self.batchNorm,  self.c,   64, kernel_size=7, stride=2)
        self.conv2   = flownet_util.conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = flownet_util.conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = flownet_util.conv(self.batchNorm, 256,  256)
        self.conv4   = flownet_util.conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = flownet_util.conv(self.batchNorm, 512,  512)
        self.conv5   = flownet_util.conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = flownet_util.conv(self.batchNorm, 512,  512)
        self.conv6   = flownet_util.conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = flownet_util.conv(self.batchNorm,1024, 1024)

        self.deconv5 = flownet_util.deconv(1024,512)
        self.deconv4 = flownet_util.deconv(1026,256)
        self.deconv3 = flownet_util.deconv(770,128)
        self.deconv2 = flownet_util.deconv(386,64)

        self.predict_flow6 = flownet_util.predict_flow(1024)
        self.predict_flow5 = flownet_util.predict_flow(1026)
        self.predict_flow4 = flownet_util.predict_flow(770)
        self.predict_flow3 = flownet_util.predict_flow(386)
        self.predict_flow2 = flownet_util.predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = flownet_util.crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = flownet_util.crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = flownet_util.crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = flownet_util.crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = flownet_util.crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = flownet_util.crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = flownet_util.crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = flownet_util.crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        final_upsample = torch.nn.Upsample(x.size()[-2:])
        flow2 = final_upsample(flow2)
        return flow2

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class FlowNetSPredictor(Encoder):
    def __init__(self, options):
        super(Encoder, self).__init__()
        self.options = options
        self.layers = FlowNetS(c=options['c'], batchNorm=False)


    def forward(self, x):
        return self.layers(x), None
