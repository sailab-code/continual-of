import torch
import torch.nn.functional as F
import cv2


class Blur:

    def __init__(self, w, h, c, device=None):
        kernel_size = int(0.25 * min(w, h))
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
        self.__sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.3  # OpenCV rule - 0.5
        self.__channels = c

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

    def __call__(self, img, blur_factor=0.3):
        if blur_factor < 0.0 or blur_factor > 1.0:
            raise ValueError("Invalid blur factor (it must be in [0,1]): " + str(blur_factor))

        if blur_factor > 0.000001:
            scaling = 1.0 - blur_factor
            sigma = blur_factor * self.__sigma
            kernel_size = int(2 * ((sigma - 0.3) / 0.3 + 1) + 1)
            if kernel_size % 2 == 0:
                kernel_size = kernel_size + 1

            if kernel_size > 1:
                padding = int((kernel_size - 1) / 2)

                kernel = self.__get_gaussian_kernel_1d(kernel_size, sigma, self.__channels)  # fake-2D

                if img.dtype != torch.float32:
                    img_float32 = img.to(torch.float32)
                else:
                    img_float32 = img

                c1 = F.conv2d(F.pad(img_float32, (padding,padding,0,0), 'reflect'),
                              kernel, bias=None, padding=0, groups=self.__channels)
                c1 = F.conv2d(F.pad(c1.transpose(2,3), (padding,padding,0,0), 'reflect'),
                              kernel, bias=None, padding=0, groups=self.__channels)

                # zero-padding-based implementation
                #c1 = F.conv2d(img_float32, kernel, bias=None, padding=(0,padding), groups=self.__channels)
                #c1 = F.conv2d(c1.transpose(2,3), kernel, bias=None, padding=(0,padding), groups=self.__channels)
                return scaling * c1.transpose(2,3).contiguous()
            else:
                return scaling * img
        else:
            return img

    def __get_gaussian_kernel_1d(self, kernel_size, sigma, channels):

        # create a grid of integer coordinate pairs over the kernel area
        x_cord = torch.arange(kernel_size, device=self.device).float()

        # computing the gaussian kernel
        mean = (kernel_size - 1) / 2.
        variance = sigma * sigma
        gaussian_kernel = torch.exp((-((x_cord - mean) ** 2.)) / (2 * variance))

        # normalization
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # reshaping to match the right size needed by pytorch
        gaussian_kernel = gaussian_kernel.view(1, 1, 1, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)  # first element: output channels of the convolution

        return gaussian_kernel

    # this method is not used right now, but it might be a useful reference, do not delete it
    def __get_gaussian_kernel_2d(self, kernel_size, sigma, channels):

        # create a grid of integer coordinate pairs over the kernel area
        x_cord = torch.arange(kernel_size, device=self.device)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        # computing the gaussian kernel
        mean = (kernel_size - 1) / 2.
        variance = sigma * sigma
        gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1, dtype=torch.float32) / (2 * variance))

        # normalization
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # reshaping to match the right size needed by pytorch
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)  # first element: output channels of the convolution

        return gaussian_kernel




