import cv2
import numpy as np


class BlurCV:

    def __init__(self, w, h, c, device=None):
        kernel_size = int(0.25 * min(w, h))
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
        self.__sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.3  # OpenCV rule - 0.5
        self.__channels = c  # unused right now

    def __call__(self, img, blur_factor):
        if blur_factor < 0.0 or blur_factor > 1.0:
            raise ValueError("Invalid blur factor (it must be in [0,1]): " + str(blur_factor))

        if blur_factor > 0.000001:
            scaling = 1.0 - blur_factor
            sigma = blur_factor * self.__sigma
            kernel_size = int(2 * ((sigma - 0.3) / 0.3 + 1) + 1)
            if kernel_size % 2 == 0:
                kernel_size = kernel_size + 1

            if kernel_size > 1:
                gaussian_filter = cv2.getGaussianKernel(kernel_size, sigma)  # 1D
                if img.ndim == 4:
                    h = img.shape[1]
                    w = img.shape[2]
                    c = img.shape[3]

                    # batch
                    _img = np.ndarray(img.shape, dtype=img.dtype)
                    for i in range(0, img.shape[0]):
                        _img[i,:] = cv2.sepFilter2D(img[i,:], -1, gaussian_filter, gaussian_filter).reshape(h, w, c)
                    img = scaling * _img
                else:

                    # single
                    img = scaling * cv2.sepFilter2D(img, -1, gaussian_filter, gaussian_filter)
            else:
                img = scaling * img
        return img

