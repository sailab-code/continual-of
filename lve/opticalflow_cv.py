import cv2
import numpy as np
from gzip import GzipFile


class OpticalFlowCV:

    def __init__(self, backward=True):
        self.prev_frame_gray_scale = None
        self.frame_gray_scale = None
        self.optical_flow = None
        self.backward = backward

    def __call__(self, frame):
        if self.frame_gray_scale is not None:
            prev_frame_gray_scale = self.frame_gray_scale
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame_gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame_gray_scale = frame

            self.prev_frame_gray_scale = self.frame_gray_scale
            self.frame_gray_scale = frame_gray_scale

            # motion flow is estimated from frame at time t to frame at time t-1 (yes, backward...)
            if self.backward:
                frames = (frame_gray_scale, prev_frame_gray_scale)
            else:
                frames = (prev_frame_gray_scale, frame_gray_scale)

            self.optical_flow = cv2.calcOpticalFlowFarneback(frames[0],
                                                             frames[1],
                                                             self.optical_flow,
                                                             pyr_scale=0.4,
                                                             levels=5,  # pyramid levels
                                                             winsize=12,
                                                             iterations=10,
                                                             poly_n=5,
                                                             poly_sigma=1.1,
                                                             flags=0)

            # ---------------------------------------------------------------------------------------
            # this is the code I used to qualitatively evaluate the optical flow field
            # when used to rebuild the destination image given the source image and the
            # field itself
            # ---------------------------------------------------------------------------------------
            # import numpy as np
            # import lve
            # import cv2
            #
            # t_minus_1 = cv2.cvtColor(cv2.imread("t-1.png"), cv2.COLOR_BGR2GRAY)
            # t = cv2.cvtColor(cv2.imread("t.png"), cv2.COLOR_BGR2GRAY)
            #
            # of = lve.OpticalFlowCV()
            #
            # of(t_minus_1)
            # m = of(t)
            #
            # h, w = t.shape
            # t_rebuilt_from_t_minus_1 = np.zeros((h, w), t.dtype)
            #
            # for i in range(0, h):
            #     for j in range(0, w):
            #         ii = max(min(int(round(i + m[i][j][1])), h - 1), 0)
            #         jj = max(min(int(round(j + m[i][j][0])), w - 1), 0)
            #         t_rebuilt_from_t_minus_1[i][j] = t_minus_1[ii][jj]
            #
            # cv2.imwrite("_t_minus_1.png", t_minus_1)
            # cv2.imwrite("_t.png", t)
            # cv2.imwrite("_t-rebuilt_from_t_minus_1.png", t_rebuilt_from_t_minus_1)
            # cv2.imwrite("_m_map.png", lve.OpticalFlowCV.draw_flow_map(t, m))
            # cv2.imwrite("_m_lines.png", lve.OpticalFlowCV.draw_flow_lines(t, m, line_step=5))
            # ---------------------------------------------------------------------------------------

        else:
            if self.frame_gray_scale is None:
                if not ((frame.ndim == 3 and frame.shape[2] == 1) or frame.ndim == 2):
                    a, b, c = frame.shape
                    self.optical_flow = np.zeros((a, b, 2), np.float32)
                    self.frame_gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    if frame.ndim == 2:
                        a, b = frame.shape
                    else:
                        a, b, c = frame.shape
                    self.optical_flow = np.zeros((a, b, 2), np.float32)
                    self.frame_gray_scale = frame

        return self.optical_flow

    @staticmethod
    def load_flow(file_name):

        # loading optical flow
        file = GzipFile(file_name, 'rb')
        optical_flow = np.load(file)
        file.close()

        return optical_flow

    @staticmethod
    def save_flow(file_name, optical_flow, prev_frame, save_visualizations):

        # saving optical flow
        with GzipFile(file_name, 'wb') as file:
            np.save(file, optical_flow)

        # saving a couple of images to show the computed optical flow
        if save_visualizations:
            cv2.imwrite(file_name + ".lines.png", OpticalFlowCV.draw_flow_lines(prev_frame, optical_flow))
            cv2.imwrite(file_name + ".map.png", OpticalFlowCV.draw_flow_map(prev_frame, optical_flow))

    @staticmethod
    def draw_flow_lines(frame, optical_flow, line_step=16, line_color=(0, 255, 0)):
        frame_with_lines = frame.copy()
        line_color = (line_color[2], line_color[1], line_color[0])

        for y in range(0, optical_flow.shape[0], line_step):
            for x in range(0, optical_flow.shape[1], line_step):
                fx, fy = optical_flow[y, x]
                cv2.line(frame_with_lines, (x, y), (int(x + fx), int(y + fy)), line_color)
                cv2.circle(frame_with_lines, (x, y), 1, line_color, -1)

        return frame_with_lines

    @staticmethod
    def draw_flow_map(frame, optical_flow):
        hsv = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.float32)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        frame_flow_map = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return frame_flow_map
