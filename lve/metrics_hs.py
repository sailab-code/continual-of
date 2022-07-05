import numpy
import numpy as np
import collections
from collections import OrderedDict
import lve
import os
import torch
import copy


def compute_confusion(y, y_pred):
    indices = 2 * y.to(torch.int64) + y_pred.to(torch.int64)
    m = torch.bincount(indices,
                       minlength=2 ** 2).reshape(2, 2)
    return m

class HsMetricsContainer:
    def __init__(self, output_stream, options):
        self.output_stream = output_stream
        self.window_size = options['window']  # these are the metrics-related options only
        self.hs_metrics = HsMetrics(output_stream, options)
        # only for structure

        self.__stats = OrderedDict({
            'whole_frame':
                {
                    'running':
                        {
                            "hs_invariance": None,
                            "hs_smoothness": None,
                            "hs": None,
                            "flow_std": None,
                            "photo": None,
                            "photo_and_smooth": None
                        },
                    'window':
                        {
                            "hs_invariance": None,
                            "hs_smoothness": None,
                            "hs": None,
                            "flow_std": None,
                            "photo": None,
                            "photo_and_smooth": None
                        },
                }
        })

        self.output_stream.register_output_elements(self.get_output_types())


    def update(self, updated, hs_invariance_term, hs_smoothness_term, hs_loss, photo_and_smooth_loss, photo_term,
               flow_std,
               motion_mask, predicted_motion_mask, recon_acc
               ):
        self.hs_metrics.update(updated=updated,
                               hs_invariance_term=hs_invariance_term,
                               hs_smoothness_term=hs_smoothness_term,
                               hs_loss=hs_loss,
                               photo_and_smooth_loss=photo_and_smooth_loss,
                               photo_term=photo_term,
                               flow_std=flow_std,
                               motion_mask=motion_mask,
                               predicted_motion_mask=predicted_motion_mask,
                               recon_acc=recon_acc)

    def compute(self):
        self.hs_metrics.compute()
        # update output stream

        hs_stats = self.hs_metrics.get_stats()

        self.__stats = hs_stats

        self.output_stream.save_elements({"stats.metrics": self.__stats,  # dictionary
                                          "logs.metrics": self.__convert_stats_values_to_list(),  # CSV log
                                          "tb.metrics": self.__stats}, prev_frame=True)

    def save(self, model_folder):
        self.hs_metrics.save(model_folder)

    def load(self, model_folder, metrics_options):
        self.hs_metrics.load(model_folder, metrics_options)

    def get_output_types(self):
        output_types = {
            "stats.metrics": {'data_type': lve.OutputType.JSON, 'per_frame': True},
            "logs.metrics": {'data_type': lve.OutputType.TEXT, 'per_frame': False},
            "logs.metrics__header": ['frame'] + self.__convert_stats_keys_to_list()
        }
        return output_types

    def __convert_stats_values_to_list(self):
        stats_list = []
        for area, area_d in self.__stats.items():
            for setting, setting_d in area_d.items():
                for metric, metric_v in setting_d.items():
                    if isinstance(metric_v, list):
                        for m_v in metric_v:
                            stats_list.append(m_v)
                    else:
                        stats_list.append(metric_v)
        return stats_list

    def __convert_stats_keys_to_list(self):
        stats_list = []
        for area, area_d in self.__stats.items():
            for setting, setting_d in area_d.items():
                for metric, metric_v in setting_d.items():
                    if isinstance(metric_v, list):
                        ml = len(metric_v)
                        for k in range(0, ml - 1):
                            stats_list.append(metric + '_c' + str(k))
                        stats_list.append(metric + '_glob')
                    else:
                        stats_list.append(metric)
        return stats_list

class ComplexMetric():
    def __init__(self, thresh, window_size):
        self.thresh = thresh
        self.window_size = window_size
        if window_size == 0:
            self.m = {t: 0.0 for t in thresh}
        else:
            self.m = {t: collections.deque(maxlen=window_size) for t in thresh}

    def increment(self, new):
        if self.window_size == 0:
           self.m = {t: self.m[t] + new[t] for t in self.thresh}
        else:
            for t in self.thresh:
                self.m[t].appendleft(new[t])

    def get(self, frac):
        if self.window_size == 0:
            return {t: self.m[t] / frac for t in self.thresh}
        else:
            return {t: np.sum(self.m[t]) / len(self.m[t]) for t in self.thresh}

    @staticmethod
    def load(m, thresh, window_size):
        x = ComplexMetric(thresh, window_size)
        x.m = m
        return x

class Confusion:
    def __init__(self, labels, predictions):
        self.cm = numpy.zeros((labels, predictions))

    def get_cm(self):
        return self.cm

class HsMetrics:
    def __init__(self, output_stream, options):
        # options
        self.output_stream = output_stream
        self.window_size = options['window']  # these are the metrics-related options only

        # references to the confusion and contingency matrices
        self.running_hs_invariance_term = 0.0
        self.running_hs_smoothness_term = 0.0
        self.running_hs_loss = 0.0
        self.running_photo_and_smooth_loss = 0.0
        self.running_photo_term = 0.0
        self.running_error_rate = 0.0
        self.running_flow_std = 0.0
        self.running_motion_confusion = Confusion(2, 2)
        self.running_recon_acc = ComplexMetric(options['recon_acc_thresh'], window_size=0)

        self.t_counter = 0

        self.window_updated = collections.deque(maxlen=self.window_size)
        self.window_hs_invariance_term = collections.deque(maxlen=self.window_size)
        self.window_hs_smoothness_term = collections.deque(maxlen=self.window_size)
        self.window_hs_loss = collections.deque(maxlen=self.window_size)
        self.window_photo_and_smooth_loss = collections.deque(maxlen=self.window_size)
        self.window_photo_term = collections.deque(maxlen=self.window_size)
        self.window_flow_std = collections.deque(maxlen=self.window_size)

        self.window_motion_confusion = collections.deque(maxlen=self.window_size)
        self.window_recon_acc = ComplexMetric(options['recon_acc_thresh'], window_size=self.window_size)

        self.__stats = OrderedDict({
            'whole_frame':
                {
                    'running':
                        {
                            'hs_invariance': None,
                            'hs_smoothness': None,
                            'hs': None,
                            'photo_and_smooth': None,
                            'photo': None,
                            'moving_acc': None,
                            'moving_f1': None,
                            'moving_cm': [None]  * 4,
                            'recon_acc': None
                        },
                    'window':
                        {
                            'updated': None,
                            'hs_invariance': None,
                            'hs_smoothness': None,
                            'hs': None,
                            'photo_and_smooth': None,
                            'photo': None,
                            'moving_acc': None,
                            'moving_f1': None,
                            'moving_cm': [None] * 4,
                            'moving_precision': None,
                            'moving_recall': None,
                            'recon_acc': None
                        },
                }
        })

    def get_stats(self):
        return self.__stats

    def save(self, model_folder):
        metrics_model_folder = model_folder + os.sep + "hs_metrics" + os.sep
        if not os.path.exists(metrics_model_folder):
            os.makedirs(metrics_model_folder)

        # saving metrics-status related tensors
        torch.save({"running_hs_loss": self.running_hs_loss,
                    "running_photo_and_smooth_loss": self.running_photo_and_smooth_loss,
                    "running_hs_invariance_term": self.running_hs_invariance_term,
                    "running_hs_smoothness_term": self.running_hs_smoothness_term,
                    "running_photo_term": self.running_photo_term,
                    "running_motion_confusion": self.running_motion_confusion,
                    "running_recon_acc": self.running_recon_acc.m,
                    "running_error_rate": self.running_error_rate,
                    "window_hs_loss": self.window_hs_loss,
                    "window_hs_photo_and_smooth_loss": self.window_photo_and_smooth_loss,
                    "window_hs_invariance_term": self.window_hs_invariance_term,
                    "window_hs_smoothness_term": self.window_hs_smoothness_term,
                    "window_photo_term": self.window_photo_term,
                    "window_motion_confusion": self.window_motion_confusion,
                    "window_recon_acc": self.window_recon_acc.m
                    },
                   metrics_model_folder + "metrics.pth")

    def load(self, model_folder, metrics_options):
        metrics_model_folder = model_folder + os.sep + "hs_metrics" + os.sep

        # loading metrics-status related tensors
        if os.path.exists(metrics_model_folder + "metrics.pth"):
            metrics_status = torch.load(metrics_model_folder + "metrics.pth")

            self.running_hs_loss = metrics_status["running_hs_loss"]
            self.running_photo_and_smooth_loss = metrics_status["running_photo_and_smooth_loss"]
            self.running_hs_invariance_term = metrics_status["running_hs_invariance_term"]
            self.running_hs_smoothness_term = metrics_status["running_hs_smoothness_term"]
            self.running_photo_term = metrics_status["running_photo"]
            self.running_motion_confusion = metrics_status["running_motion_confusion"]
            self.running_recon_acc = ComplexMetric.load(metrics_status["running_recon_acc"],
                                                        metrics_options['recon_acc_thresh'],
                                                        window_size=0)
            self.running_error_rate = metrics_status["running_error_rate"]

            self.window_hs_loss = metrics_status["window_hs_term"]
            self.window_photo_and_smooth_loss = metrics_status["window_photo_and_smooth_loss"]
            self.window_hs_invariance_term = metrics_status["window_hs_invariance_term"]
            self.window_hs_smoothness_term = metrics_status["window_hs_smoothness_term"]
            self.window_photo_term = metrics_status["window_photo"]
            self.window_motion_confusion = metrics_status["window_motion_confusion"]
            self.window_recon_acc = ComplexMetric.load(metrics_status["running_recon_acc"],
                                                        metrics_options['recon_acc_thresh'],
                                                        window_size=metrics_options['window'])

    @staticmethod
    def compute_matrices_and_update_running(pred, target, running_cm, running_cm_window):
        current_cm = compute_confusion(y_pred=torch.as_tensor(pred), y=torch.as_tensor(target)).numpy()
        running_cm.cm = running_cm.cm + current_cm

        # windowed confusion matrix update
        running_cm_window.appendleft(current_cm)

    @staticmethod
    def accuracy(cm):
        acc_det = cm.sum(axis=1)
        acc_det[acc_det == 0] = 1
        per_class_accuracy = cm.diagonal() / acc_det
        avg_accuracy = np.mean(per_class_accuracy)  # macro
        glob_accuracy = cm.diagonal().sum() / cm.sum()
        return per_class_accuracy, avg_accuracy, glob_accuracy

    @staticmethod
    def f1_and_pr(cm):
        num_classes = cm.shape[0]
        per_class_f1 = np.zeros(num_classes)

        for c in range(0, num_classes):
            tp = cm[c, c]
            fn = np.sum(cm[c, :]) - tp
            fp = np.sum(cm[:, c]) - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.
            per_class_f1[c] = (2. * p * r) / (p + r) if (p + r) > 0 else 0.

        # p and r are computed for num_classes=1 (i.e. the positive class)
        avg_f1 = np.mean(per_class_f1)  # macro
        glob_f1 = per_class_f1[1]
        return per_class_f1, avg_f1, glob_f1, p, r

    @staticmethod
    def compute_metrics_from_confusion(confusion_mat):
        per_class_accuracy, avg_accuracy, glob_accuracy = HsMetrics.accuracy(confusion_mat)
        per_class_f1, avg_f1, glob_f1, precision, recall = HsMetrics.f1_and_pr(confusion_mat)

        return {'acc': glob_accuracy,
                'f1': glob_f1,
                'precision': precision,
                'recall': recall
                }

    def update(self, updated, hs_invariance_term, hs_smoothness_term, hs_loss, photo_and_smooth_loss, photo_term,
               flow_std, motion_mask, predicted_motion_mask, recon_acc):

        HsMetrics.compute_matrices_and_update_running(pred=predicted_motion_mask.flatten(),
                                                      target=motion_mask.flatten(),
                                                      running_cm=self.running_motion_confusion,
                                                      running_cm_window=self.window_motion_confusion)

        self.running_hs_invariance_term += hs_invariance_term
        self.running_hs_smoothness_term += hs_smoothness_term
        self.running_hs_loss += hs_loss
        self.running_photo_and_smooth_loss += photo_and_smooth_loss
        self.running_photo_term += photo_term
        self.running_flow_std += flow_std
        self.running_recon_acc.increment(recon_acc)

        self.window_updated.appendleft(updated)
        self.window_hs_invariance_term.appendleft(hs_invariance_term)
        self.window_hs_smoothness_term.appendleft(hs_smoothness_term)
        self.window_hs_loss.appendleft(hs_loss)
        self.window_photo_and_smooth_loss.appendleft(photo_and_smooth_loss)
        self.window_photo_term.appendleft(photo_term)
        self.window_flow_std.appendleft(flow_std)
        self.window_recon_acc.increment(recon_acc)

        self.t_counter += 1

    def compute(self):
        running_motion_confusion_metrics = HsMetrics.compute_metrics_from_confusion(self.running_motion_confusion.cm)
        window_motion_confusion_sum = np.sum(self.window_motion_confusion, axis=0)
        window_motion_confusion_metrics = HsMetrics.compute_metrics_from_confusion(window_motion_confusion_sum)

        self.__stats.update({
            'whole_frame':
                {
                    'running':
                        {
                            "hs_invariance": self.running_hs_invariance_term / self.t_counter,
                            "hs_smoothness": self.running_hs_smoothness_term / self.t_counter,
                            "hs": self.running_hs_loss / self.t_counter,
                            "photo": self.running_photo_term / self.t_counter,
                            "photo_and_smooth": self.running_photo_and_smooth_loss / self.t_counter,
                            "error_rate": self.running_error_rate / self.t_counter,
                            "flow_std": self.running_flow_std / self.t_counter,
                            "recon_acc": self.running_recon_acc.get(frac=self.t_counter),
                            "moving_acc": running_motion_confusion_metrics['acc'],
                            "moving_f1": running_motion_confusion_metrics['f1'],
                            "moving_precision": running_motion_confusion_metrics['precision'],
                            "moving_recall": running_motion_confusion_metrics['recall'],
                            "moving_cm": self.running_motion_confusion.cm.tolist()
                        },
                    'window':
                        {
                            "updated": np.sum(self.window_updated) / len(self.window_updated),
                            "hs_invariance": np.sum(self.window_hs_invariance_term) / len(self.window_hs_invariance_term),
                            "hs_smoothness": np.sum(self.window_hs_smoothness_term) / len(self.window_hs_smoothness_term),
                            "hs": np.sum(self.window_hs_loss) / len(self.window_hs_loss),
                            "photo_and_smooth": np.sum(self.window_photo_and_smooth_loss) / len(self.window_photo_and_smooth_loss),
                            "photo": np.sum(self.window_photo_term) / len(self.window_photo_term),
                            "flow_std": np.sum(self.window_flow_std) / len(self.window_flow_std),
                            "recon_acc":  self.window_recon_acc.get(frac=self.window_recon_acc),
                            "moving_acc": window_motion_confusion_metrics['acc'],
                            "moving_f1": window_motion_confusion_metrics['f1'],
                            "moving_precision": window_motion_confusion_metrics['precision'],
                            "moving_recall": window_motion_confusion_metrics['recall'],
                            "moving_cm": window_motion_confusion_sum.tolist()
                        },
                }
        })

