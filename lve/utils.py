import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import flow_vis
import wandb
device_cpu = torch.device("cpu")


def load_json(json_file_path):
    f = open(json_file_path, "r")
    if f is None or not f or f.closed:
        raise IOError("Cannot read: " + json_file_path)
    json_loaded = json.load(f)
    f.close()
    return json_loaded


def save_json(json_file_path, json_to_save):
    f = open(json_file_path, "w")
    if f is None or not f or f.closed:
        raise IOError("Cannot access: " + json_file_path)
    json.dump(json_to_save, f, indent=4)
    f.close()


def np_uint8_to_torch_float_01(numpy_img, device=None):
    if numpy_img.ndim == 2:
        h = numpy_img.shape[0]
        w = numpy_img.shape[1]
        if device is None or device == device_cpu:
            return torch.from_numpy(numpy_img).float().div_(255.0).resize_(1, 1, h, w)
        else:
            return torch.from_numpy(numpy_img).float().resize_(1, 1, h, w).to(device).div_(255.0)
    elif numpy_img.ndim == 3:
        if device is None or device == device_cpu:
            return torch.from_numpy(numpy_img.transpose(2, 0, 1)).float().unsqueeze_(0).div_(255.0)
        else:
            return torch.from_numpy(numpy_img.transpose(2, 0, 1)).float().to(device).unsqueeze_(0).div_(255.0)
    elif numpy_img.ndim == 4:
        if device is None or device == device_cpu:
            return torch.from_numpy(numpy_img.transpose(0, 3, 1, 2)).float().div_(255.0)
        else:
            return torch.from_numpy(numpy_img.transpose(0, 3, 1, 2)).float().to(device).div_(255.0)
    else:
        raise ValueError("Unsupported image type.")


def np_float32_to_torch_float(numpy_img, device=None):
    if numpy_img.ndim == 2:
        h = numpy_img.shape[0]
        w = numpy_img.shape[1]
        if device is None or device == device_cpu:
            return torch.from_numpy(numpy_img).resize_(1, 1, h, w)
        else:
            return torch.from_numpy(numpy_img).resize_(1, 1, h, w).to(device)
    elif numpy_img.ndim == 3:
        if device is None or device == device_cpu:
            return torch.from_numpy(numpy_img.transpose(2, 0, 1)).float().unsqueeze_(0)
        else:
            return torch.from_numpy(numpy_img.transpose(2, 0, 1)).float().unsqueeze_(0).to(device)
    elif numpy_img.ndim == 4:
        if device is None or device == device_cpu:
            return torch.from_numpy(numpy_img.transpose(0, 3, 1, 2)).float()
        else:
            return torch.from_numpy(numpy_img.transpose(0, 3, 1, 2)).float().to(device)
    else:
        raise ValueError("Unsupported image type.")


def torch_float32_to_grayscale_float32(torch_img):
    return torch.sum(torch_img *
                     torch.tensor([[[[0.114]], [[0.587]], [[0.299]]]],
                                  dtype=torch.float32, device=torch_img.device), 1, keepdim=True)


def torch_float_01_to_np_uint8(torch_img):
    if torch_img.ndim == 2:
        return (torch_img * 255.0).cpu().numpy().astype(np.uint8)
    elif torch_img.ndim == 3:
        return (torch_img * 255.0).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
    elif torch_img.ndim == 4:
        return (torch_img * 255.0).cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
    else:
        raise ValueError("Unsupported image type.")


def torch_2d_tensor_to_csv(tensor, file):
    with open(file, 'w+') as f:
        for i in range(0, tensor.shape[0]):
            for j in range(0, tensor.shape[1]):
                f.write(str(tensor[i,j].item()))
                if j < tensor.shape[1] - 1:
                    f.write(',')
            f.write('\n')


def indices_to_rgb(indices):
    colors = np.array(
        [[189, 183, 107], [250, 235, 215], [0, 255, 255], [127, 255, 212], [240, 255, 255], [245, 245, 220],
         [255, 228, 196], [0, 0, 0], [255, 235, 205], [0, 0, 255], [138, 43, 226], [165, 42, 42], [222, 184, 135],
         [95, 158, 160], [127, 255, 0], [210, 105, 30], [255, 127, 80], [100, 149, 237], [255, 248, 220],
         [220, 20, 60], [0, 255, 255], [0, 0, 139], [0, 139, 139], [184, 134, 11], [169, 169, 169], [0, 100, 0],
         [169, 169, 169], [240, 248, 255], [139, 0, 139], [85, 107, 47], [255, 140, 0], [153, 50, 204], [139, 0, 0],
         [233, 150, 122], [143, 188, 143], [72, 61, 139], [47, 79, 79], [47, 79, 79], [0, 206, 209], [148, 0, 211],
         [255, 20, 147], [0, 191, 255], [105, 105, 105], [105, 105, 105], [30, 144, 255], [178, 34, 34],
         [255, 250, 240], [34, 139, 34], [255, 0, 255], [220, 220, 220], [248, 248, 255], [255, 215, 0],
         [218, 165, 32], [128, 128, 128], [0, 128, 0], [173, 255, 47], [128, 128, 128], [240, 255, 240],
         [255, 105, 180], [205, 92, 92], [75, 0, 130], [255, 255, 240], [240, 230, 140], [230, 230, 250],
         [255, 240, 245], [124, 252, 0], [255, 250, 205], [173, 216, 230], [240, 128, 128], [224, 255, 255],
         [250, 250, 210], [211, 211, 211], [144, 238, 144], [211, 211, 211], [255, 182, 193], [255, 160, 122],
         [32, 178, 170], [135, 206, 250], [119, 136, 153], [119, 136, 153], [176, 196, 222], [255, 255, 224],
         [0, 255, 0], [50, 205, 50], [250, 240, 230], [255, 0, 255], [128, 0, 0], [102, 205, 170], [0, 0, 205],
         [186, 85, 211], [147, 112, 219], [60, 179, 113], [123, 104, 238], [0, 250, 154], [72, 209, 204],
         [199, 21, 133], [25, 25, 112], [245, 255, 250], [255, 228, 225], [255, 228, 181], [255, 222, 173],
         [0, 0, 128], [253, 245, 230], [128, 128, 0], [107, 142, 35], [255, 165, 0], [255, 69, 0], [218, 112, 214],
         [238, 232, 170], [152, 251, 152], [175, 238, 238], [219, 112, 147], [255, 239, 213], [255, 218, 185],
         [205, 133, 63], [255, 192, 203], [221, 160, 221], [176, 224, 230], [128, 0, 128], [255, 0, 0],
         [188, 143, 143], [65, 105, 225], [139, 69, 19], [250, 128, 114], [244, 164, 96], [46, 139, 87],
         [255, 245, 238], [160, 82, 45], [192, 192, 192], [135, 206, 235], [106, 90, 205], [112, 128, 144],
         [112, 128, 144], [255, 250, 250], [0, 255, 127], [70, 130, 180], [210, 180, 140], [0, 128, 128],
         [216, 191, 216], [255, 99, 71], [64, 224, 208], [238, 130, 238], [245, 222, 179], [255, 255, 255],
         [245, 245, 245], [255, 255, 0], [154, 205, 50]], dtype=np.uint8)
    return colors[indices.astype(np.int64)]

def plot_what_heatmap(what_img, what_ref, anchor=None, distance='euclidean', out='spatial_coherence_heatmap.png'):
    what_ref = what_ref.reshape(what_img.shape[1], 1, 1)
    if distance == 'euclidean':
        diff = what_img[0, :, :, :] - what_ref
        dist = np.linalg.norm(diff, axis=0)
    else:
        dist = 1.0 - np.sum((what_img[0, :, :, :] * what_ref), axis=0)
    fig, ax = plt.subplots()
    ax.axis('off')
    # plt.imshow(recon_img, cmap=plt.cm.gray)
    if anchor is not None:
        anchorx, anchory = anchor
        plt.plot([anchory], [anchorx], marker='x')
    plt.imshow(-dist, cmap=plt.cm.Greens)  # alpha=0.95
    plt.savefig(out, bbox_inches='tight', transparent=True, pad_inches=0.0)


def warp(old_frame, flow):
    _, _, h, w = old_frame.shape[0], old_frame.shape[1], old_frame.shape[2], old_frame.shape[3]
    warped = old_frame.clone()
    for i in range(0, h):
        for j in range(0, w):
            ii = max(min(int(round(i + flow[0][1][i][j].item())), h - 1), 0)
            jj = max(min(int(round(j + flow[0][0][i][j].item())), w - 1), 0)
            warped[0, 0, i, j] = 0.0
            warped[0, 0, ii, jj] = old_frame[0, 0, i, j]
    return warped


def backward_warp(frame, displacement):
    # _, _, h, w = old_frame.shape[0], old_frame.shape[1], old_frame.shape[2], old_frame.shape[3]
    # warped = old_frame.clone()
    # for i in range(0, h):
    #     for j in range(0, w):
    #         ii = max(min(int(round(i + flow[0][1][i][j].item())), h - 1), 0)
    #         jj = max(min(int(round(j + flow[0][0][i][j].item())), w - 1), 0)
    #         warped[0, 0, i, j] = 0.0
    #         warped[0, 0, ii, jj] = old_frame[0, 0, i, j]
    # return warped
    b, f, h, w = frame.shape
    region_h, region_w = torch.meshgrid(torch.arange(h), torch.arange(w))
    region_h = region_h.to(displacement.device)
    region_w = region_w.to(displacement.device)
    u_x = 2.0 * (region_w + displacement[:, 0, :, :]) / (w - 1) - 1.0  # w-1??
    u_y = 2.0 * (region_h + displacement[:, 1, :, :]) / (h - 1) - 1.0
    ugrid = torch.stack((u_x, u_y), dim=-1)

    old_frame = torch.nn.functional.grid_sample(frame, ugrid, padding_mode='border', mode='bilinear', align_corners=True)
    return old_frame

def plot_standard_heatmap(data, name):
    dic = {name + '_magnitude': []}
    fnames = []
    n, c, h, w = data.shape
    for i in range(data.shape[0]):
        if c > 1:
            heat = torch.sqrt(torch.sum(data ** 2, dim=1))
        else:
            heat = data[i, 0]

        plt.imshow(heat.detach().cpu().numpy(), cmap='hot')
        plt.colorbar()
        r = random.randint(0, 1000000000)
        fname = name + '_mag_' + str(r) + '.png'
        fnames.append(fname)
        plt.savefig(fname)
        plt.clf()
        dic[name + '_magnitude'].append(wandb.Image(fname))
    return dic, fnames

def visualize_flows(flows, prefix='flow'):
    flow_dic = {prefix + '_color': [], prefix + '_magnitude': [], prefix + '_x': [], prefix + '_y': []}
    fnames = []
    for i in range(flows.shape[0]):
        flow_u = flows[i, 0, :, :]
        flow_v = flows[i, 1, :, :]
        flow_uv = torch.stack((flow_u, flow_v), dim=2)
        flow_norm = torch.sqrt(torch.sum(flow_uv ** 2, dim=2))
        flow_uv = flow_uv.detach().cpu().numpy()
        flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)
        flow_dic[prefix + '_color'].append(wandb.Image(flow_color))
        plt.imshow(flow_norm.detach().cpu().numpy(), cmap='hot')
        plt.colorbar()
        r = random.randint(0, 1000000000)
        fname = prefix + '_mag_' + str(r) + '.png'
        fnames.append(fname)
        plt.savefig(fname)
        plt.clf()
        flow_dic[prefix + '_magnitude'].append(wandb.Image(fname))

        plt.imshow(flow_u.detach().cpu().numpy(), cmap='hot')
        plt.colorbar()
        fname = prefix + '_x_' + str(r) + '.png'
        fnames.append(fname)
        plt.savefig(fname)
        plt.clf()
        flow_dic[prefix + '_x'].append(wandb.Image(fname))

        plt.imshow(flow_v.detach().cpu().numpy(), cmap='hot')
        plt.colorbar()
        fname = prefix + '_y_' + str(r) + '.png'
        fnames.append(fname)
        plt.savefig(fname)
        plt.clf()
        flow_dic[prefix + '_y'].append(wandb.Image(fname))
    return flow_dic, fnames

def crop_center(img, cropx, cropy):
    y,x,c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    return img[starty:starty+cropy, startx:startx+cropx, :]