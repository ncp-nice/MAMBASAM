import torch
import numpy as np
import cv2
from scipy.spatial.distance import directed_hausdorff


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def _list_tensor(x, y):
    m = torch.nn.Sigmoid()
    if type(x) is list:
        x = torch.tensor(np.array(x))
        y = torch.tensor(np.array(y))
        if x.min() < 0:
            x = m(x)
    else:
        x, y = x, y
        if x.min() < 0:
            x = m(x)
    return x, y


def iou(pr, gt, eps=1e-7, threshold=0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_, dim=[1, 2, 3])
    union = torch.sum(gt_, dim=[1, 2, 3]) + torch.sum(pr_, dim=[1, 2, 3]) - intersection
    return ((intersection + eps) / (union + eps)).cpu().numpy()


def dice(pr, gt, eps=1e-7, threshold=0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_, dim=[1, 2, 3])
    union = torch.sum(gt_, dim=[1, 2, 3]) + torch.sum(pr_, dim=[1, 2, 3])
    return ((2. * intersection + eps) / (union + eps)).cpu().numpy()


def f_beta(pr, gt, beta=0.3, eps=1e-7, threshold=0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    tp = torch.sum(gt_ * pr_, dim=[1, 2, 3])
    precision = tp / (torch.sum(pr_, dim=[1, 2, 3]) + eps)
    recall = tp / (torch.sum(gt_, dim=[1, 2, 3]) + eps)
    f_beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + eps)
    return f_beta.detach().cpu().numpy()  # 同样调用detach()


def mae(pr, gt):
    pr_, gt_ = _list_tensor(pr, gt)
    return torch.mean(torch.abs(pr_ - gt_)).detach().cpu().numpy()

def ber(pr, gt, eps=1e-7, threshold=0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    tp = torch.sum(gt_ * pr_, dim=[1, 2, 3])
    tn = torch.sum((1 - gt_) * (1 - pr_), dim=[1, 2, 3])
    np = torch.sum(gt_, dim=[1, 2, 3])
    nn = torch.sum(1 - gt_, dim=[1, 2, 3])
    ber = (1 - 0.5 * (tp / (np + eps) + tn / (nn + eps))) * 100
    return ber.cpu().numpy()


def hausdorff_distance(pr, gt, threshold=0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)

    pr_np = pr_.cpu().numpy().astype(np.bool)
    gt_np = gt_.cpu().numpy().astype(np.bool)

    # Flatten arrays to list of points
    pr_points = np.argwhere(pr_np)
    gt_points = np.argwhere(gt_np)

    if len(pr_points) == 0 or len(gt_points) == 0:
        return np.inf

    # Compute the directed Hausdorff distance
    hd_1 = directed_hausdorff(pr_points, gt_points)[0]
    hd_2 = directed_hausdorff(gt_points, pr_points)[0]

    # The Hausdorff distance is the maximum of the directed distances
    hd = max(hd_1, hd_2)

    return hd


def SegMetrics(pred, label, metrics):
    metric_list = []
    if isinstance(metrics, str):
        metrics = [metrics, ]
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'iou':
            metric_list.append(np.mean(iou(pred, label)))
        elif metric == 'dice':
            metric_list.append(np.mean(dice(pred, label)))
        elif metric == 'f_beta':
            metric_list.append(np.mean(f_beta(pred, label)))
        elif metric == 'mae':
            metric_list.append(np.mean(mae(pred, label)))
        elif metric == 'ber':
            metric_list.append(np.mean(ber(pred, label)))
        elif metric == 'hd':
            metric_list.append(np.mean(hausdorff_distance(pred, label)))
        else:
            raise ValueError('metric %s not recognized' % metric)
    if pred is not None:
        metric = np.array(metric_list)
    else:
        raise ValueError('metric mistakes in calculations')
    return metric
