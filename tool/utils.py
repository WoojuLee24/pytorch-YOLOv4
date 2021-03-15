import sys
import os
import time
import math
import numpy as np

import itertools
import struct  # get_image_size
import imghdr  # get_image_size
import tqdm

def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1.)


def softmax(x):
    x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
    x = x / np.expand_dims(x.sum(axis=1), axis=1)
    return x


def convert_coordinate(box):
    c_x = box[:, 0] + (box[:, 2] - box[:, 0]) / 2
    c_y = box[:, 1] + (box[:, 3] - box[:, 1]) / 2
    w = box[:, 2] - box[:, 0]
    h = box[:, 3] - box[:, 1]
    new_box = np.stack((c_x, c_y, w, h), axis=1)
    return new_box


def bbox_iou(box1, box2, x1y1x2y2=True):
    
    # print('iou box1:', box1)
    # print('iou box2:', box2)

    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]

        mx = min(box1[0], box2[0])
        Mx = max(box1[0] + w1, box2[0] + w2)
        my = min(box1[1], box2[1])
        My = max(box1[1] + h1, box2[1] + h2)
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def bbox_iou2(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    b1_x1, b1_x2 = box1[0], box1[2]
    b1_y1, b1_y2 = box1[1], box1[3]
    b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    # Intersection area
    inter_area = max(inter_rect_x2 - inter_rect_x1 + 1, 0) * max(inter_rect_y2 - inter_rect_y1 + 1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    
    return np.array(keep)



def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img


def save_pred_cv2(boxes, savename=None):

    start = time.time()
    for i in range(len(boxes)):
        box = boxes[i]
        x = (box[2] - box[0]) / 2 + box[0]
        y = (box[3] - box[1]) / 2 + box[1]
        w = box[2] - box[0]
        h = box[3] - box[1]
        cls_conf = box[5]
        # cls_id = box[6]
        if box[6] < 2:
            cls_id = box[6] + 1
        else:
            cls_id = box[6] + 2
        if savename:
            # print("save prediction results to %s" % savename)
            with open(savename, 'a') as f:
                f.write(str(cls_id) +" " + str(x) +" " + str(y) + " " + str(w) +  " " + str(h) + "\n")
    end = time.time()
    # print('-----------------------------------')
    # print('       writing files : %f' % (end-start))
    # print('-----------------------------------')




def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size / 5, 5)  # to avoid single truth problem
        return truths
    else:
        return np.array([])


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names



def post_processing(img, conf_thresh, nms_thresh, output):

    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors

    # [batch, num, 1, 4]
    box_array = output[0]
    # [batch, num, num_classes]
    confs = output[1]

    t1 = time.time()

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    t2 = time.time()

    bboxes_batch = []
    for i in range(box_array.shape[0]):
       
        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            # ll_box_array = convert_coordinate(ll_box_array)
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)
            
            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])

        bboxes_batch.append(bboxes)

    t3 = time.time()

    # print('-----------------------------------')
    # print('       max and argmax : %f' % (t2 - t1))
    # print('                  nms : %f' % (t3 - t2))
    # print('Post processing total : %f' % (t3 - t1))
    # print('-----------------------------------')
    
    return bboxes_batch


def get_batch_statistics(outputs, targets, iou_threshold=0.5):
    """ Compute true positives, predicted scores and predicted labels per sample """
    k = np.shape(outputs)
    # dic = {11: 0, 12: 0, 21:1, 22: 1, 23:1, 24:1, 25:1, 26:1, 31:1, 32:1, 33:1, 34:1,
    #        41:2, 42:2, 51:3, 52:3, 61:4, 62:4, 71:5, 72:5, 81:6, 82:6}
    batch_metrics = []
    true_positives = np.zeros(7)
    pred_detected = np.zeros(7)
    target_detected = np.zeros(7)
    targets = targets.data.cpu().numpy()
    epsilon = 1e-5
    for batch in range(len(outputs)):
        if outputs[batch] is None:
            continue
        output = outputs[batch]
        target = targets[batch]
        target_detected[int(target[0])] += 1
        # target_detected[target[0]] += 1
        for sample_i in range(len(output)):
            if output[sample_i] is None:
                continue
            boxes = output[sample_i]
            pred_box = boxes[:4]
            pred_score = boxes[4]
            pred_label = boxes[-1]
            target_label = int(target[0])
            target_box = target[1:]
            iou = bbox_iou2(pred_box, target_box)
            if iou > iou_threshold and pred_label == target_label:
                true_positives[pred_label] += 1
            pred_detected[pred_label] += 1

    precision = true_positives / (pred_detected + epsilon)
    recall = true_positives / (target_detected + epsilon)

    # f1 = 2 * precision * recall / (precision + recall + epsilon)
    # macro_f1 = f1.mean()

    return true_positives, pred_detected, target_detected


def get_f1(true_positives, pred_detected, target_detected):
    epsilon = 1e-5
    precision = true_positives / (pred_detected + epsilon)
    recall = true_positives / (target_detected + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    macro_f1 = f1.mean()
    return macro_f1


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap