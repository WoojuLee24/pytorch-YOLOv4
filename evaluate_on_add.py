# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
from dataset import ADDEvalDataset
from torch.utils.data import DataLoader
import argparse

"""hyper parameters"""
use_cuda = True


def evaluate_batch_images_cv2(cfgfile, weightfile, imgfolder, savepath):
    import cv2
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    elif num_classes == 7:
        namesfile = 'data/ADD.names'
    class_names = load_class_names(namesfile)

    dataset = ADDEvalDataset(imgfolder)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, sampler=None, pin_memory=True, num_workers=2, drop_last=False)

    true_positives = np.zeros(7)
    pred_detected = np.zeros(7)
    target_detected = np.zeros(7)
    # _, (inputs, labels) = next(enumerate(loader))
    for cur_iter, (inputs, labels) in enumerate(loader):
        boxes = tensor_detect(m, inputs, 0.4, 0.6, use_cuda)
        (tp, pd, td) = get_batch_statistics(boxes, labels)
        true_positives += tp
        pred_detected += pd
        target_detected += td
    macro_f1 = get_f1(true_positives, pred_detected, target_detected)
    print("macro f1: ", macro_f1)


def detect_batch_images_cv2(cfgfile, weightfile, imgfolder, savepath):
    import cv2
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    elif num_classes == 7:
        namesfile = 'data/ADD.names'
    class_names = load_class_names(namesfile)

    dataset = ADDDataset(imgfolder)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, sampler=None, pin_memory=True, num_workers=4, drop_last=False)

    for cur_iter, (inputs, impath) in enumerate(loader):
        # start = time.time()
        boxes = tensor_detect(m, inputs, 0.4, 0.6, use_cuda)
        # finish = time.time()
        for b in range(len(boxes)):
            txtname = os.path.join(savepath, os.path.splitext(os.path.basename(impath[b]))[0] + ".txt")
            save_pred_cv2(boxes[b], savename=txtname)


def detect_images_cv2(cfgfile, weightfile, imgfolder, savepath):
    import cv2
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    elif num_classes == 7:
        namesfile = 'data/ADD.names'
    class_names = load_class_names(namesfile)

    dataset = ADDDataset(imgfolder)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, sampler=None, pin_memory=True, num_workers=4, drop_last=False)


    # image list
    imglist = sorted(os.listdir(imgfolder))
    for imgfile in imglist:
        txtname = os.path.join(savepath, os.path.splitext(os.path.basename(imgfile))[0] + ".txt")
        imgfile = os.path.join(imgfolder, imgfile)
        img = cv2.imread(imgfile)

        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        # for i in range(2):
        #     start = time.time()
        #     boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        #     finish = time.time()
        #     if i == 1:
        #         print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        # print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))
        save_pred_cv2(boxes[0], savename=txtname)

        # plot_boxes_cv2(img, boxes[0], savename='/ws/data/team_ksaj/imgname' + '.txt', class_names=class_names)


def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    elif num_classes == 7:
        namesfile = 'data/ADD.names'
    class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch1.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        default='/ws/data/ADD_TEST/1_01_000000_314.jpg',
                        help='path of your image file.', dest='imgfile')
    parser.add_argument('-imgfolder', type=str,
                        default='/ws/data/ADD_TEST',
                        help='path of your image folder.', dest='imgfolder')
    parser.add_argument('-savepath', type=str,
                        default='/ws/data/team_ksaj',
                        help='path of your image folder.', dest='savepath')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    start = time.time()
    args = get_args()
    # if args.imgfile:
    #    detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
    if args.imgfolder:
        evaluate_batch_images_cv2(args.cfgfile, args.weightfile, args.imgfolder, args.savepath)
    else:
        detect_cv2_camera(args.cfgfile, args.weightfile)
    end = time.time()
    print("total time: ", end-start)
