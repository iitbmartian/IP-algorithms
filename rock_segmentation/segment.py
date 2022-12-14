import argparse
import cv2
import numpy as np
import os

import numpy as np
import math
import cv2
import importlib
import json
import torch
from collections import OrderedDict
import torchvision.transforms as transforms
import PIL.Image as Image
import sys
sys.path.append("./")
print(sys.path)

from image_super_resolution.CARN_pytorch.carn.infer import infer

class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

cfg = {
    "group": 1,
    "ckpt_path": "./image_super_resolution/CARN_pytorch/checkpoint/carn.pth",
    "model": "carn",
    "scale": 4,
    "shave": 20,
}
cfg = Dict2Class(cfg)

module = importlib.import_module("image_super_resolution.CARN_pytorch.carn.model.{}".format(cfg.model))


def gen_sp_slic(I, region_size_=20, algo=cv2.ximgproc.SLICO):
    # Superpixel Generation ::  Slic superpixels compared to state-of-the-art superpixel methods
    num_iter = 4
    sp_slic = cv2.ximgproc.createSuperpixelSLIC(
        I, algorithm=algo, region_size=region_size_, ruler=10.0)
    sp_slic.iterate(num_iterations=num_iter)

    return sp_slic


def get_segmented_image(img, region_size=30, algo=cv2.ximgproc.SLICO):

    segment_obj = gen_sp_slic(img, region_size_=region_size, algo=algo)

    SP_labels = segment_obj.getLabels()
    segment_obj.enforceLabelConnectivity()
    num = segment_obj.getNumberOfSuperpixels()
    mask = segment_obj.getLabelContourMask()
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    not_mask = cv2.bitwise_not(mask)

    border_img = np.zeros(img.shape)
    border_img[:,:,0] = np.ones(img.shape[:-1])*255
    border_img = border_img.astype('uint8')
    segmented_image = cv2.bitwise_and(img, not_mask) + cv2.bitwise_and(border_img, mask)
    return segmented_image, SP_labels

def mouse_click_segmented(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:
        # font for right click event
        cv2.destroyWindow(f"Segmented Image({param})")

def mouse_click_image(event, x, y, flags, param):
    # to check if left mouse
    # button was clicked
    scale = 4
    if event == cv2.EVENT_LBUTTONDOWN:
        # font for left click event
        lbl = SP_labels[y, x]
        min_x = 1000
        max_x = -1
        min_y = 1000
        max_y = -1
        for x1 in range(max(x-50, 0), min(x+50, width - 1)):
            for y1 in range(max(y - 50, 0), min(y + 50, height - 1)):
                if SP_labels[y1, x1] == lbl:
                    min_x = min(min_x, x1)
                    max_x = max(max_x, x1)
                    min_y = min(min_y, y1)
                    max_y = max(max_y, y1)
        cropped = frame[min_y:max_y, min_x:max_x, :]

        tensor_cropped = torch.from_numpy(cropped)
        super_reolution_image = image_super_resolution(tensor_cropped)
        resized = cv2.resize(cropped, (cropped.shape[1]*scale, cropped.shape[0]*scale), cv2.INTER_CUBIC)
        resized = resized.astype('float32')
        stacked = np.hstack([resized/255, super_reolution_image])
        cv2.imshow(f"Segmented Image({lbl})", stacked)
        cv2.setMouseCallback(f'Segmented Image({lbl})', mouse_click_segmented, lbl)

def get_contour_image(image):
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(imgray, 100, 200)
    _, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c)>20]
    return cv2.drawContours(image, contours, -1, (0,255,0), 3)


def image_super_resolution(img):

    net = module.Net(multi_scale=True,
                     group=cfg.group)

    state_dict = torch.load(cfg.ckpt_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

    device = torch.device("cpu")
    net = net.to(device)

    lr = Image.fromarray(np.uint8(img))
    lr = lr.convert("RGB")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    return infer(net, device, transform(lr), cfg, return_img=True)

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument('--webcam', type=bool, default=False)
    parser.add_argument('--test_image', type=str, default="./rock_segmentation/sample.png")
    parser.add_argument('--segment', type=bool, default=True)
    parser.add_argument('--classify', type=bool, default=True)
    args = parser.parse_args()

    height, width = None, None

    if args.test_image is not None:
        frame = cv2.imread(args.test_image)
        height, width, _ = frame.shape
        segmented_image, SP_labels = get_segmented_image(frame)
        cv2.imwrite("segmented_output.png", segmented_image)
        cv2.imshow("Segmented Image", segmented_image)
        cv2.setMouseCallback('Segmented Image', mouse_click_image)
        cv2.waitKey(0)
    if args.webcam:
        capture = cv2.VideoCapture(0)
        while True:
            ret_val, frame = capture.read()
            if args.segment:
                segmented_image, _ = get_segmented_image(frame)
            cv2.imshow("Segmented Image", segmented_image)

            if args.classify:
                raise Exception("Not implemented yet")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
