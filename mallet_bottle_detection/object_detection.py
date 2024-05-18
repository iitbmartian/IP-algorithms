import ultralytics
from ultralytics.utils.plotting import Annotator
import numpy as np
import torch
import cv2

ultralytics.checks()

class ObjectDetection:
    def __init__(self) -> None:
        """
         states are 0 to num_checks
        """
        self.detection_state = 0
        self.prev_xyxy = None
    
    def detect(self):
        raise NotImplementedError
    
    @staticmethod
    def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou
    
    def valid_detect(self, xyxy, conf):
        if self.prev_xyxy is not None :
            return ObjectDetection.bb_intersection_over_union(self.prev_xyxy, xyxy) > self.iou_thres and conf > self.conf_thres
        else :
            return conf > self.conf_thres
        
    def stateful_detect(self,
                        img,
                        get_annotated_img=False, 
                        return_cpu=True
                        ):
        xyxy_list, _, conf_list = self.detect(img, get_annotated_img=False, return_cpu=True)
        idx = np.argmax(conf_list)
        xyxy = xyxy_list[idx]
        conf = conf_list[idx]
        if not self.valid_detect(conf, xyxy):
            self.detection_state = 0
            return None, None, None
        
        self.detection_state = (self.detection_state + 1) % self.num_checks
        
        if self.detection_state == (self.num_checks - 1):
            return self.detect(img, get_annotated_img, return_cpu)
                 
    
class MalletDetection(ObjectDetection):
    def __init__(self, num_checks=5, conf_thres=0.4, iou_thres=0.8) -> None:
        self.model = ultralytics.YOLO('models/best_mallet_detection.pt')
        self.num_checks = num_checks
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
    
    def detect(self, img, get_annotated_img=False, return_cpu=True):
        """returns xyxy (top left and bottom right), annotated img, confidence

        Args:
            img (numpy array): uint8 H,W,3
        """
        res = self.model.predict(img, iou=0.05)[0]
        annotated_img = None
        
        if get_annotated_img:
            annotator = Annotator(img)
            for label in res.boxes.data:
                annotator.box_label(label[0:4],f"{round(label[-2], 2)}")
            annotated_img = annotator.im

        if return_cpu and self.model.device == torch.device("cpu"):
            return res.boxes.xyxy.cpu(), annotated_img.cpu() if get_annotated_img else None, res.boxes.conf.cpu()
        
        
        return res.boxes.xyxy, annotated_img, res.boxes.conf            
        
        
    

class BottleDetection(ObjectDetection):
    def __init__(self) -> None:
        self.model = ultralytics.YOLO('models/best_mallet_detection.pt')
    
    def detect(self, img):
        """returns xyxy (top left and bottom right), 

        Args:
            img (_type_): _description_
        """
        self.model.predict()
    