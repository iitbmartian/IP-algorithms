import ultralytics
from ultralytics.utils.plotting import Annotator
import numpy as np

import cv2

ultralytics.checks()

class ObjectDetection:
    def __init__(self) -> None:
        pass 
    
    def detect(self):
        raise NotImplementedError
    
    def stateful_detect(self
                        num_checks=5,
                        conf_thres=0.2
                        ):
        
    

class MalletDetection(ObjectDetection):
    def __init__(self) -> None:
        self.model = ultralytics.YOLO('models/best_mallet_detection.pt')
    
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
    