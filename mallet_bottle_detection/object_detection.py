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
        self.prev_ann_img = None
        self.prev_conf = None

    
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
                        return_cpu=True,
                        log_state=False
                        ):
        xyxy_list, _, conf_list = self.detect(img, get_annotated_img=False, return_cpu=return_cpu)
        
        valid_detection = True
        if len(conf_list) == 0:
            valid_detection = False
        else:        
            idx = np.argmax(conf_list)
            xyxy = xyxy_list[idx]
            conf = conf_list[idx]
            prev_conf = conf
            conf = self.conf_mod(img, xyxy, conf)
            valid_detection = self.valid_detect(xyxy, conf)
            
        if not valid_detection and self.detection_state >= self.num_checks:
            self.detection_state += 2
            if self.detection_state > 2*self.num_checks:
                self.detection_state = 0
                return None, None, None
            else :
                return self.prev_xyxy, self.prev_ann_img, self.prev_conf
                
        elif not valid_detection:
            self.prev_xyxy = None
            self.prev_conf = None
            self.prev_ann_img = None
            self.detection_state = 0
            return None, None, None
        
        self.prev_xyxy = xyxy
        self.prev_conf = conf

        if self.detection_state < self.num_checks:
            self.detection_state = (self.detection_state + 1)
        
        if log_state:
            print("Detection state : ", self.detection_state)
        
        if self.detection_state >= self.num_checks:
            if get_annotated_img:
                annotator = Annotator(img)
                annotator.box_label(xyxy,f"{round(conf.item(), 2)}, d={round((conf-prev_conf).item(), 2)}", color=(255, 0, 0))
                annotated_img = annotator.im
            else :
                annotated_img = None
            
            self.prev_ann_img = annotated_img
            return xyxy, annotated_img, conf
        else :
            return None, None, None
                 
    
class MalletDetection(ObjectDetection):
    def __init__(self, num_checks=5, conf_thres=0.3, iou_thres=0.6) -> None:
        super().__init__()
        self.model = ultralytics.YOLO('models/best_mallet_close.pt')
        self.num_checks = num_checks
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        hue = 10
        self.lower_thres = np.array([hue-6,30,30]) 
        self.upper_thres = np.array([hue+6,255,255]) 
        self.mid_frac = 0.06
    
    def conf_mod(self, img, xyxy, conf):
        xyxy = xyxy.numpy().astype(np.int32)
        img_copy = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]].copy()
        hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV) 
        mask = cv2.inRange(hsv, self.lower_thres, self.upper_thres)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        max_cnt_area = -1
        max_cnt = None
        for cnt in contours :
            cnt_area = cv2.contourArea(cnt)
            if cnt_area > max_cnt_area:
                max_cnt = cnt 
                max_cnt_area = cnt_area
        hammer_img = np.zeros_like(mask)
        if max_cnt is not None:
            cv2.drawContours(hammer_img, [max_cnt], -1, (255, 255, 255), -1) 
        hammer_img = hammer_img.astype(np.float32)
        hammer_img /= 255
        frac = hammer_img.sum()/hammer_img.size
        print("frac : " ,frac)
        frac = np.tanh(10*(frac-self.mid_frac))
        delta_conf = (0.1 if frac > 0 else 0.2)*frac
        cv2.imshow("Mask", hammer_img)
        cv2.waitKey(5)
        print("conf + delta_conf :", conf + delta_conf)
        
        return torch.min(torch.tensor([conf + delta_conf, 1.0]))
        
    
    def detect(self, img, get_annotated_img=False, return_cpu=True):
        """returns xyxy (top left and bottom right), annotated img, confidence

        Args:
            img (numpy array): uint8 H,W,3
        """
        res = self.model.predict(img, conf=0.10, iou=0.05)[0]
        annotated_img = None
        
        if get_annotated_img:
            annotator = Annotator(img)
            for label in res.boxes.data:
                annotator.box_label(label[0:4],f"{round(label[-2].item(), 2)}", color=(255, 0, 0))
                    
            annotated_img = annotator.im

        if return_cpu and self.model.device != torch.device("cpu"):
            return res.boxes.xyxy.cpu(), annotated_img.cpu() if get_annotated_img else None, res.boxes.conf.cpu()
        
        return res.boxes.xyxy, annotated_img, res.boxes.conf            
        
        
    

class BottleDetection(ObjectDetection):
    def __init__(self) -> None:
        self.model = ultralytics.YOLO('models/best_mallet_close.pt')
    
    def detect(self, img):
        """returns xyxy (top left and bottom right), 

        Args:
            img (_type_): _description_
        """
        self.model.predict()
    