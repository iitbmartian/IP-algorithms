import cv2 
import numpy as np
import os
import datetime
import time

GET_CAM_BY_ID = os.environ.get("GET_CAM_BY_ID", default=False)

CAM_IDS = [
    "",
    "",
    ""
]

class ImageSaver:
    def __init__(self, num_cameras = 3, path="./images", freq_inv = 5) -> None:
        self.num_cameras = num_cameras
        self.freq_inv = freq_inv
        self.path = path
        
        os.makedirs(self.path, exist_ok=True)
        self.cam_objs = []
        if GET_CAM_BY_ID:
            for path in CAM_IDS:
                self.cam_objs.append(cv2.VideoCapture(path))
        else:
            for idx in range(2*num_cameras + 5):
                if len(self.cam_objs) >= num_cameras:
                    break
                cap = cv2.VideoCapture(idx) 
                print(idx)
                if cap is None or not cap.isOpened():
                    print('Warning: unable to open video source: ', idx)
                    continue 
                self.cam_objs.append(cap)
        
        if len(self.cam_objs) == 0:
            exit()
        if len(self.cam_objs) < num_cameras:
            print(f"Number of expected cameras is {num_cameras}, but found only {len(self.cam_objs)}")  

    def save_images(self):
        dt = time.time()
        savepath = os.path.join(self.path, str(dt))
        os.mkdir(savepath)
        for i, elem in enumerate(self.cam_objs):
            ret, frame = elem.read()
            if not ret:
                print("Error reading feed {i}")
                continue
            cv2.imwrite(os.path.join(savepath, str(i) + ".png"), frame)
            
    
    def save_stream(self):
        
        while True:
            dt = time.time()
            savepath = os.path.join(self.path, str(dt))
            os.mkdir(savepath)
            for i, elem in enumerate(self.cam_objs):
                ret, frame = elem.read()
                if not ret:
                    print("Error reading feed {i}")
                    continue
                cv2.imwrite(os.path.join(savepath, str(i) + ".png") , frame)
                print("Image saved")
            time.sleep(self.freq_inv)
            
if __name__ == "__main__":
    img_saver = ImageSaver(num_cameras=3)
    img_saver.save_stream()
    