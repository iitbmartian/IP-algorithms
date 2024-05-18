import cv2 
import os
from os import listdir
from os.path import isfile, join

import numpy as np
import json
import time
import imutils
from stitching import Stitcher 

import argparse


"""
HIKVISION Image Shape : (480, 640) 
Parse file should have these keys : "rotation_north": in degrees  "gps" "elevation": in meters "acc_range"
{
  "rotation_north": $rot, # degrees
  "gps": {
    "lat": $lat,
    "lon": $lon,
    "elevation": $elev,
  },
  "acc_range": {
    "rotation_north": $rot_acc,
    "lat": $lat_acc,
    "lon": $lon_acc,
    "elevation": $elev_acc
  }
}
"""
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=255)
  return result


# VID_PATH1 = os.environ.get('VID1', 0)
# VID_PATH2 = os.environ.get('VID2', 1)
STITCH_SAVE_PATH = os.environ.get("STITCH_SAVE_PATH", "stitched.png")

  
  
class ImageCombiner:
    def __init__(self, imgs, parse_file_path=None) -> None:
        self.imgs = imgs
        self.parse_res = None
        if parse_file_path is not None:
            with open(parse_file_path) as json_file:
                self.parse_res = json.load(json_file)
        
        self.text_params_gps = [(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA]
        self.text_params_elev = [(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA]
        self.text_acc_range = [(10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA]
        self.scale_text = [(10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA]

        
        self.cardinal_dir_img = cv2.imread("cardinal_dir.png")
        self.stitcher = Stitcher(detector="sift", confidence_threshold=0.2) 
        
    def combine_imgs(self):
        try:
            stitched = self.stitcher.stitch(self.imgs) 
        except Exception as e:
            print("Unable to stitch due to error :", e)
            return None
        
        return stitched
        
    def add_cardinal_dirs(self, img=None):
        if img is None:
            img = self.stitched.copy()
        else:
            img = img.copy()

        tmp = self.cardinal_dir_img.copy()
        axis_img = self.cardinal_dir_img.copy()
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY) 
        axis_img = cv2.cvtColor(axis_img, cv2.COLOR_BGR2GRAY) 

        tmp = rotate_image(tmp, self.parse_res["rotation_north"])
        axis_img = rotate_image(axis_img, self.parse_res["rotation_north"])

        tmp[tmp < 200] = 0
        tmp[tmp >= 200] = 255
        
        tmp_dilated = cv2.erode(tmp, np.ones((3,3)), iterations=1) 
        
        tmp = tmp.astype("float32")
        tmp /= 255.0
        
        if img.dtype.kind == 'u' or img.dtype.kind == 'i':
            img = img.astype("float32")
            img /= 255.0

        s = int(img.shape[1]/5)
        tmp = cv2.resize(tmp,(s, s))

        
        
        img[:s,-s:] *= np.expand_dims(tmp, axis=2)
        axis_img = cv2.resize(axis_img,(s, s))
        axis_img = axis_img.astype("float32") / 255.0

        # cv2.imshow("Overlayed", (1.0 - tmp))
        # cv2.imshow("Overlayed2", axis_img)
        # cv2.imshow("Overlayed3", np.clip(axis_img*(1.0 - tmp), a_min=0, a_max=1))
        
        # cv2.waitKey(0)
        
        img[:s,-s:] += np.expand_dims(axis_img*(1.0 - tmp), axis=2)
        carinal_angle_text = [(img.shape[1] - 70, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA]
        img = cv2.putText(img, f"{self.parse_res['rotation_north']} N", *carinal_angle_text) 
        

        return img
        
    
    def gps_loc(self, img):
        if self.parse_res is None:
            return 0
        # Using cv2.putText() method 
        image = cv2.putText(img, f"GPS Coords :: Lat : {self.parse_res['gps']['lat']} Lon : {self.parse_res['gps']['lon']}", *self.text_params_gps) 
        return image
    
    def add_elev(self, img):
        if self.parse_res is None:
            return 0
        # Using cv2.putText() method 
        image = cv2.putText(img, f"Elevation : {self.parse_res['gps']['elevation']}", *self.text_params_elev) 
        return image
    
    def accuracy_range(self, img):
        if self.parse_res is None:
            return 0
        # Using cv2.putText() method 
        image = cv2.putText(img, f"Acc. range angle : {self.parse_res['acc_range']['rotation_north']}", *self.text_acc_range) 
        return image
    
    def add_scale(self, img, scale_for_300px = 1):
        h,w = img.shape[:2]
        img = cv2.rectangle(img, (min(20, h//10), h-min(20, h//10)), (min(20, h//10)+300, h-min(20, h//10)+5), (255,255,255), -1) 
        scale_text = [(min(20, h//10), h-min(20, h//10) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA]
        
        img = cv2.putText(img, f"Scale : 300px-{scale_for_300px}m", *scale_text) 
        return img
    
    def parse_file():
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Directory of images to be stitched', default="./")
    parser.add_argument('--save-path', help='Save path for stitched image', default="./")
    args = parser.parse_args()
    
    imgs = sorted([os.path.join(args.root, f) for f in listdir(args.root) if isfile(join(args.root, f))])
    imgcomb = ImageCombiner(imgs=imgs, parse_file_path="parse_file.json")
    
    img = imgcomb.combine_imgs()
    if img is None:
        exit(1)
    
    img = imgcomb.accuracy_range(img)
    img = imgcomb.add_elev(img)
    img = imgcomb.gps_loc(img)
    img = imgcomb.add_cardinal_dirs(img)
    img = imgcomb.add_scale(img)

    
    cv2.imwrite(f"{str(time.time())}_final_pano.png", img)
    
    cv2.imshow("Overlayed", img)
    cv2.waitKey(0)
    

    # vid = cv2.VideoCapture(0) 
    
    # while(True): 
        
    #     # Capture the video frame 
    #     # by frame 
    #     ret, frame = vid.read() 
    #     print(frame.shape)
    #     # Display the resulting frame 
    #     cv2.imshow('frame', frame) 
        
    #     # the 'q' button is set as the 
    #     # quitting button you may use any 
    #     # desired button of your choice 
    #     if cv2.waitKey(1) & 0xFF == ord('q'): 
    #         break
    
    # # After the loop release the cap object 
    # vid.release() 
    # # Destroy all the windows 
    # cv2.destroyAllWindows() 