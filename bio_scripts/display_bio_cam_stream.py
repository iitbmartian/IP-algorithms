import time
import argparse
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

from datetime import datetime
import cv2
import glob
import os
import numpy as np
DISPLAY_RESOLUTION_WIDTH = 1920

import time 
IM_H = None
DATE = ""

def on_created(event):
    global IM_H
    global DATE
    print("Created")
    time.sleep(3)
    img_path_arr = glob.glob(f"{os.path.join(event.src_path, '*')}")
    img_arr = []
    print(event.src_path)
    print(img_path_arr)
    for path in img_path_arr:
        if os.path.isfile(path):
            try:   
                img_arr.append(cv2.imread(path))
            except Exception as e:
                print(f"{path} is not an image. Or some other error. {e}")
    
    if len(img_arr) == 0:
        return
    
    IM_H = cv2.hconcat(img_arr)
    print(IM_H.shape)
    if IM_H.shape[1] > DISPLAY_RESOLUTION_WIDTH - 30:
        (h, w) = IM_H.shape[:2]
        dim = ((DISPLAY_RESOLUTION_WIDTH-30), int(h *(DISPLAY_RESOLUTION_WIDTH-30)/w))
        IM_H = cv2.resize(IM_H, dim)
    
    DATE = datetime.fromtimestamp(float(os.path.split(event.src_path)[1])).strftime("%A, %B %d, %Y %I:%M:%S")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patterns', nargs='+', help='<Required> Set flag', default=["*"])
    parser.add_argument('--path', type=str, default="fetched_images/images")
    parser.add_argument('--savepath', type=str, default="bio_saved_imgs")
    
    args = parser.parse_args()
    patterns = args.patterns
    ignore_patterns = None
    ignore_directories = False
    case_sensitive = False
    my_event_handler = PatternMatchingEventHandler(patterns, ignore_patterns, ignore_directories, case_sensitive)
    my_event_handler.on_created = on_created
    
    path = args.path
    go_recursively = False
    my_observer = Observer()
    my_observer.schedule(my_event_handler, path, recursive=go_recursively)  
    my_observer.start()
    save_img_idx = 0
    
    text_params = [(10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA]
    os.makedirs(args.savepath, exist_ok=True)
    while True:
        if IM_H is not None:
            img = cv2.putText(IM_H, DATE, *text_params) 
            cv2.imshow("Bio-cam", img)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                exit()
            elif key == ord('s'):
                cv2.imwrite(os.path.join(args.savepath, f"bio_cams_{str(save_img_idx)}.png"), IM_H)
                save_img_idx += 1
                
                