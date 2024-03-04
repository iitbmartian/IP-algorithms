# import ultralytics
# ultralytics.checks()

# from ultralytics import YOLO
# model = YOLO('models/best_mallet_detection.pt')


# model.predict(source="./videos/mallet.avi", show=True, conf=0.2, iou=0.99, save=False, show_conf = False, classes=[0])

import cv2 
import numpy as np 
  
# Create a VideoCapture object and read from input file 
import cv2 
import numpy as np  
  
cap = cv2.VideoCapture('./videos/bottle.avi') 
vid_writer = cv2.VideoWriter("bottle_detected.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, frameSize=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
  
# This drives the program into an infinite loop. 
while(1):         
    # Captures the live stream frame-by-frame 
    _, frame = cap.read()  
    # Converts images from BGR to HSV 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    lower_blue = np.array([80,100,120]) 
    upper_blue = np.array([130,255,255]) 
  
  
    # lower_blue = np.array([0,100,120]) 
    # upper_blue = np.array([10,255,255]) 
    # Here we are defining range of bluecolor in HSV 
    # This creates a mask of blue coloured  
    # objects found in the frame. 
    mask = cv2.inRange(hsv, lower_blue, upper_blue) 
  
    # The bitwise and of the frame and mask is done so  
    # that only the blue coloured objects are highlighted  
    # and stored in res 
    res = cv2.bitwise_and(frame,frame, mask= mask) 

    
    ## 
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    max_cnt_area = -1
    
    for cnt in contours :
        cnt_area = cv2.contourArea(cnt)
        if cnt_area > max_cnt_area:
            max_cnt = cnt 
            max_cnt_area = cnt_area
    
    print(max_cnt)
    if max_cnt_area < 15:
        vid_writer.write(frame)
        continue
    x,y,w,h = cv2.boundingRect(max_cnt) 
    eps = 2
    cv2.rectangle(frame, (x-eps,y-eps), (x+w + eps, y+h + eps), (0, 0, 255), 2) 
    cv2.putText(frame, 'bottle', (x-eps, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,12,255), 1)
    vid_writer.write(frame)
    cv2.imshow('frame',frame) 
    cv2.imshow('mask',mask) 
    cv2.imshow('res',res)        
    # This displays the frame, mask  
    # and res which we created in 3 separate windows. 
    k = cv2.waitKey(5) & 0xFF
    if k == 27: 
        break
  
# Destroys all of the HighGUI windows. 
cv2.destroyAllWindows() 
vid_writer.release()
# release the captured frame 
cap.release() 