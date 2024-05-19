import cv2
import argparse
import numpy as np

import numpy as np

def rgb_to_hsv(r, g, b):
    r = float(r)
    g = float(g)
    b = float(b)
    high = max(r, g, b)
    low = min(r, g, b)
    h, s, v = high, high, high

    d = high - low
    s = 0 if high == 0 else d/high

    if high == low:
        h = 0.0
    else:
        h = {
            r: (g - b) / d + (6 if g < b else 0),
            g: (b - r) / d + 2,
            b: (r - g) / d + 4,
        }[high]
        h /= 6

    return h, s, v

parser = argparse.ArgumentParser()
parser.add_argument('--camera', default=0)
args = parser.parse_args()

# define a video capture object 
vid = cv2.VideoCapture(int(args.camera)) 
print("Video device fetched")

count = 0
while(True): 
    count += 1
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 

    # Display the resulting frame 
    h,w = frame.shape[:2]
    rgb = frame[h//2, w//2][::-1].copy()
    hsv = rgb_to_hsv(*rgb)
    print("Centre color : ", rgb, " === ", hsv)
    frame[h//2 - 3:h//2 + 3, w//2-3:w//2+3] = np.array([0,0,255], dtype=np.uint8)
    cv2.imshow('Detected Image', frame) 
    
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'): 
        break
    if key & 0xFF == ord('s'): 
        with open("color_ranges.txt", "a") as f:
            f.write(f"RGB : {rgb} :: HSV {hsv}\n")

# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 