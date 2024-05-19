import object_detection 
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--streaming',action="store_true", default=False)
parser.add_argument('--camera', default=0)
args = parser.parse_args()

detector = object_detection.MalletDetection(conf_thres=0.20)


if not args.streaming:
    xyxy, annotated_img, conf = detector.detect(cv2.imread("./test_images/mallet_test_img.jpg"))
    cv2.imshow("Detected Mallet", annotated_img)
    
else :
    # define a video capture object 
    vid = cv2.VideoCapture(int(args.camera)) 
    
    while(True): 
        
        # Capture the video frame 
        # by frame 
        ret, frame = vid.read() 
    
        # Display the resulting frame 
        xyxy, annotated_img, conf = detector.stateful_detect(frame, get_annotated_img=True, log_state=True)
        # xyxy, annotated_img, conf = detector.detect(frame, get_annotated_img=True)
        
        cv2.imshow('Detected Image', annotated_img if annotated_img is not None else frame) 
        
        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    # After the loop release the cap object 
    vid.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 