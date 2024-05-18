import detect_obj
import cv2
detector = detect_obj.MalletDetection()

xyxy, annotated_img, conf = detector.detect(cv2.imread("./test_images/mallet_test_img.jpg"))