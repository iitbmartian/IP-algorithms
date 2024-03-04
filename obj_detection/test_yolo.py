import ultralytics
ultralytics.checks()

from ultralytics import YOLO
model = YOLO('models/best_mallet_detection.pt')


model.predict(source="./videos/mallet.avi", show=True, conf=0.2, iou=0.99, save=False, show_conf = False, classes=[0])
