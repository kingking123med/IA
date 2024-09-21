import cv2
from tracker import *
import os
os.chdir("D:\object traking\object_tracking")
result = cv2.VideoWriter('ahmedd.mp4',  cv2.VideoWriter_fourcc(*'XVID'), 20, (250,250))
# Create tracker object 
cap = cv2.VideoCapture("highway.mp4")
# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50) # if history is big number it will be hight 
while True:
    ret, frame = cap.read()        
    if ret is not True:        
        break              
    height, width, _ = frame.shape
    # Extract Region of interest
    roi = frame[340: 720,500: 800]
    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements       
        area = cv2.contourArea(cnt)       
        if area > 100:       
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)              
            x, y, w, h = cv2.boundingRect(cnt)               
            detections.append([x, y, w, h])              
    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    result.write(frame)
    key = cv2.waitKey(30)
    if key == 27:

        break        
result.release()
cap.release()
cv2.destroyAllWindows()
 