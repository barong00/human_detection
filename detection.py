import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8l.pt")

gender_net = cv2.dnn.readNetFromCaffe("C:/Users/boran/Desktop/Dosyalar/python/human_detection/deploy_gender.prototxt","C:/Users/boran/Desktop/Dosyalar/python/human_detection/gender_net.caffemodel")
age_net = cv2.dnn.readNetFromCaffe("C:/Users/boran/Desktop/Dosyalar/python/human_detection/deploy_age.prototxt",
                                    "C:/Users/boran/Desktop/Dosyalar/python/human_detection/age_net.caffemodel")
GENDER_LIST = ['Male','Female']
AGE_LIST = ['(0-2)','(4-6)','(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)

    # detected objects
    detections = results[0].boxes

    for detection in detections:
        
        if int(detection.cls) == 0:   
            x1, y1, x2, y2 = map(int, detection.xyxy[0])  
            person_roi = frame[y1:y2, x1:x2]
            faces = face_cascade.detectMultiScale(person_roi, scaleFactor = 1.1, minNeighbors=5)

            for (fx,fy,fw,fh) in faces:
                face = person_roi[fy:fy+fh, fx:fx+fw]

                # Gender prediction
                blob = cv2.dnn.blobFromImage(face,1.0, (227,227), (104.0,177.0,123.0), swapRB = False)
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = GENDER_LIST[gender_preds[0].argmax()]

                # age prediction
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = AGE_LIST[age_preds[0].argmax()]

                label = f"{gender}, {age}"
                cv2.putText(frame, label, (x1 + fx, y1 + fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.rectangle(frame, (x1 + fx, y1 + fy), (x1 + fx + fw, y1 + fy + fh), (0, 255, 0), 2)




    
    cv2.imshow("Frame", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
