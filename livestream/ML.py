

import cv2


face_cascade = cv2.CascadeClassifier('/home/popikeyshen/js/haarcascade_frontalface_default.xml') 
smile_cascade = cv2.CascadeClassifier('/home/popikeyshen/js/haarcascade_smile.xml') 

# smile
def detect_smile(frame): 

    #cv2.imshow('frame',frame)
    #cv2.waitKey(1)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 

    smiles =[]
    for (x, y, w, h) in faces: 
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2) 
        roi_gray = gray[y:y + h, x:x + w] 
        roi_color = frame[y:y + h, x:x + w] 
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20) 
  
        for (sx, sy, sw, sh) in smiles: 
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2) 
    return smiles, frame
