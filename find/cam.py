import os
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
import glob
import pickle
from datetime import datetime
from flask import Flask, request, render_template

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in glob.glob("static/feature/*"):
    features.append(pickle.load(open(feature_path, 'rb')))
    img_paths.append('static/img/' + os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')



from PIL import Image
import numpy as np
import cv2
import imutils

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img = Image.fromarray(frame, "RGB")

    query = fe.extract(img)
    dists = np.linalg.norm(features - query, axis=1)  # Do search
    ids = np.argsort(dists)[:6] # Top 30 results
    scores = [(dists[id], img_paths[id]) for id in ids]

    #print(scores[0][1])
    out1 = cv2.imread(scores[0][1])
    out2 = cv2.imread(scores[1][1])
    out3 = cv2.imread(scores[2][1])

    out1 = imutils.resize(out1, width=400)    
    out2 = imutils.resize(out2, width=400)    
    out3 = imutils.resize(out3, width=400)

    cv2.imshow('out1',out1)
    cv2.imshow('out2',out2)
    cv2.imshow('out3',out3)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



