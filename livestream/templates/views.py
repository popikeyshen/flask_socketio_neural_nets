# -*- coding: utf-8 -*-

from __future__ import print_function
from livestream import app, socketio
from flask import render_template, Response,request,session
from flask_socketio import emit,join_room

from io import BytesIO
from PIL import Image
import numpy as np
import base64
import cv2
#from utils import base64_to_pil_image, pil_image_to_base64
#from makeup_artist import Makeup_artist
#from camera import Camera
#camera = Camera(Makeup_artist())

@app.route("/")
def home():
    """The home page with webcam."""
    return render_template('index.html')

@socketio.on('connect', namespace='/live')
def test_connect():
    """Connect event."""
    # room = session.get('room')
    # join_room(room)
    print('Client wants to connect.',request.sid)
    emit('response', {'data': 'OK'})


@socketio.on('disconnect', namespace='/live')
def test_disconnect():
    """Disconnect event."""
    print('Client disconnected')


@socketio.on('event', namespace='/live')
def test_message(message):
    """Simple websocket echo."""
    emit('response', {'data': message['data']})
    print(message['data'])


import ML
# import ML2
# import ML3

@socketio.on('livevideo', namespace='/live')
def test_live(message):
    """Video stream reader."""
    data = message['data']
    data = data.split(",")[1]

    img  = Image.open(BytesIO(base64.b64decode(data)))
    cv_im = np.array(img) 
    cv_im = cv2.cvtColor( cv_im, cv2.COLOR_BGR2RGB)

    # 1) detect smiles
    smiles, cv_im =ML.detect_smile(cv_im)
    cv_im = cv2.cvtColor( cv_im, cv2.COLOR_RGB2BGR) 
    img = Image.fromarray(cv_im, "RGB")


    buf = BytesIO()
    img.save(buf, format="JPEG")
    out = base64.b64encode(buf.getvalue())
    emit('response',{'data': {
            "smiles":
            ,"out":out
        }
    })
    
##########################################
    # data = message['data']
    # data = data.split(",")[1]

    # img  = Image.open(BytesIO(base64.b64decode(data)))
    # cv_im = np.array(img) 
    # cv_im = cv2.cvtColor( cv_im, cv2.COLOR_BGR2RGB)

    # # 1) detect smiles
    # smiles, cv_im =ML.detect_smile(cv_im)    

    # 2) detect likes
    #rectangles = ML2.detect_like(cv_im)
    #draw = cv_im.copy()
    #for rectangle in rectangles:
    #     cv2.putText(draw,str(rectangle[4]),(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
    #     cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
    #cv2.imshow("test",draw)

    # 3) detect pupils
    #cv_im = ML3.pupils(cv_im)

    # show image and data
    #cv2.imshow('image',cv_im)
    #cv2.waitKey(1)
    #print(smiles)

    #emit('response',{'data': cv_im})
    # send image to site
    
###########################################

    #camera.enqueue_input(data)
    #frame = camera.get_frame() #pil_image_to_base64(camera.get_frame())
    #    yield (b'--frame\r\n'
    #           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    #print(message['data'])






