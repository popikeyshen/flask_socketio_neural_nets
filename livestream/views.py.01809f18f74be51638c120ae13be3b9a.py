# -*- coding: utf-8 -*-

from __future__ import print_function
from livestream import app, socketio
from flask import render_template, Response,request,session
from flask_socketio import emit,join_room


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
    emit('response',
         {'data': message['data']})
    print(message['data'])


@socketio.on('livevideo', namespace='/live')
def test_live(message):
    """Video stream reader."""
    emit('response',{'data': "new Image"})
