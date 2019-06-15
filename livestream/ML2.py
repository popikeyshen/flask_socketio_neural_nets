import sys
sys.path.append('.')
sys.path.append('/home/popikeyshen/Lib/caffe/python')
import tools_matrix as tools
import caffe
import cv2
import numpy as np


deploy = '/home/popikeyshen/js/12net.prototxt'
#caffemodel = 'my_likes.caffemodel' 
#caffemodel = 'solver12_iter_200000.caffemodel'
#caffemodel = 'solver12_xavier_iter_100000.caffemodel'  # good
#caffemodel = 'solver12_iter_120000.caffemodel'
#caffemodel = 'solver12_xavier_all_iter_200000.caffemodel' #last 
caffemodel = '/home/popikeyshen/js/solver12_iter_100000full.caffemodel'
caffemodel = '/home/popikeyshen/js/solver12_iter_10000.caffemodel' #good bad apple
net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = '/home/popikeyshen/js/24net.prototxt'
#caffemodel = '24net.caffemodel'
#caffemodel = 'solver24_iter_50000.caffemodel'
#caffemodel = 'solver24_iter_20000.caffemodel'
caffemodel = '/home/popikeyshen/js/solver24_iter_100000.caffemodel'  # xavier all
net_24 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = '/home/popikeyshen/js/48net.prototxt'
#caffemodel = '48net.caffemodel'
#caffemodel = 'solver48_iter_20000.caffemodel'  # good
#caffemodel = 'solver48_iter_70000.caffemodel'  #list_all
#caffemodel = 'solver48_iter_100000.caffemodel' #list_all
#caffemodel = 'solver48_iter_250000.caffemodel' #list_all  # last
caffemodel = '/home/popikeyshen/js/solver48_iter_100000full.caffemodel'

net_48 = caffe.Net(deploy,caffemodel,caffe.TEST)

#tools.calculateScales(frame)
scales=[0.9,0.5,0.36,0.26,0.18,0.13,0.09,0.06]


def detect_like(img):
    
    draw = img.copy()

    threshold = [0.19,0.4,0.4]
    caffe_img = (img.copy()-127.5)/128
    origin_h,origin_w,ch = caffe_img.shape

    out = []
    for scale in scales:
        hs = int(origin_h*scale)
        ws = int(origin_w*scale)
        scale_img = cv2.resize(caffe_img,(ws,hs))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_12.blobs['data'].reshape(1,3,ws,hs)
        net_12.blobs['data'].data[...]=scale_img
	caffe.set_device(0)
	caffe.set_mode_gpu()
	out_ = net_12.forward()
        out.append(out_)
    image_num = len(scales)
    rectangles = []
    for i in range(image_num):    
        cls_prob = out[i]['prob1'][0][1]
        roi      = out[i]['conv4-2'][0]
        out_h,out_w = cls_prob.shape
        out_side = max(out_h,out_w)
        rectangle = tools.detect_face_12net(cls_prob,out_side,1/scales[i],origin_w,origin_h,threshold[0])
        rectangles.extend(rectangle)
    rectangles = tools.NMS(rectangles,0.7,'iou')

    #print("like")
    #print(rectangles)

    
    if len(rectangles)==0:
        return rectangles, draw
    '''
    net_24.blobs['data'].reshape(len(rectangles),3,24,24)
    crop_number = 0
    for rectangle in rectangles:

        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img,(24,24))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_24.blobs['data'].data[crop_number] =scale_img 
        crop_number += 1
    out = net_24.forward()
    cls_prob = out['prob1']
    roi_prob = out['conv5-2']
    rectangles = tools.filter_face_24net(cls_prob,roi_prob,rectangles,origin_w,origin_h,threshold[1])

    #print("lik1")
    #print(rectangles)

    
    if len(rectangles)==0:
        return rectangles, draw
    net_48.blobs['data'].reshape(len(rectangles),3,48,48)
    crop_number = 0
    for rectangle in rectangles:

        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img,(48,48))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_48.blobs['data'].data[crop_number] =scale_img 
        crop_number += 1
    out = net_48.forward()
    cls_prob = out['prob1']
    roi_prob = out['conv6-2']
    pts_prob = out['conv6-3']
    rectangles = tools.filter_face_48net(cls_prob,roi_prob,pts_prob,rectangles,origin_w,origin_h,threshold[2])
    '''
    #print("like2")
    #print(rectangles)

    for rectangle in rectangles:
    #    cv2.putText(img,str(rectangle[4]),(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(0,0,255),1)

    cv2.imshow('draw',draw)
    cv2.waitKey(1)

    return rectangles, draw


