from PIL import Image
import numpy
import cv2

class Makeup_artist(object):
    def __init__(self):
        pass

    def apply_makeup(self, img):

        buf = img
        #pil_image = Image.open(buf).convert('RGB') 
        cv_im = numpy.array(buf) # cv2.cvtColor( , cv2.COLOR_RGB2BGR)
        #cv_im = numpy.array(buf)
        
        cv2.imshow('image',cv_im)
        cv2.waitKey(1)

	img = Image.fromarray(cv_im, "RGB")
        #img = Image.fromstring("RGB", cv.GetSize(cv_im), cv_im.tostring())

        return img.transpose(Image.FLIP_LEFT_RIGHT)
