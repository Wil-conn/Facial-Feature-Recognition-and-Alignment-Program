from facenet_pytorch import MTCNN
import numpy as np
import cv2 as cv
import time

#trackbar requires a callback function even though we aren't actually going to be using the callback function in the program so we create an empty callback function
def nothing(x):
    pass

class face_detect:
    #f is the factor the MTCNN uses, images is a list of images we want to allign and mode is used to designate which facial feature we want to allign on
    def __init__(self, f, images, mode):
        self.mtcnn = MTCNN(factor = f)
        self.images = images
        self.mode = mode
        self.landmarks = {} #dictionary holding the landmarks (i.e eyes location, mouth location, nose location etc.) of every photo. to get the landmarks of the first photo, use key "0"

    def detect(self):
        for idx, image in np.ndenumerate(self.images):
            boxes, probs, landmarks = self.mtcnn.detect(image, landmarks = True)
            self.landmarks[str(idx[0])] = landmarks #not entirely sure why idx is a tuple in the form (index , ). using idx[0] fixes this by just getting the index number
        if boxes is None:
            return False

        print(self.landmarks.get("0"))
        #self.landmarks = landmarks
        '''
        if self.mode == 'eyes':
            angle_of_eyes_tilt
        '''

#    def angle_of_eyes_tilt(left_eye_coord, right_eye_coord):

    def display(self):
        idx = 0
        cv.namedWindow('slideshow')
        trackbar_name = 'image # %d' % idx
        cv.createTrackbar(trackbar_name, 'slideshow', 0, len(self.images), nothing)
        '''
        for idx, image in np.ndenumerate(self.images):
            cv.imshow('image' + str(idx), image)
        '''
        while idx < len(self.images):
            print(idx)
            img = self.images[cv.getTrackbarPos(trackbar_name, 'slideshow')]

            height = 600
            #dim = (int(height/self.images[idx].shape[0] * self.images[idx].shape[1]), height)
            #img = cv.resize(self.images[idx], dim)
            #cv.imshow('image', self.images[idx])
            cv.imshow('slideshow', img)
            k = cv.waitKey(500)





