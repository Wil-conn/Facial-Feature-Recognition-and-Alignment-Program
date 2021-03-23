from facenet_pytorch import MTCNN
import numpy as np
import cv2 as cv
import time

BACKGROUND_DIMS = 500

'''
A note on landmarks array structure:

left eye coordinates are accessed by creating a tuple consisting of: (landmarks[0][0][0] , landmarks[0][0][1])
right eye coordinates are aceesed by creating a tuple consisting of: (landmarks[0][1][0] , landmarks[0][1][1]

'''


#trackbar requires a callback function even though we aren't actually going to be using the callback function in the program so we create an empty callback function
def nothing(x):
    pass

class face_detect:
    #f is the factor the MTCNN uses, images is a list of images we want to allign and mode is used to designate which facial feature we want to allign on
    def __init__(self, f, images, mode):
        self.mtcnn = MTCNN(factor = f)
        self.images = images
        self.mode = mode

        #dictionary holding the landmarks (i.e eyes location, mouth location, nose location etc.) of every photo.
        #to get the landmarks of the first photo, use key "0"
        self.landmarks = {}

    def detect(self):
        for idx, image in np.ndenumerate(self.images):
            boxes, probs, landmarks = self.mtcnn.detect(image, landmarks = True)

            #not entirely sure why idx is a tuple in the form (index , ).
            #using idx[0] fixes this by just getting the index number
            self.landmarks[str(idx[0])] = landmarks

            #if we are not able to detect the features for a photo we update out images array to remove that photo and we
            #continue to the next iteration
            if boxes is None:
                self.images = np.delete(self.images, idx[0])
                pass
            else:
                #cv.circle(image, (50,50), 10, (255,0,0))
                cv.circle(image, (landmarks[0][0][0], landmarks[0][0][1]), 5, (0, 255, 0))
                cv.circle(image, (landmarks[0][1][0], landmarks[0][1][1]), 5, (0, 255, 0))

        print(self.landmarks.get("0"))
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

        while idx < len(self.images):

            #we will be placing the image on top of a black background of a fixed size
            #we do this because not all the images have the same dimensions and that causes problems with the trackbar
            #doing this will also make life 100x easier when we begin applying our transformations on the photos
            background = np.zeros((BACKGROUND_DIMS, BACKGROUND_DIMS, 3), np.uint8)

            #uses the value of our trackbar to get that photo and place it on screen
            img = self.images[cv.getTrackbarPos(trackbar_name, 'slideshow')]

            #gets the offset so that we place our photos in the center of the background image
            x_offset = int((BACKGROUND_DIMS - img.shape[1])/2)
            y_offset = int((BACKGROUND_DIMS - img.shape[0])/2)

            #places the image on the background
            background[y_offset: y_offset + img.shape[0], x_offset: x_offset + img.shape[1]] = img


            height = 600
            #dim = (int(height/self.images[idx].shape[0] * self.images[idx].shape[1]), height)
            #img = cv.resize(self.images[idx], dim)
            #cv.imshow('image', self.images[idx])
            cv.imshow('slideshow', background)
            k = cv.waitKey(500)





