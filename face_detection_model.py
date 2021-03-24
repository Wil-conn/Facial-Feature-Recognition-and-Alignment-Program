from facenet_pytorch import MTCNN
import numpy as np
import cv2 as cv
import math
import time

BACKGROUND_DIMS = 500

'''
A note on landmarks array structure:

left eye coordinates are accessed by creating a tuple consisting of: (landmarks[0][0][0] , landmarks[0][0][1])
right eye coordinates are aceesed by creating a tuple consisting of: (landmarks[0][1][0] , landmarks[0][1][1]

nose coordinates are accessed by creating a tuple consisting of: (landmarks[0][2][0], landmarks[0][2][1])

right mouth coordinates are accessed by creating a tuple consisting of: (landmarks[0][3][0], landmarks[0][3][1])
left mouth coordinates are accessed by creating a tuple consisting of: (landmarks[0][4][0], landmarks[0][4][1])

'''


# trackbar requires a callback function even though we aren't actually going to
# be using the callback function in the program so we create an empty callback function
def nothing(x):
    pass


class face_detect:
    # f is the factor the MTCNN uses, images is a list of images we want
    # to allign and mode is used to designate which facial feature
    # we want to allign on
    def __init__(self, f, images, mode):
        self.mtcnn = MTCNN(factor = f)
        self.images = images
        self.mode = mode

        # dictionary holding the landmarks (i.e eyes location, mouth location,
        # nose location etc.) of every photo.
        # to get the landmarks of the first photo, use key "0"
        self.landmarks = {}

    def detect(self):
        for idx, image in np.ndenumerate(self.images):

            # gray and mask are for testing
            # gray = cv.cvtColor(~image, cv.COLOR_BGR2GRAY)
            # mask = cv.threshold(gray, 220, 255, cv.THRESH_BINARY)[1]

            # for testing
            # contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            # cv.drawContours(mask, contours, -1, (0, 255, 0), 2)

            boxes, probs, landmarks = self.mtcnn.detect(image, landmarks = True)

            print(probs)

            # not entirely sure why idx is a tuple in the form (index , ).
            # using idx[0] fixes this by just getting the index number
            self.landmarks[str(idx[0])] = landmarks

            # if we are not able to detect the features for a photo we update
            # out images array to remove that photo and we continue
            # to the next iteration
            if boxes is None:
                self.images = np.delete(self.images, idx[0])
                pass
            else:
                cv.rectangle(image, (boxes[0][0], boxes[0][1]), (boxes[0][2], boxes[0][3]), (0, 255, 0), thickness=1)
                # Drawing circles on the eyes. just used for testing
                cv.circle(image, (landmarks[0][0][0], landmarks[0][0][1]), 3, (0, 255, 0))
                cv.circle(image, (landmarks[0][1][0], landmarks[0][1][1]), 3, (0, 255, 0))
                # Drawing circles on nose. just used for testing
                cv.circle(image, (landmarks[0][2][0], landmarks[0][2][1]), 3, (0, 255, 0))
                # Drawing circles on mouth. just used for testing
                cv.circle(image, (landmarks[0][3][0], landmarks[0][3][1]), 3, (0, 255, 0))
                cv.circle(image, (landmarks[0][4][0], landmarks[0][4][1]), 3, (0, 255, 0))

                eye_tilt_angle = self.get_angle_of_tilt((landmarks[0][0][0], landmarks[0][0][1]),
                                                        (landmarks[0][1][0], landmarks[0][1][1]))

                mouth_tile_angle = self.get_angle_of_tilt((landmarks[0][3][0], landmarks[0][3][1]),
                                                        (landmarks[0][4][0], landmarks[0][4][1]))

                cv.putText(image, str(eye_tilt_angle), (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)
                cv.putText(image, str(mouth_tile_angle), (10, 300), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)

        print(self.landmarks.get("0"))
        '''
        if self.mode == 'eyes':
            angle_of_eyes_tilt
        '''

#    def angle_of_eyes_tilt(left_eye_coord, right_eye_coord):

    # computes the angle of tilt between 2 sets of coordinates. used for computer eye tilt and mouth tilt
    def get_angle_of_tilt(self, left_coord, right_coord):
        # to solve for the angle we need 2 sides, so we calculate the hypotenuse and the adjacent side from the angle
        hype = math.sqrt((left_coord[0] - right_coord[0]) ** 2 + (left_coord[1] - right_coord[1]) ** 2)
        adj = right_coord[0] - left_coord[0]

        # if right eye's y is higher than the left's then we say it has an 'upwards' tilt
        if left_coord[1] > right_coord[1]:
            print("left eye y: " + str(left_coord[1]))
            print("right eye y: " + str(right_coord[1]))
            print("eyes have an 'upwards' angler")
            angle = math.acos(adj/hype)
            return (math.degrees(angle))

        # if right eye's y is lower than left eye's y, it has a 'downwards' tilt
        elif left_coord[1] < right_coord[1]:
            print("left eye y: " + str(left_coord[1]))
            print("right eye y: " + str(right_coord[1]))
            print("eyes have an 'downwards' angler")
            angle = math.acos(adj/hype)
            return (math.degrees(angle))

    def display(self):
        idx = 0

        cv.namedWindow('slideshow')
        trackbar_name = 'image # %d' % idx
        cv.createTrackbar(trackbar_name, 'slideshow', 0, len(self.images)-1, nothing)

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





