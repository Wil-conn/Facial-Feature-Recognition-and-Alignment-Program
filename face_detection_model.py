from facenet_pytorch import MTCNN
import numpy as np
import cv2 as cv
import math
import time

BACKGROUND_DIMS = 600

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
        self.transformed = np.empty(images.shape, dtype=object)
        self.mode = mode

        # dictionary holding the landmarks (i.e eyes location, mouth location,
        # nose location etc.) of every photo.
        # to get the landmarks of the first photo, use key "0"
        self.landmarks = []

        # dictionary holding mouth angle and direction of every photo.
        # to get the mouth info of the first photo, use key "0"
        self.mouth = []

        # dictionary holding eye angle and direction of every photo.
        # to get the eye info of the first photo, use key "0"
        self.eye = []

        self.nose = []

    def detect(self):
        #for idx, image in np.ndenumerate(self.images):
        idx = 0
        for image in self.images:

            # gray and mask are for testing
            # gray = cv.cvtColor(~image, cv.COLOR_BGR2GRAY)
            # mask = cv.threshold(gray, 220, 255, cv.THRESH_BINARY)[1]

            # for testing
            # contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            # cv.drawContours(mask, contours, -1, (0, 255, 0), 2)

            boxes, probs, landmarks = self.mtcnn.detect(image, landmarks = True)
            

            # not entirely sure why idx is a tuple in the form (index , ).
            # using idx[0] fixes this by just getting the index number
            if type(landmarks) is np.ndarray:
                self.landmarks.append(landmarks)


            # if we are not able to detect the features for a photo we update
            # out images array to remove that photo and we continue
            # to the next iteration
            if boxes is None:
                self.images = np.delete(self.images, idx)
                self.transformed = np.delete(self.transformed, idx)
                idx -= 1
            else:
                eye_tilt_angle, eye_tilt_dir, eye_distance = self.get_angle_of_tilt((landmarks[0][0][0], landmarks[0][0][1]),
                                                        (landmarks[0][1][0], landmarks[0][1][1]))

                mouth_tilt_angle, mouth_tilt_dir, mouth_distance = self.get_angle_of_tilt((landmarks[0][3][0], landmarks[0][3][1]),
                                                        (landmarks[0][4][0], landmarks[0][4][1]))
                #get midpoints of eyes and mouth
                eye_mid = self.get_mid_point(landmarks[0][0], landmarks[0][1])
                mouth_mid = self.get_mid_point(landmarks[0][3], landmarks[0][4])

                
                self.eye.append([eye_tilt_angle, eye_tilt_dir, eye_distance, eye_mid])
                self.mouth.append([mouth_tilt_angle, mouth_tilt_dir, mouth_distance, mouth_mid])
                self.nose.append([landmarks[0][2][0], landmarks[0][2][1]])


            idx += 1


    # computes the angle of tilt between 2 sets of coordinates. used for computer eye tilt and mouth tilt
    def get_angle_of_tilt(self, left_coord, right_coord):
        # to solve for the angle we need 2 sides, so we calculate the hypotenuse and the adjacent side from the angle
        hype = math.sqrt((left_coord[0] - right_coord[0]) ** 2 + (left_coord[1] - right_coord[1]) ** 2)
        adj = right_coord[0] - left_coord[0]
        # if right eye's y is higher than the left's then we say it has an 'upwards' tilt, return 1
        if left_coord[1] > right_coord[1]:
            angle = math.acos(adj/hype)
            return (math.degrees(angle), 1, hype)

        # if right eye's y is lower than left eye's y, it has a 'downwards' tilt, return 0
        elif left_coord[1] < right_coord[1]:
            angle = math.acos(adj/hype)
            return (math.degrees(angle), 0, hype)

    def display(self):
        idx = 0
        cv.namedWindow('slideshow')
        trackbar_name = 'image # %d' % idx
        cv.createTrackbar(trackbar_name, 'slideshow', 0, len(self.images)-1, nothing)

        #calculate globals to be used
        avg_eye_dist = self.get_avg_dist(self.eye)#average distance between 2 eyes
        avg_mouth_dist = self.get_avg_dist(self.mouth)#average distance between mouth edges
        avg_eye_mid = self.get_avg_mid(self.eye)#average midpoint between eyes
        avg_nose_loc = self.get_avg_loc(self.landmarks, 2)#average nose location
        avg_mouth_mid = self.get_avg_mid(self.mouth)#average midpoint of mouth
        max_img_size = self.get_max_size(self.images)#largest image size
        avg_arr = self.get_avg_pts(self.landmarks)#average location of all 5 points

        
        for i in range (len(self.images)):
            #we will be placing the image on top of a black background of a fixed size
            #we do this because not all the images have the same dimensions and that causes problems with the trackbar
            #doing this will also make life 100x easier when we begin applying our transformations on the photos
            background = np.zeros((BACKGROUND_DIMS, BACKGROUND_DIMS, 3), np.uint8)

            #uses the value of our trackbar to get that photo and place it on screen
            img = self.images[i]

            #gets the offset so that we place our photos in the center of the background image
            x_offset = int((BACKGROUND_DIMS - img.shape[1])/2)
            y_offset = int((BACKGROUND_DIMS - img.shape[0])/2)

            if self.mode == "eyes":
                if self.eye[i][1] >= 0:
                    cv.circle(img, (int(self.landmarks[i][0][0][0]), int(self.landmarks[i][0][0][1])), 3, (0, 255, 0))
                    cv.circle(img, (self.landmarks[i][0][1][0], self.landmarks[i][0][1][1]), 3, (0, 255, 0))
                    cv.circle(img, (int(self.eye[i][3][0]), int(self.eye[i][3][1])), 3, (0, 0, 255))
                    img = self.rotate(img, self.eye[i][0], self.eye[i][1], self.eye[i][3])#rotate image at midpoint of eyes so eyes are even 
                    img, scale_val = self.scale(img, self.eye[i][2], avg_eye_dist, self.eye[i][3], avg_eye_mid)#scale image so distance between eyes are the same
                    img = self.translation(img, self.eye[i][3], avg_eye_mid, max_img_size, scale_val)#move image so eye midpoints are the same
            elif self.mode == "mouth":
                if self.mouth[i][1] >= 0:
                    cv.circle(img, (int(self.landmarks[i][0][3][0]), int(self.landmarks[i][0][3][1])), 3, (0, 255, 0))
                    cv.circle(img, (self.landmarks[i][0][4][0], self.landmarks[i][0][4][1]), 3, (0, 255, 0))
                    cv.circle(img, (int(self.mouth[i][3][0]), int(self.mouth[i][3][1])), 3, (0, 0, 255))
                    img = self.rotate(img, self.mouth[i][0], self.mouth[i][1], self.mouth[i][3])#rotate image at mouth midpoint so mouth is even 
                    img, scale_val = self.scale(img, self.mouth[i][2], avg_mouth_dist, self.mouth[i][3], avg_mouth_mid)#scale image so distance between mouth corners are the same
                    img = self.translation(img, self.mouth[i][3], avg_mouth_mid, max_img_size, scale_val)#move image so mouth midpoints are the same
            elif self.mode == "nose":
                cv.circle(img, (self.landmarks[i][0][2][0], self.landmarks[i][0][2][1]), 3, (0, 255, 0))                
                img = self.translation(img, [self.nose[i][0], self.nose[i][1]], avg_nose_loc, max_img_size, 0)#move image so nose locations are the same
            elif self.mode == "all":
                if type(self.landmarks[i]) is np.ndarray:
                    cv.circle(img, (int(self.landmarks[i][0][0][0]), int(self.landmarks[i][0][0][1])), 3, (0, 255, 0))
                    cv.circle(img, (self.landmarks[i][0][1][0], self.landmarks[i][0][1][1]), 3, (0, 255, 0))
                    cv.circle(img, (int(self.landmarks[i][0][3][0]), int(self.landmarks[i][0][3][1])), 3, (0, 255, 0))
                    cv.circle(img, (self.landmarks[i][0][4][0], self.landmarks[i][0][4][1]), 3, (0, 255, 0))
                    img = self.align(img, np.float32(self.landmarks[i][0]), avg_arr, max_img_size)#homography so point locations are the same

            background[y_offset: y_offset + img.shape[0], x_offset: x_offset + img.shape[1]] = img

            height = 600
            
            self.transformed[i] = background
        #display images
        while idx < len(self.images):

            background = self.transformed[cv.getTrackbarPos(trackbar_name, 'slideshow')]
            cv.imshow('slideshow', background)
            k = cv.waitKey(500)
            if k & 0xFF == ord('q'):
                break
            
    #rotates image
    def rotate(self, image, angle, direction, mid):
        rotated = np.copy(image)
        if direction >= 0:
            if direction == 1:
                angle = angle*-1
            rotation = cv.getRotationMatrix2D((mid[0], mid[1]), angle, 1)
            rotated = cv.warpAffine(rotated, rotation, (rotated.shape[1],rotated.shape[0]))
        return rotated

    #scales image
    def scale(self, image, dist, avg, mid_point, avg_mid):
        if(dist > 0.0):
            scale_val = avg/dist
            scaled = cv.resize(image, None, fx=scale_val, fy=scale_val, interpolation=cv.INTER_CUBIC)
            return scaled, scale_val
        else:
            return img, 0

    #translates image
    def translation(self, img, loc, avg_loc, max_size, scale_val):
        h, w = img.shape[:2]

        #translation amounts for nose
        x_shift = avg_loc[0] - loc[0] - ((max_size[0] - w)/2)
        y_shift = avg_loc[1] - loc[1] - ((max_size[1] - h)/2)
        #translation amounts for mouth or eyes
        if scale_val > 0:
            x_shift = avg_loc[0] - (loc[0]*scale_val) - ((max_size[0]*scale_val - w)/(2*scale_val))
            y_shift = avg_loc[1] - (loc[1]*scale_val) - ((max_size[1]*scale_val - h)/(2*scale_val))
        T = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        translated = cv.warpAffine(img, T, (w, h))
        return translated

    #homography to align points
    def align(self, image, lm, avg, max_size):
        h, w = image.shape[:2]
        #get difference between image size and max image size
        x_dif = ((max_size[0] - w)/2)
        y_dif = ((max_size[1] - h)/2)
        #calculate average point locations
        avg_dif = []
        for pt in avg:
            avg_dif.append([pt[0]-x_dif, pt[1]-y_dif])
        avg_dif = np.float32(avg_dif)
        homography, mask = cv.findHomography(lm, avg_dif, cv.RANSAC)
        warped = cv.warpPerspective(image, homography,(image.shape[1],image.shape[0]))
        return warped

    #get the average distance value
    def get_avg_dist(self, feature):
        total = 0.0
        count = 0
        for value in feature:
            if value[2] > 0:
                total += value[2]
                count += 1
        return total/count

    #get average location of point
    def get_avg_loc(self, lm, feature):
        avg = []
        x_tot = 0.0
        y_tot = 0.0
        count = 0
        for value in lm:
            if type(value) is np.ndarray:
                x_tot += value[0][feature][0]
                y_tot += value[0][feature][1]
                count += 1
        avg = [x_tot/count, y_tot/count]
        return avg

    #returns midpoint between 2 points
    def get_mid_point(self, left, right):
        return [(left[0]+right[0])/2, (left[1]+right[1])/2]

    #calculates average of all midpoints
    def get_avg_mid(self, lm):
        avg = []
        x_tot = 0.0
        y_tot = 0.0
        count = 0
        for val in lm:
            if type(val) is list:
                x_tot += val[3][0]
                y_tot += val[3][1]
                count += 1
        avg = [x_tot/count, y_tot/count]
        return avg

    #returns average location of every point
    def get_avg_pts(self, lm):
        lex_tot = 0.0
        rex_tot = 0.0
        nx_tot = 0.0
        lmx_tot = 0.0
        rmx_tot = 0.0
        ley_tot = 0.0
        rey_tot = 0.0
        ny_tot = 0.0
        lmy_tot = 0.0
        rmy_tot = 0.0
        count = 0
        for value in lm:
            if type(value) is np.ndarray:
                lex_tot += value[0][0][0]
                ley_tot += value[0][0][1]
                rex_tot += value[0][1][0]
                rey_tot += value[0][1][1]
                nx_tot += value[0][2][0]
                ny_tot += value[0][2][1]
                lmx_tot += value[0][3][0]
                lmy_tot += value[0][3][1]
                rmx_tot += value[0][4][0]
                rmy_tot += value[0][4][1]
                count += 1
        avg_le = [lex_tot/count, ley_tot/count]
        avg_re = [rex_tot/count, rey_tot/count]
        avg_n = [nx_tot/count, ny_tot/count]
        avg_lm = [lmx_tot/count, lmy_tot/count]
        avg_rm = [rmx_tot/count, rmy_tot/count]
        return np.float32([avg_le, avg_re, avg_n, avg_lm, avg_rm])

    #calculates the maximum size of images
    def get_max_size(self, images):
        x_max = 0
        y_max = 0
        for img in images:
            h, w = img.shape[:2]
            if h > y_max:
                y_max = h
            if w > x_max:
                x_max = w
        return [x_max, y_max]










