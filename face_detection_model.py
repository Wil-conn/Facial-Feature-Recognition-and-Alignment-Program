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
            #print('land')
            #print(landmarks)

            # not entirely sure why idx is a tuple in the form (index , ).
            # using idx[0] fixes this by just getting the index number
            if type(landmarks) is np.ndarray:
                self.landmarks.append(landmarks)


            # if we are not able to detect the features for a photo we update
            # out images array to remove that photo and we continue
            # to the next iteration
            if boxes is None:
                print("IMAGE NUMBER " + str(idx) + " SHOULD BE DELETED")
                self.images = np.delete(self.images, idx)
                idx -= 1
                #self.eye.append([0, -1, 0])
                #self.mouth.append([0, -1, 0])
                #self.nose.append([-1, -1])
                #pass
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

                eye_tilt_angle, eye_tilt_dir, eye_distance = self.get_angle_of_tilt((landmarks[0][0][0], landmarks[0][0][1]),
                                                        (landmarks[0][1][0], landmarks[0][1][1]))

                mouth_tilt_angle, mouth_tilt_dir, mouth_distance = self.get_angle_of_tilt((landmarks[0][3][0], landmarks[0][3][1]),
                                                        (landmarks[0][4][0], landmarks[0][4][1]))
                #get midpoints of eyes and mouth
                eye_mid = self.get_mid_point(landmarks[0][0], landmarks[0][1])
                mouth_mid = self.get_mid_point(landmarks[0][3], landmarks[0][4])
                #print('eye mid')
                #print(eye_mid)
                cv.circle(image, (int(eye_mid[0]), int(eye_mid[1])), 3, (0, 0, 255))
                cv.circle(image, (int(mouth_mid[0]), int(mouth_mid[1])), 3, (0, 0, 255))
                self.eye.append([eye_tilt_angle, eye_tilt_dir, eye_distance, eye_mid])
                self.mouth.append([mouth_tilt_angle, mouth_tilt_dir, mouth_distance, mouth_mid])
                self.nose.append([landmarks[0][2][0], landmarks[0][2][1]])

                cv.putText(image, str(landmarks[0][0][0]), (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)
                cv.putText(image, str(eye_tilt_angle), (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)
                cv.putText(image, str(mouth_tilt_angle), (10, 300), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)
            idx += 1

        #print(self.landmarks[0])
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
        # if right eye's y is higher than the left's then we say it has an 'upwards' tilt, return 1
        if left_coord[1] > right_coord[1]:
            #print("left eye y: " + str(left_coord[1]))
            #print("right eye y: " + str(right_coord[1]))
            #print("eyes have an 'upwards' angler")
            angle = math.acos(adj/hype)
            return (math.degrees(angle), 1, hype)

        # if right eye's y is lower than left eye's y, it has a 'downwards' tilt, return 0
        elif left_coord[1] < right_coord[1]:
            #print("left eye y: " + str(left_coord[1]))
            #print("right eye y: " + str(right_coord[1]))
            #print("eyes have an 'downwards' angler")
            angle = math.acos(adj/hype)
            return (math.degrees(angle), 0, hype)

    def display(self):
        idx = 0
        cv.namedWindow('slideshow')
        trackbar_name = 'image # %d' % idx
        cv.createTrackbar(trackbar_name, 'slideshow', 0, len(self.images)-1, nothing)

        #calculate the average distance between eyes/mouth for scaling
        avg_eye_dist = self.get_avg_dist(self.eye)
        avg_mouth_dist = self.get_avg_dist(self.mouth)
        avg_eye_mid = self.get_avg_mid(self.eye)#left eye
        avg_nose_loc = self.get_avg_loc(self.landmarks, 2)#nose
        avg_mouth_mid = self.get_avg_mid(self.mouth)#left mouth
        max_img_size = self.get_max_size(self.images, self.eye, self.mouth, avg_eye_dist, avg_mouth_dist)

        avg_arr = self.get_avg_pts(self.landmarks)
        #print(avg_arr)
        while idx < len(self.images):
            #print(self.landmarks[cv.getTrackbarPos(trackbar_name, 'slideshow')])
            #we will be placing the image on top of a black background of a fixed size
            #we do this because not all the images have the same dimensions and that causes problems with the trackbar
            #doing this will also make life 100x easier when we begin applying our transformations on the photos
            background = np.zeros((BACKGROUND_DIMS, BACKGROUND_DIMS, 3), np.uint8)

            #uses the value of our trackbar to get that photo and place it on screen
            img = self.images[cv.getTrackbarPos(trackbar_name, 'slideshow')]

            #gets the offset so that we place our photos in the center of the background image
            x_offset = int((BACKGROUND_DIMS - img.shape[1])/2)
            y_offset = int((BACKGROUND_DIMS - img.shape[0])/2)
            cv.circle(img, (int(avg_eye_mid[0]), int(avg_eye_mid[1])), 3, (255, 0, 0))
            cv.circle(img, (int(avg_mouth_mid[0]), int(avg_mouth_mid[1])), 3, (255, 0, 0))
            #print("MODE " + str(self.mode))
            scale_val = [1, 1]
            if self.mode == "eyes":
                if self.eye[cv.getTrackbarPos(trackbar_name, 'slideshow')][1] >= 0:
                    
                    img, scale_val = self.scale(img, self.eye[cv.getTrackbarPos(trackbar_name, 'slideshow')][2], avg_eye_dist, self.eye[cv.getTrackbarPos(trackbar_name, 'slideshow')][3], avg_eye_mid, max_img_size[0])
                    img = self.translation(img, self.eye[cv.getTrackbarPos(trackbar_name, 'slideshow')][3], avg_eye_mid, max_img_size[1], scale_val)
                    img = self.rotate(img, self.eye[cv.getTrackbarPos(trackbar_name, 'slideshow')][0], self.eye[cv.getTrackbarPos(trackbar_name, 'slideshow')][1], self.eye[cv.getTrackbarPos(trackbar_name, 'slideshow')][3])
                else:
                    cv.putText(img, 'no angle key', (10, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)
                cv.putText(img, 'Eye Aligned', (10, 450), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)

            elif self.mode == "mouth":
                if self.mouth[cv.getTrackbarPos(trackbar_name, 'slideshow')][1] >= 0:
                    
                    img, scale_val = self.scale(img, self.mouth[cv.getTrackbarPos(trackbar_name, 'slideshow')][2], avg_mouth_dist, self.mouth[cv.getTrackbarPos(trackbar_name, 'slideshow')][3], avg_mouth_mid, max_img_size[0])
                    img = self.translation(img, self.mouth[cv.getTrackbarPos(trackbar_name, 'slideshow')][3], avg_mouth_mid, max_img_size[2], scale_val)
                    img = self.rotate(img, self.mouth[cv.getTrackbarPos(trackbar_name, 'slideshow')][0], self.mouth[cv.getTrackbarPos(trackbar_name, 'slideshow')][1], self.mouth[cv.getTrackbarPos(trackbar_name, 'slideshow')][3])

                else:
                    cv.putText(img, 'no angle key', (10, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)
                cv.putText(img, 'Mouth Aligned', (10, 450), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)

            elif self.mode == "nose":
                img = self.translation(img, [self.nose[cv.getTrackbarPos(trackbar_name, 'slideshow')][0], self.nose[cv.getTrackbarPos(trackbar_name, 'slideshow')][1]], avg_nose_loc, max_img_size[0], [0, 0])

            elif self.mode == "all":
                if type(self.landmarks[cv.getTrackbarPos(trackbar_name, 'slideshow')]) is np.ndarray:
                    img = self.align(img, np.float32(self.landmarks[cv.getTrackbarPos(trackbar_name, 'slideshow')][0]), avg_arr, max_img_size[0])
                else:
                    pass
                    #print(cv.getTrackbarPos(trackbar_name, 'slideshow'))
                cv.putText(img, 'All Aligned', (10, 450), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)

            else:
                cv.putText(img, 'Non Aligned', (10, 450), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)

            background[y_offset: y_offset + img.shape[0], x_offset: x_offset + img.shape[1]] = img
            #rotates image so eyes/mouth are level
            #if the image didnt pass the box check in detect then it will not have rotation angle information and will result in error.
            #trackbar used as bool, 1=on, if both on, nothing done

            height = 600
            #dim = (int(height/self.images[idx].shape[0] * self.images[idx].shape[1]), height)
            #img = cv.resize(self.images[idx], dim)
            #cv.imshow('image', self.images[idx])
            cv.imshow('slideshow', background)
            k = cv.waitKey(500)

    def rotate(self, image, angle, direction, mid):
        rotated = np.copy(image)
        #print(mid)
        #print((rotated.shape[1]/2,rotated.shape[0]/2))
        if direction >= 0:
            if direction == 1:
                angle = angle*-1
            rotation = cv.getRotationMatrix2D((mid[0], mid[1]), angle, 1)
            rotated = cv.warpAffine(rotated, rotation, (rotated.shape[1],rotated.shape[0]))
        return rotated

    #just returns the original image for now
    def scale(self, image, dist, avg, mid_point, avg_mid, max_size):
        if(dist > 0.0):
            scale_val = avg/dist
            
            print('scale')
            '''print(image.shape[1])
            print(image.shape[0])
            print(avg)
            print(scale_val)
            print(dist)
            print(dist*scale_val)'''
            
            #if len(mid_point) > 0:
            h, w = image.shape[:2]
            
            scaled = cv.resize(image, None, fx=scale_val, fy=scale_val, interpolation=cv.INTER_CUBIC)
            h2, w2 = scaled.shape[:2]
            print(h)
            print(h2)
            print(h*scale_val)
            #return np.float32(sc), quadrants
            return scaled, [w2-w, h2-h]
        else:
            return img, 1
    
    def translation(self, img, loc, avg_loc, max_size, scale_dif):
        h, w = img.shape[:2]
        print('tran')
        print(h)
        print(w)
        print(loc)
        print(avg_loc)
        print(scale_dif)
        print(max_size)
        x_shift = avg_loc[0] - loc[0] - ((max_size[0] - w)/2) - scale_dif[0]
        y_shift = avg_loc[1] - loc[1] - ((max_size[1] - h)/2) - scale_dif[1]
        #cv.circle(img, (int(loc[0]+x_shift), int(loc[1]+y_shift)), 3, (255, 0, 0))
        T = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        translated = cv.warpAffine(img, T, (w, h))
        
        #print(x_shift)
        #print(y_shift)
        
        return translated

    def align(self, image, lm, avg, max_size):
        h, w = image.shape[:2]
        x_dif = ((max_size[0] - w)/2)
        y_dif = ((max_size[1] - h)/2)
        print('here')
        print(lm)
        print(avg)
        print(x_dif)
        print(y_dif)
        avg_dif = []
        for pt in avg:
            print(pt)
            avg_dif.append([pt[0]-x_dif, pt[1]-y_dif])
        print(avg_dif)   
        avg_dif = np.float32(avg_dif)

        homography, mask = cv.findHomography(lm, avg_dif, cv.RANSAC)
        #print(homography)
        #warp the image
        warped = cv.warpPerspective(image, homography,(image.shape[1],image.shape[0]))


        return warped

    def get_avg_dist(self, feature):
        total = 0.0
        count = 0
        for value in feature:
            if value[2] > 0:
                total += value[2]
                count += 1
        return total/count

    def get_avg_loc(self, lm, feature):
        #print(self.landmarks)
        avg = []
        x_tot = 0.0
        y_tot = 0.0
        count = 0
        for value in lm:
            print('val')
            print(value)
            if type(value) is np.ndarray:
                x_tot += value[0][feature][0]
                y_tot += value[0][feature][1]
                count += 1
        avg = [x_tot/count, y_tot/count]
        return avg
    
    def get_mid_point(self, left, right):
        return [(left[0]+right[0])/2, (left[1]+right[1])/2]
    
    def get_avg_mid(self, lm):
        
        #print('midpoints')
        #print(lm)
        avg = []
        x_tot = 0.0
        y_tot = 0.0
        count = 0
        for val in lm:
            print(type(val))
            if type(val) is list:
                x_tot += val[3][0]
                y_tot += val[3][1]
                count += 1
                print(count)
        avg = [x_tot/count, y_tot/count]
        print(avg)
        return avg
    
    def get_avg_pts(self, lm):
        #print('len')
        #print(len(self.landmarks))
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

    def get_max_size(self, images, eyes, mouths, edst, mdst):
        x_max = 0
        y_max = 0
        ex_max = 0
        ey_max = 0
        mx_max = 0
        my_max = 0
        count = 0
        for img in images:
            print(img)
            h, w = img.shape[:2]
            if h > y_max:
                y_max = h
            if w > x_max:
                x_max = w
            e_val = edst/self.eye[count][2]
            m_val = mdst/self.mouth[count][2]
            if h*e_val > ey_max:
                ey_max = h*e_val
            if w*e_val > ex_max:
                ex_max = w*e_val
            if h*m_val > my_max:
                my_max = h*m_val
            if w*m_val > mx_max:
                mx_max = w*m_val
            count += 1
        return [[x_max, y_max], [ex_max, ey_max], [mx_max, my_max]]
















