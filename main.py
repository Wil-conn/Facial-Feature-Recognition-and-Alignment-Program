import torch
import cv2 as cv
import numpy as np
import os
import sys
from face_detection_model import *

def get_photos(path):
    path, dirs, files = next(os.walk(path))
    file_count = len(files)

    images = np.empty(file_count, dtype = object)
    c = 0
    for entry in os.scandir(path):
        if entry.path.endswith(".jpg") and entry.is_file():
            print(str(entry.path) + " is picture number " + str(c) + " in the images array")
            images[c] = cv.imread(str(entry.path))
            c += 1
    return images

def main():
    print(sys.argv[1])
    #img = cv.imread('images/face1.jpg')
    fd = face_detect(0.7, get_photos(sys.argv[1]), 'eyes')
    #fd = face_detect(0.7, arr, 'eyes')
    fd.detect()
    fd.display()

    #faces = detect(img, mtcnn)
    #cv.imshow('image', faces)
    #print(fd.landmarks)


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
