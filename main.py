import torch
import cv2 as cv
import numpy as np
import os
import sys
import tkinter
from face_detection_model import *
from user_interface import *

def get_photos(path):
    path, dirs, files = next(os.walk(path))
    file_count = len(files)

    images = np.empty(file_count, dtype = object)
    c = 0
    for entry in os.scandir(path):
        if entry.path.endswith(".jpg") and entry.is_file():
            print(str(entry.path) + " is picture number " + str(c) +
                                    " in the images array")
            images[c] = cv.imread(str(entry.path))
            c += 1
    return images

def main():
    #print(sys.argv[1])
    #img = cv.imread('images/face1.jpg')
    root = tkinter.Tk()
    ui = user_interface(master=root)
    ui.mainloop()
    root.destroy()
    fd = face_detect(0.7, get_photos(ui.dir_location), ui.selection)
    #fd = face_detect(0.7, arr, 'eyes')
    fd.detect()
    fd.display()
    video = cv.VideoWriter("output.avi", cv.VideoWriter_fourcc(*'XVID'), 18, (600,600))
    for image in fd.transformed:
        background = np.zeros((600, 600, 3), np.uint8)
        x_offset = int((BACKGROUND_DIMS - image.shape[1])/2)
        y_offset = int((BACKGROUND_DIMS - image.shape[0])/2)
        background[y_offset: y_offset + image.shape[0], x_offset: x_offset + image.shape[1]] = image
        video.write(background)
    '''
    for image in fd.images:
        cv.imshow('test', image)
        k = cv.waitKey(500)
    '''
    #faces = detect(img, mtcnn)
    #cv.imshow('image', faces)
    #print(fd.landmarks)


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
