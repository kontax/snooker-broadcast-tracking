#!/usr/bin/env python2
# Extracts a selection of random images from the downloaded videos and saves them to a folder

import cv2
import random
from os import listdir
from os.path import isfile, join

random.seed(10)
images_to_extract = 50 # The number of images to extract from each video
directory_name = '../../data/short_videos/fixed/'
video_extension = '.mp4'
files = [f for f in listdir(directory_name) if isfile(join(directory_name, f))]

for video in files:
    print video

    # Get the autonumber created from youtube-dl for reference
    filename = video[:5]
    full_filename = directory_name + video

    # Load the video into the script and get the number of frames in it
    cap = cv2.VideoCapture(full_filename)
    frame_count = cap.get(7)

    for i in range(0, images_to_extract):

        # Get a random frame from the video and save it
        random_number = random.randint(0, frame_count-1)
        cap.set(1, random_number)
        ret, image = cap.read()
        image_filename = filename + '-' + '{0:02d}'.format(i) + '.jpg'
        cv2.imwrite(directory_name + '../images/' + image_filename, image)

    # Make sure to close the capture class or all sorts of problems will happen
    cap.release()
