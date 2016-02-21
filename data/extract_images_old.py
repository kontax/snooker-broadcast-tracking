#!/usr/bin/env python2
# Extracts a selection of random images from the downloaded videos and saves them to a folder

import imageio
import random
from os import listdir
from os.path import isfile, join

random.seed(10)
images_to_extract = 50
directory_name = '../../data/short_videos/fixed/'
video_extension = '.mp4'
files = [f for f in listdir(directory_name) if isfile(join(directory_name, f))]

for video in files:
    print video

    filename = video[:5]
    full_filename = directory_name + video
    with imageio.get_reader(full_filename, 'ffmpeg') as vid:
        frame_count = vid.get_meta_data()['nframes']

        for i in range(0,images_to_extract):
            random_number = random.randint(0, frame_count)
            #print random_number
            image = vid.get_data(random_number-1)
            image_filename = filename + '-' + '{0:02d}'.format(i) + '.jpg'
            imageio.imwrite(directory_name + '../images/' + image_filename, image)

