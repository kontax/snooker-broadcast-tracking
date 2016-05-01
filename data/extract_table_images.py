#!/usr/bin/env python2
# Extracts a selection of random images from the downloaded videos and saves them to a folder.
# This version uses the pre-trained caffe model in order to only get images that are relevant,
# ie. those that are of the specific view of the table we want to view.

import cv2
import random
import sys
import os
from os import listdir
from os.path import isfile, join, splitext

caffe_root = '/home/james/dev/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

random.seed(10)
images_to_extract = 50  # The number of images to extract from each video

model_def = os.path.join(caffe_root, 'models/snooker_table/deploy.prototxt')
model_weights = os.path.join(caffe_root,
                             'models/snooker_table/initial_model.caffemodel')

directory_name = '../../data/videos/'
video_extension = '.mp4'
files = [f for f in listdir(directory_name) if isfile(join(directory_name, f))]


def init_caffe_model():
    caffe.set_mode_cpu()

    net = caffe.Net(model_def, model_weights, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # move image channels
    transformer.set_raw_scale('data', 255)  # rescale pixel numbers
    transformer.set_channel_swap('data', (2, 1, 0))  # swap from RGB to BGR

    # Reshape image to batch size of 1, 3 channels, H: 85, W: 150
    net.blobs['data'].reshape(1, 3, 85, 150)

    return net, transformer


def check_output(frame, net, transformer):
    transformed_frame = transformer.preprocess('data', frame)
    net.blobs['data'].data[...] = transformed_frame
    output = net.forward()
    output_prob = output['score'][0]

    return output_prob.argmax() == 1


def video_has_been_done(video):
    video_name = splitext(video)[0]
    image_dir = os.path.join(directory_name, 'images')
    for i in os.listdir(image_dir):
        if os.path.isfile(os.path.join(image_dir, i)) and video_name in i:
            return True

    return False


def extract_images(video_files):
    net, transformer = init_caffe_model()

    for video in video_files:
        print video

        # Check to see if the video has already been done
        if video_has_been_done(video):
            print video + ' has been done already'
            continue

        # Get the auto-number created from youtube-dl for reference
        filename = video[:5]
        full_filename = directory_name + video

        # Load the video into the script and get the number of frames in it
        cap = cv2.VideoCapture(full_filename)
        frame_count = cap.get(7)

        i = 0
        while i < images_to_extract:

            # Get a random frame from the video and save it
            random_number = random.randint(0, frame_count - 1)
            cap.set(1, random_number)
            ret, image = cap.read()

            if image is None:
                continue

            if check_output(image, net, transformer):
                print filename + ': ' + str(i) + ' is valid'
                image_filename = filename + '-' + '{0:02d}'.format(i) + '.jpg'
                full_image_path = os.path.join(directory_name, 'images',
                                               image_filename)
                cv2.imwrite(full_image_path, image)
                i += 1

            else:
                print filename + ': ' + str(i) + ' is NOT valid'

        # Make sure to close the capture class or all sorts of problems will happen
        cap.release()


if __name__ == '__main__':
    extract_images(files)
