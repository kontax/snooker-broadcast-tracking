#!/usr/bin/env python

import numpy as np
import cv2
import sys

caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')
import caffe

video = '00248.mp4'
fps = 30

model_def = '/home/james/dev/caffe/models/snooker_table/deploy.prototxt'
model_weights = '/home/james/dev/caffe/models/snooker_table/initial_model.caffemodel'

def overlay_text(frame):
    #frame = cv2.rectangle(frame,(10,10),(30,30),(0,0,255),3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'TABLE!!!!', (10,100), font, 2, (0,0,255), 2)


def init_caffe_model():
    caffe.set_mode_cpu()

    net = caffe.Net(model_def, model_weights, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))      # move image channels
    transformer.set_raw_scale('data', 255)          # rescale pixel numbers
    transformer.set_channel_swap('data', (2,1,0))   # swap from RGB to BGR

    # Reshape image to batch of 50, 3 channels, H: 85, W: 150
    net.blobs['data'].reshape(1, 3, 85, 150)

    return net, transformer


def check_output(frame, net, transformer):
    transformed_frame = transformer.preprocess('data', frame)
    net.blobs['data'].data[...] = transformed_frame
    output = net.forward()
    output_prob = output['score'][0]

    return output_prob.argmax() == 1


def play_video(filename, fps, net, transformer):

    # Set the wait time for the video depending on the chosen FPS
    wait = int((100.0 / fps) * 10)
    cap = cv2.VideoCapture(filename)

    while(cap.isOpened()):
        ret, frame = cap.read()

        if check_output(frame, net, transformer):
            overlay_text(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    net, transformer = init_caffe_model()
    play_video(video, fps, net, transformer)
