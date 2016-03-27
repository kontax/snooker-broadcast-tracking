#!/usr/bin/python2

import extract_data
import numpy as np
import shutil
import sys

from os.path import expanduser
from os.path import join
from os.path import isdir
from pylab import *
from subprocess import call

caffe_root = join(expanduser('~'), 'dev/caffe')  # Root directory of caffe install
sys.path.insert(0, caffe_root + 'python')  # Directory of pycaffe install
import caffe

from caffe import layers as L, params as P

EXTRACT_DIR = extract_data.EXTRACT_DIR
TRAIN_FILE = extract_data.TRAIN_FILE
TEST_FILE = extract_data.TEST_FILE

TRAIN_LMDB = 'train_lmdb'  # The LMDB directory containing train data
TEST_LMDB = 'test_lmdb'  # The LMDB directory containing test data
TRAIN_PROTOTXT = 'train.prototxt'  # The prototxt file with the train layers
TEST_PROTOTXT = 'test.prototxt'  # The prototxt file with the test layers
TRAIN_BATCH_SIZE = 25  # Batch size to use for training
TEST_BATCH_SIZE = 100  # Batch size to use for testing
SOLVER_PROTOTXT = 'solver.prototxt'  # The prototxt file containing model parameters

METHOD = 'CPU'  # Whether to use the CPU or GPU to run the model
MODEL_OUTPUT = 'output.caffemodel'
ACCURACY_CHART = 'accuracy.png'


def create_lmdb(labels, image_dir, lmdb_dir):
    """
    Creates an LMDB directory which the caffe model uses to train
    :param labels: The file containing the image names and labels
    :param image_dir: The directory containing the images
    :param lmdb_dir: The directory to save the LMDB files to
    :return:
    """
    # Remove the lmdb folder if it exists already, as otherwise we get an error
    if isdir(lmdb_dir):
        shutil.rmtree(lmdb_dir)

    # Use the convert_imageset executable to create an LMDB for use in caffe
    convert = join(caffe_root, 'build/tools/convert_imageset')

    # The full command to execute
    command = "{0} --shuffle {1} {2} {3}".format(convert, image_dir,
                                                 labels, lmdb_dir)
    call(command.split())


def snooker_net(lmdb, batch_size, output_type="Train"):
    """
    Creates a prototxt file containing a description of the layers within the
    network.
    :param lmdb: The LMDB file containing the image and label data
    :param batch_size: The batch size to load data into the network
    :param output_type: Whether the prototxt file is for Training or Deployments
    :return: A string in prototxt format describing the network layers
    """
    n = caffe.NetSpec()

    if output_type == "Train":
        n.data, n.label = L.Data(
            batch_size=batch_size,  # The number to train at a time
            backend=P.Data.LMDB,  # The backend database used
            source=lmdb,  # The image source
            transform_param=dict(scale=1. / 255),  # Converting image pixels to 0-1
            ntop=2  # The number of classes output
        )

    # First convolutional layer
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20,
                            weight_filler=dict(type='xavier'))

    # First max pooling layer
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    # Second convolutional layer
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50,
                            weight_filler=dict(type='xavier'))

    # Second pooling layer
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    # Fully connected layer
    n.fc1 = L.InnerProduct(n.pool2, num_output=500,
                           weight_filler=dict(type='xavier'))

    # FC1 uses a Rectified Linear activation function
    n.relu1 = L.ReLU(n.fc1, in_place=True)

    # The output layer
    n.score = L.InnerProduct(n.relu1, num_output=2,
                             weight_filler=dict(type='xavier'))

    # The loss function is a softmax
    if output_type == "Train":
        n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()


def create_solver(train_net, test_net):
    """
    Creates a solver prototxt file, which outlines the parameters a network should
    follow when training and testing.
    :param train_net: The location of the training prototxt model
    :param test_net: The location of the testing prototxt model
    :return: A string containing colon-separated values on new lines to save
    """
    solver = {
        # The location of the train & test prototxt files
        "train_net": '"' + train_net + '"',
        "test_net": '"' + test_net + '"',

        # The number of forward passes the test carries out
        "test_iter": 100,
        # The interval whereby a test should be performed
        "test_interval": 25,

        # The interval by which the model should display values
        "display": 100,
        # The maximum number of iterations to perform
        "max_iter": 500,

        # The learning rate decay policy
        "lr_policy": '"inv"',
        "base_lr": 0.01,
        "gamma": 0.0001,
        "power": 0.75,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        # Using stochastic gradient descent
        "type": '"SGD"',

        # Whether to use the CPU or GPU
        "solver_mode": METHOD,
        "random_seed": extract_data.SEED,

        # The iteration to take a snapshot on, and the folder to save them
        "snapshot": 100,
        "snapshot_prefix": '"snapshots"'
    }

    # Return a colon-separated string with newlines to save
    return "\n".join(["{0}: {1}".format(k, v) for k, v in solver.iteritems()]) + "\n"


def train(solver_prototxt, method):
    """
    Trains the neural network with the specified solver.
    :param solver_prototxt: The prototxt file containing the network parameters
    :param method: Whether to use the CPU or GPU for training
    :return: A TrainingResults object containing the following:
     * niter: number of iterations
     * test_interval: interval which tests were performed
     * test_acc: The accuracy of the test results
     * train_loss: The loss calculated
     * net: The trained network of weights
    """

    class TrainingResults(object):
        pass

    if method == "GPU":
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    solver = None
    solver = caffe.SGDSolver(solver_prototxt)

    TrainingResults.niter = 500
    TrainingResults.test_interval = 25
    # losses will also be stored in the log
    TrainingResults.train_loss = zeros(TrainingResults.niter)
    TrainingResults.test_acc = zeros(
        int(np.ceil(TrainingResults.niter / TrainingResults.test_interval)))
    output = zeros((TrainingResults.niter, 8, 2))

    # the main solver loop
    for it in range(TrainingResults.niter):
        solver.step(1)  # SGD by Caffe

        # store the train loss
        TrainingResults.train_loss[it] = solver.net.blobs['loss'].data

        # store the output on the first test batch
        # (start the forward pass at conv1 to avoid loading new data)
        solver.test_nets[0].forward(start='conv1')
        output[it] = solver.test_nets[0].blobs['score'].data[:8]

        # run a full test every so often
        if it % TrainingResults.test_interval == 0:
            print 'Iteration', it, 'testing...'
            correct = 0
            for test_it in range(100):
                solver.test_nets[0].forward()
                correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1) ==
                               solver.test_nets[0].blobs['label'].data)
            TrainingResults.test_acc[
                it // TrainingResults.test_interval] = correct / 1e4

    TrainingResults.net = solver.net
    return TrainingResults


def save_accuracy(results, png_location):
    _, ax1 = subplots()
    ax2 = ax1.twinx()
    ax1.plot(arange(results.niter), results.train_loss)
    ax2.plot(results.test_interval * arange(len(results.test_acc)),
             results.test_acc,
             'r')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('test accuracy')
    ax2.set_title('Test Accuracy: {:.2f}'.format(results.test_acc[-1]))


def train_model():
    extract_data.extract_data()

    print "Saving a train LMDB into the {0} directory".format(TRAIN_LMDB)
    create_lmdb(TRAIN_FILE, EXTRACT_DIR, TRAIN_LMDB)

    print "Saving a test LMDB into the {0} directory".format(TEST_LMDB)
    create_lmdb(TEST_FILE, EXTRACT_DIR, TEST_LMDB)

    print "Creating a train prototxt file with batch size {0} into {1}".format(
        TRAIN_BATCH_SIZE, TRAIN_LMDB)
    with open(TRAIN_PROTOTXT, 'w') as f:
        f.write(str(snooker_net(TRAIN_LMDB, TRAIN_BATCH_SIZE)))

    print "Creating a test prototxt file with batch size {0} into {1}".format(
        TEST_BATCH_SIZE, TEST_LMDB)
    with open(TEST_PROTOTXT, 'w') as f:
        f.write(str(snooker_net(TEST_LMDB, TEST_BATCH_SIZE)))

    print "Creating a solver prototxt into the {0} file".format(SOLVER_PROTOTXT)
    with open(SOLVER_PROTOTXT, 'w') as f:
        f.write(str(create_solver(TRAIN_PROTOTXT, TEST_PROTOTXT)))

    print "Training the model"
    results = train(SOLVER_PROTOTXT, METHOD)

    print "Saving the trained model to {0}".format(MODEL_OUTPUT)
    results.net.save(MODEL_OUTPUT)

    print "Saving accuracy chart to {0}".format(ACCURACY_CHART)
    save_accuracy(results, ACCURACY_CHART)


    # Test model accuracy & save graph


if __name__ == '__main__':
    train_model()
