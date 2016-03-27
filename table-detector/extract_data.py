#!/usr/bin/python2

import os
import random
import shutil
import sqlite3
import tarfile

from os import listdir
from os.path import basename


DATABASE_FILE = 'images.db'             # SQLite3 file containing the labels
IMAGE_GZIP = 'images.tar.gz'            # Compressed file containing the images
IMAGE_DIR = '/tmp/snooker/'             # Temporary directory to extract images to
EXTRACT_DIR = os.path.join(IMAGE_DIR, 'images/')

SEED = 1000                             # The seed to use to replicate results
TRAIN_COUNT = 2000                      # Number of images to use for training
TEST_COUNT = 549                        # Number of images to use for testing

TRAIN_FILE = './train.txt'
TEST_FILE = './test.txt'


def extract_images(gzipped_file, target_directory):
    """
    Extracts the specified gzipped file to a directory
    :param gzipped_file: The gzipped file containing the images
    :param target_directory: The directory to extract the images to
    """
    if os.path.isdir(target_directory):
        shutil.rmtree(target_directory)
    os.makedirs(target_directory)

    with tarfile.open(gzipped_file, "r:gz") as tar:
        tar.extractall(path=target_directory)


def get_labels(image_path, label_database):
    """
    Returns a list containing the path of the image alongside a label contained
    within the SQLite3 database given.
    :param image_path: The full path of the directory containing the images
    :param label_database: The SQLite3 database containing the lables
    :return: A list of strings containing the path and label of the image
    """
    image_file_list = [os.path.join(image_path, f)
                       for f in sorted(listdir(image_path))]
    filename_and_label = []

    conn = sqlite3.connect(label_database)
    curs = conn.cursor()

    for image_file in image_file_list:
        path = basename(image_file)
        statement = "SELECT CASE WHEN tag1 = 'yes' THEN 1 ELSE 0 END AS tag" \
                    " FROM images WHERE path = '{p}';".format(p=path)
        curs.execute(statement)
        tag = curs.fetchall()
        full_name = path + " " + str(tag[0][0])
        filename_and_label.append(full_name)

    return filename_and_label


def split_input(image_list, train_count, test_count):
    """
    Splits the list of labelled images into TRAIN and TEST batches.
    :param image_list: The list of strings containing an image filename and label
    :param train_count: The number of images to use for training
    :param test_count: The number of images to use for testing
    :return: Two lists containing test and train batches
    """

    # If the number of items in the list of images doesn't match the summation
    # of the train/test counts then something is not right
    if len(image_list) != train_count + test_count:
        raise ValueError("The count of images doesn't match the train/test count.",
                         train_count, test_count, len(image_list))

    # Shuffle the list first then split it into two
    random.shuffle(image_list)
    train_list = image_list[:train_count]
    test_list = image_list[-test_count:]
    return train_list, test_list


def save_list_as_text(list_to_save, location):
    """
    Saves a specified list as a text file
    :param list_to_save: The list to save
    :param location: The full file location to save to
    """
    with open(location, 'w') as f:
        f.write("\n".join(map(lambda x: str(x), list_to_save)) + "\n")


def extract_data():
    print "Extracting {im} to {dir}".format(im=IMAGE_GZIP, dir=IMAGE_DIR)
    extract_images(IMAGE_GZIP, IMAGE_DIR)

    print "Getting labels from {db}".format(db=DATABASE_FILE)
    image_list = get_labels(EXTRACT_DIR, DATABASE_FILE)

    print "Splitting out train and test files, saving to {tr} and {te}".format(
        tr=TRAIN_FILE, te=TEST_FILE
    )
    train, test = split_input(image_list, TRAIN_COUNT, TEST_COUNT)
    save_list_as_text(train, TRAIN_FILE)
    save_list_as_text(test, TEST_FILE)


if __name__ == '__main__':
    random.seed(SEED)
    extract_data()

