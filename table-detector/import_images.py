import random
import numpy as np
import cv2
import sqlite3
import gzip
import cPickle
from os import listdir
from os.path import isfile, join

DIRECTORY_NAME = 'images/'
DATABASE = 'images.db'
STATEMENT = 'SELECT path, tag1 AS tag FROM images ORDER BY path;'


def load_images(directory, dtype=np.float32):
    # Load images from the directory
    images = [cv2.imread(directory + f, 0)
              for f in listdir(directory)
              if isfile(join(directory, f))]

    if dtype == np.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

    return images


def serialize_array(filename, images, tags):
    data = (images, tags)
    data_file = gzip.open(filename, 'wb')
    cPickle.dump(data, data_file)

    data_file.flush()
    data_file.close()


def deserialize_array(filename):
    f = gzip.open(filename, 'rb')
    return cPickle.load(f)


def get_tags(database, statement):
    conn = sqlite3.connect(database)
    curs = conn.cursor()
    curs.execute(statement)
    tagged_files = curs.fetchall()
    conn.close()
    return tagged_files
