import sqlite3
import tensorflow as tf
from os import listdir
from os.path import basename
from os.path import splitext


BATCH_SIZE = 20
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 549
MAX_STEPS = 20000


def get_tagged_image(filename_and_label):
    filename, label = tf.decode_csv(filename_and_label, [[""], [""]], " ")
    image = tf.read_file(filename)

    converted = tf.image.decode_jpeg(image)
    tag = tf.string_to_number(label, tf.int32)

    return converted, tag


def get_image_list(image_path, tagged_database):
    image_file_list = [image_path + f for f in sorted(listdir(image_path))]
    filename_and_label = []

    conn = sqlite3.connect(tagged_database)
    curs = conn.cursor()

    for image_file in image_file_list:
        path = basename(image_file)
        statement = "SELECT CASE WHEN tag1 = 'yes' THEN 1 ELSE 0 END AS tag FROM images WHERE path = '" + path + "';"
        curs.execute(statement)
        tag = curs.fetchall()
        full_name = image_file + " " + str(tag[0][0])
        filename_and_label.append(full_name)

    return filename_and_label


def load_data():
    class SnookerImage(object):
        pass

    result = SnookerImage()

    result.width = 150
    result.height = 85
    result.depth = 3

    image_list = get_image_list("images/", "images.db")
    image_queue = tf.train.string_input_producer(image_list)

    image, result.label = get_tagged_image(image_queue.dequeue())
    depth_major = tf.reshape(image, [result.depth, result.height, result.width])
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    result.label = tf.cast(result.label, tf.int32)

    return result


def get_shuffled_images():
    read_input = load_data()
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = 85
    width = 150
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 0.4)

    float_image = tf.image.per_image_whitening(reshaped_image)
    images, label_batch = tf.train.shuffle_batch(
        [float_image, read_input.label],
        batch_size=BATCH_SIZE,
        num_threads=16,
        capacity=min_queue_examples + 3 * BATCH_SIZE,
        min_after_dequeue=min_queue_examples)

    tf.image_summary('image', images)
    return images, tf.reshape(label_batch, [BATCH_SIZE])
