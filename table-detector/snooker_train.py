import tensorflow as tf
import numpy as np
import os
import time
from datetime import datetime

import snooker_import
import snooker_functions

TRAIN_DIR = 'train/'


with tf.Graph().as_default():

    global_step = tf.Variable(0, trainable=False)

    images, labels = snooker_import.get_shuffled_images()

    logits = snooker_functions.inference(images)
    loss = snooker_functions.loss(logits, labels)
    train_op = snooker_functions.train(loss, global_step)
    saver = tf.train.Saver(tf.all_variables())
    summary_op = tf.merge_all_summaries()
    init = tf.initialize_all_variables()

    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=False))
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)
    summary_writer = tf.train.SummaryWriter(TRAIN_DIR, graph_def=sess.graph_def)

    for step in xrange(snooker_import.MAX_STEPS):
        start_time = time.time()
        _, loss_value = sess.run([train_op, loss])
        duration = time.time() - start_time

        assert not np.isnan(loss_value)

        if step % 10 == 0:
            num_examples_per_step = snooker_import.BATCH_SIZE
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print (format_str % (datetime.now(), step, loss_value,
                                 examples_per_sec, sec_per_batch))

        if step % 100 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)

        if step % 1000 == 0 or (step + 1) == snooker_import.MAX_STEPS:
            checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
