"""
Baseline for machine learning project on road segmentation.
This simple baseline consits of a CNN with two convolutional+pooling layers with a soft-max loss

Credits: Aurelien Lucchi, ETH Zürich
"""



import gzip
import os
import sys
import urllib
import scipy.misc
import matplotlib.image as mpimg
from PIL import Image
from scipy import ndimage as ndi
import math
import code
import tensorflow.python.platform
import numpy
import random
import tensorflow as tf

NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SET = range(31, 126)  # the first couple of images are left for validation
TEST_SIZE = 50  # ideally: 50
SEED = 464972  # Set to None for random seed.
BATCH_SIZE = 16  # 64 (?)
NUM_EPOCHS = 1  # actually 1 epoch takes 4 days, so we just kill the training process after some time...
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000
SAVING_MODEL_TO_DISK_STEP = 10000
BATCH_SIZE_FOR_PREDICTION = 32
PADDING_COLOR = 0.0  # 0.0 = black, 0.5 = gray
SPECIAL_PADDING_COLOR_FOR_GROUNDTRUTH = 0.54  # pixels which have this colour in the groundtruth come from padding during a rotation and should not be used
                                              # warning: if this is set to 0.5 in create_rotated_training_set.py, then it comes up at 0.54 in the image files...

# Set image patch size in pixels (should be a multiple of 4 for some reason)
IMG_PATCH_SIZE = 48  # ideally, like 48

tf.app.flags.DEFINE_string('train_dir', 'tmp/mnist',
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS

# paths to stuff
data_dir = 'training/'
train_data_filename = data_dir + 'images/'
train_labels_filename = data_dir + 'groundtruth/'
test_data_filename = 'test_set_images/test_'



def pad_image(im):
    # pad the image with 0.5 (gray)
    padded_image = numpy.full(
        (IMG_PATCH_SIZE + im.shape[0] + IMG_PATCH_SIZE, IMG_PATCH_SIZE + im.shape[1] + IMG_PATCH_SIZE, im.shape[2]),
        PADDING_COLOR, dtype='float32')
    padded_image[IMG_PATCH_SIZE:IMG_PATCH_SIZE + im.shape[0], IMG_PATCH_SIZE:IMG_PATCH_SIZE + im.shape[1], :] = im
    return padded_image


def get_padded_images(filename, images_range):
    imgs = []
    for i in images_range:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(pad_image(img))
        else:
            print ('File ' + image_filename + ' does not exist')
    return imgs


def extract_samples_of_labels(filename, images_range):
    gt_imgs = []
    for i in images_range:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    ret = [[],[]]
    for k in range(num_images):
        for i in range(0, gt_imgs[k].shape[1]): # height
            for j in range(0, gt_imgs[k].shape[0]): # width
                is_on = gt_imgs[k][j,i] > 0.5
                if abs(gt_imgs[k][j,i] - SPECIAL_PADDING_COLOR_FOR_GROUNDTRUTH) < 0.01:
                    # ignore this pixel, since it comes from a padding-during-rotation
                    # (or possibly is legitimate, but is close to 0.5 so it doesn't hurt to not use it)
                    pass
                else:
                    ret[is_on].append((k, j, i))
    return ret


def get_patch(padded_image, j, i):
    j += IMG_PATCH_SIZE
    i += IMG_PATCH_SIZE
    assert(len(padded_image.shape) == 3)
    ret = padded_image[j-IMG_PATCH_SIZE//2:j+IMG_PATCH_SIZE//2, i-IMG_PATCH_SIZE//2:i+IMG_PATCH_SIZE//2, :]
    assert(ret.shape == (IMG_PATCH_SIZE, IMG_PATCH_SIZE, 3))
    return ret


def get_data_from_tuples(train_tuples, train_images_padded):
    ret = []
    for (k,j,i) in train_tuples:
        assert(len(train_images_padded) > k)
        patch = get_patch(train_images_padded[k], j, i)
        assert(len(patch.shape) == 3)
        ret.append(patch)
    return numpy.array(ret)


def get_labels_from_simple_labels(train_labels_simple):
    ret = []
    for x in train_labels_simple:
        if x == 0:
            ret.append([1, 0])  # note it's kind of backwards...
        else:
            ret.append([0, 1])
    return numpy.array(ret)


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])


# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight):
        for j in range(0,imgwidth):
            # note it's kind of backwards... (because we mapped: 0 -> [1,0], 1 -> [0,1])
            array_labels[j, i] = 1 - labels[idx][0]
            idx = idx + 1
    return array_labels


def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg


def main(argv=None):  # pylint: disable=unused-argument

    def prepare_training_tuples_and_simple_labels():
        # preparing tuples takes ~20 seconds, but longer if rewritten to numpy
        train_images_padded = get_padded_images(train_data_filename, TRAINING_SET)
        train_tuples_of_label = extract_samples_of_labels(train_labels_filename, TRAINING_SET)

        print('Tuples and labels are loaded')

        c0 = len(train_tuples_of_label[0])
        c1 = len(train_tuples_of_label[1])
        print ('Number of data points per class: c0 =', c0, ', c1 =', c1)

        print ('Balancing training tuples...')
        sys.stdout.flush()
        assert(c0 > c1)  # this is what happens in the training set
        # add copies of c1 so that c1 > c0, then truncate c1 to become c0
        random.shuffle(train_tuples_of_label[1])
        multiplier = int(math.ceil(c0 / c1))
        train_tuples_of_label[1] *= multiplier  # e.g. [1,2,3] * 2 = [1,2,3,1,2,3]
        del train_tuples_of_label[1][c0:]  # truncate
        c1 = len(train_tuples_of_label[1])
        assert(c0 == c1)

        # now merge the training tuples: first c0, then c1
        train_tuples = numpy.array(train_tuples_of_label[0] + train_tuples_of_label[1])
        train_labels_simple = numpy.array([0] * c0 + [1] * c1)

        print('Training tuples are ready')

        return train_images_padded, train_tuples, train_labels_simple


    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS))

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 64), 512],
                            stddev=0.1,
                            seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    fc2_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))


    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv2 = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        pool2 = tf.nn.max_pool(relu2,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        # Uncomment these lines to check the size of each layer
        # print 'data ' + str(data.get_shape())
        # print 'conv ' + str(conv.get_shape())
        # print 'relu ' + str(relu.get_shape())
        # print 'pool ' + str(pool.get_shape())
        # print 'pool2 ' + str(pool2.get_shape())


        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool2.get_shape().as_list()
        reshape = tf.reshape(
            pool2,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        out = tf.matmul(hidden, fc2_weights) + fc2_biases

        return out



    # new in our code
    prediction_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE_FOR_PREDICTION, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS))
    output_for_prediction = tf.nn.softmax(model(prediction_data_node))

    # Get prediction for given input image 
    def get_prediction(img):
        padded_image = pad_image(img)

        pairs_JI = []
        for i in range(img.shape[1]):  # height
            for j in range(img.shape[0]):  # width
                pairs_JI.append((j,i))
        output_predictions = []  # array of output predictions
        for offset in range(0, len(pairs_JI), BATCH_SIZE_FOR_PREDICTION):
            print('Beginning offset', offset, 'out of', len(pairs_JI))
            sys.stdout.flush()
            current_pairs_JI = pairs_JI[offset : offset+BATCH_SIZE_FOR_PREDICTION]

            # if the batch is not full, then we have to pad it with some junk to the right length
            # (in the first axis) because the tf.placeholder prediction_data_node is of fixed size
            # so we just add the first row the right number of times
            padding_rows = BATCH_SIZE_FOR_PREDICTION - len(current_pairs_JI)
            for _ in range(padding_rows):
                current_pairs_JI.append(current_pairs_JI[0])

            current_data = numpy.asarray([get_patch(padded_image,j,i) for (j,i) in current_pairs_JI])
            current_output_prediction = s.run(output_for_prediction, feed_dict={prediction_data_node: current_data})

            # now unpad the result
            if padding_rows > 0:
                current_output_prediction = current_output_prediction[ : current_output_prediction.shape[0] - padding_rows]

            output_predictions.append(current_output_prediction)
        output_prediction = numpy.concatenate(output_predictions)
        img_prediction = label_to_img(img.shape[0], img.shape[1], output_prediction)

        return img_prediction


    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True) # BATCH_SIZE*NUM_LABELS
    # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits, train_labels_node))

    all_params_node = [conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc1_weights, fc1_biases, fc2_weights, fc2_biases]
    all_grads_node = tf.gradients(loss, all_params_node)
    all_grad_norms_node = []
    for i in range(0, len(all_grads_node)):
        norm_grad_i = tf.global_norm([all_grads_node[i]])
        all_grad_norms_node.append(norm_grad_i)

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        1000000,             # Decay step.
        0.95,                # Decay rate. (note that decay is slow: we do 1600000 iters / hour)
        staircase=True)

    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.0).minimize(loss, global_step=batch)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Create a local session to run this computation.
    with tf.Session() as s:


        # TODO: resuming the training from partial results?
        if RESTORE_MODEL:
            # Restore variables from disk.
            saver.restore(s, FLAGS.train_dir + "/model.ckpt")
            print("Model restored.")

        else:
            # load the training data
            train_images_padded, train_tuples, train_labels_simple = prepare_training_tuples_and_simple_labels()
            train_size = len(train_tuples)

            # Run all the initializers to prepare the trainable parameters.
            tf.initialize_all_variables().run()

            print ('Initialized!')
            # Loop through training steps.
            print ('Total number of iterations = ' + str(int(NUM_EPOCHS * train_size / BATCH_SIZE)))

            training_indices = range(train_size)

            for iepoch in range(NUM_EPOCHS):
                print("Starting epoch number", iepoch+1)

                # Permute training indices
                perm_indices = numpy.random.permutation(training_indices)

                for step in range (int(train_size / BATCH_SIZE)):

                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                    batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.

                    batch_data = get_data_from_tuples(train_tuples[batch_indices, :], train_images_padded)
                    batch_labels = get_labels_from_simple_labels(train_labels_simple[batch_indices])

                    # This dictionary maps the batch data (as a numpy array) to the
                    # node in the graph is should be fed to.
                    feed_dict = {train_data_node: batch_data,
                                 train_labels_node: batch_labels}

                    # Run the graph and fetch some of the nodes.
                    _, l, lr, predictions = s.run(
                        [optimizer, loss, learning_rate, train_prediction],
                        feed_dict=feed_dict)

                    if step % RECORDING_STEP == 0:
                        print('Epoch %.2f' % (float(step) * BATCH_SIZE / train_size))
                        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                        print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
                        sys.stdout.flush()

                    if step % SAVING_MODEL_TO_DISK_STEP == 0:
                        # Save the variables to disk.
                        save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")
                        print("Model saved in file: %s" % save_path)

        # Getting the prediction heat-map for the training images. Stored in 'predictions_training'
        print ("Running prediction on training set")
        prediction_training_dir = "predictions_training/"
        if not os.path.isdir(prediction_training_dir):
            os.mkdir(prediction_training_dir)
        for i in TRAINING_SET:
            print("Processing image", i)
            imageid = "satImage_%.3d" % i
            image_filename = train_data_filename + imageid + ".png"
            pimg = get_prediction(mpimg.imread(image_filename))
            scipy.misc.imsave(prediction_training_dir + "prediction_" + str(i) + ".png", pimg)

        # Getting the prediction heat-map for the test images. Stored in 'predictions_test'
        print ("Running prediction on test set")
        prediction_test_dir = "predictions_test/"
        if not os.path.isdir(prediction_test_dir):
            os.mkdir(prediction_test_dir)
        for i in range(1, TEST_SIZE+1):
            print("Processing image", i)
            image_filename = test_data_filename + str(i) + '/test_' + str(i) + '.png'
            pimg = get_prediction(mpimg.imread(image_filename))
            scipy.misc.imsave(prediction_test_dir + "prediction_" + str(i) + ".png", pimg)

if __name__ == '__main__':
    tf.app.run()
