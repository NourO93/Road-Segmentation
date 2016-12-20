import skimage.transform
import os
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


TRAINING_SIZE = 100
FIRST_NUMBER_TO_WRITE_TO = 101
PROBABILITY_OF_ROTATING = 0.3
PADDING_COLOR = 0.0  # 0.0 = black, 0.5 = gray
SPECIAL_PADDING_COLOR_FOR_GROUNDTRUTH = 0.5

# paths to stuff
data_dir = 'training/'
train_data_filename = data_dir + 'images/'
train_labels_filename = data_dir + 'groundtruth/'

next_number_to_write_to = FIRST_NUMBER_TO_WRITE_TO

random.seed(3500)

for i in range(1, TRAINING_SIZE+1):
    if random.uniform(0, 1) < PROBABILITY_OF_ROTATING:
        print('Image', i, 'was chosen')

        image_filename = train_data_filename + ("satImage_%.3d" % i) + ".png"
        original_image = mpimg.imread(image_filename)

        angle = random.uniform(-90, 90)  # in degrees

        gt_filename = train_labels_filename + ("satImage_%.3d" % i) + ".png"
        original_gt = mpimg.imread(gt_filename)

        rotated_image = skimage.transform.rotate(original_image,
                                                 angle=angle,
                                                 resize=True,
                                                 mode='constant',
                                                 cval=PADDING_COLOR)
        rotated_gt = skimage.transform.rotate(original_gt,
                                              angle=angle,
                                              resize=True,
                                              mode='constant',
                                              cval=SPECIAL_PADDING_COLOR_FOR_GROUNDTRUTH)

        rotated_image_filename = train_data_filename + ("satImage_%.3d" % next_number_to_write_to) + ".png"
        rotated_gt_filename = train_labels_filename + ("satImage_%.3d" % next_number_to_write_to) + ".png"
        next_number_to_write_to += 1

        scipy.misc.imsave(rotated_image_filename, rotated_image)
        scipy.misc.imsave(rotated_gt_filename, rotated_gt)

print('Generated images up to', next_number_to_write_to-1)