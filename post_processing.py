#!/usr/bin/env python3

import os
from os import listdir
from os.path import isfile, join
import numpy as np
import scipy
import matplotlib.image as mpimg
import re
from evaluation import mfs_files
from PIL import Image
from scipy import ndimage as ndi



IMG_PATCH_SIZE = 16

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(os.path.join(test_pred_dir, image_filename))
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img
    
def fill_holes(im):
    return 1 - ndi.binary_fill_holes(1 - im, structure=np.ones((3,3))).astype(int)
    
    
def greyscale_to_pred(im, w, h, hole_filling):
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    pred = np.zeros(im.shape)
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
                if(np.mean(im_patch) > foreground_threshold):
                    pred[j:j+w, i:i+h] = np.ones(im_patch.shape)
                else:
                    pred[j:j+w, i:i+h] = np.zeros(im_patch.shape)
            else:
                print('Not 2d')
    if(hole_filling == True):
        pred = fill_holes(pred)
    return pred
    

test_pred_dir = 'predictions_test'
test_dir = 'test_set_images'
test_preds = [f for f in listdir(test_pred_dir) if (isfile(join(test_pred_dir, f)) and f.startswith('prediction'))]
test_preds = test_preds
NUM_TEST_PREDS = len(test_preds)

# Get predicted images in binary
for test_pred in test_preds:
    img = mpimg.imread(os.path.join(test_pred_dir, test_pred))
    pred = greyscale_to_pred(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE, True)
    scipy.misc.imsave(os.path.join(test_pred_dir, 'bw_' + test_pred), pred)
    
# Overlay for test
for i in range(1, NUM_TEST_PREDS + 1):
    img = mpimg.imread(os.path.join(test_dir, 'test_' + str(i) + '.png'))
    prd = mpimg.imread(os.path.join(test_pred_dir, 'bw_prediction_') + str(i) + '.png')
    overlay = make_img_overlay(img, prd)
    overlay.save(os.path.join(test_pred_dir, 'overlay_' + str(i) + '.png'))
    
if(NUM_TEST_PREDS == 50):
    submission_filename = 'submission.csv'
    masks_to_submission(submission_filename, *test_preds)
    

########## TRAIN DATA ##############

train_pred_dir = 'predictions_training'
train_dir = 'training'
train_preds = [f for f in listdir(train_pred_dir) if (isfile(join(train_pred_dir, f)) and f.startswith('prediction'))]
NUM_TRAIN_PREDS = len(train_preds)

for train_pred in train_preds:
    img = mpimg.imread(os.path.join(train_pred_dir, train_pred))
    pred = greyscale_to_pred(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE, True)
    scipy.misc.imsave(os.path.join(train_pred_dir, 'bw_' + train_pred), pred)
    
for i in range(1, NUM_TRAIN_PREDS + 1):
    imId = "satImage_%.3d" % i
    img = mpimg.imread(os.path.join(train_dir, 'images', imId + '.png'))
    prd = mpimg.imread(os.path.join(train_pred_dir, 'bw_prediction_') + str(i) + '.png')
    overlay = make_img_overlay(img, prd)
    overlay.save(os.path.join(train_pred_dir, 'overlay_' + str(i) + '.png'))
    
total_score=0.0
for i in range(1, NUM_TRAIN_PREDS + 1):
    gt='training/groundtruth'+("/chunky_satImage_%.3d" % i)+'.png'
    pred_nn='predictions_training/bw_prediction_'+str(i)+'.png'
    sc=mfs_files(pred_nn, gt,foreground_threshold)
    total_score+=sc
    print(('Score for Training sample %.3d'%i)+(' %.3f'%sc))
print('Average Score %.3f'%(total_score/NUM_TRAIN_PREDS))

