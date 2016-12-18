#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.image as mpimg
import re
from evaluation import mfs_files

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

if __name__ == '__main__':
    total_score=0.0
    for i in range(1, 101):
        gt='training\groundtruth'+("\satImage_%.3d" % i)+'.png'
        pred_nn='predictions_training\prediction_'+str(i)+'.png'
        sc=mfs_files(pred_nn, gt,foreground_threshold)
        total_score+=sc
        print(('Score for Training sample %.3d'%i)+(' %.3f'%sc))
    print('Average Score %.3f'%total_score)
