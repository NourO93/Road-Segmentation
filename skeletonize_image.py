from skimage.morphology import skeletonize
# from skimage import io
from skimage import color
# import numpy as np

'''
Skeletonizes a ground truth image
(returns a numpy array of booleans)
'''


def skeletonize_ground_truth_image(ground_truth_image):

    # convert ground_truth_image into binary
    ground_truth_image = color.rgb2gray(ground_truth_image)
    THRESHOLD = 127
    ground_truth_image[ground_truth_image < THRESHOLD] = 0
    ground_truth_image[ground_truth_image >= THRESHOLD] = 1

    # pad with ones on border (does not help; and padding with zeros does not do anything)
    # BORDER_PAD_WIDTH = 10
    # paddedImage = np.ones((BORDER_PAD_WIDTH+image.shape[0]+BORDER_PAD_WIDTH, BORDER_PAD_WIDTH+image.shape[1]+BORDER_PAD_WIDTH))
    # paddedImage[BORDER_PAD_WIDTH:BORDER_PAD_WIDTH+image.shape[0], BORDER_PAD_WIDTH:BORDER_PAD_WIDTH+image.shape[1]] = image
    # image = paddedImage

    # perform skeletonization
    skeleton = skeletonize(ground_truth_image)

    # unpad back
    # skeleton = skeleton[BORDER_PAD_WIDTH:BORDER_PAD_WIDTH+image.shape[0], BORDER_PAD_WIDTH:BORDER_PAD_WIDTH+image.shape[1]]

    # saving the resulting image to file is done as follows (just running imsave gives a black image):
    # skeleton = skeleton.astype('uint8')
    # skeleton[skeleton > 0] = 255
    # io.imsave('out%d.png' % BORDER_PAD_WIDTH, skeleton, cmap=plt.cm.gray)

    # on how to just display the skeleton, see:
    # http://scikit-image.org/docs/0.12.x/auto_examples/edges/plot_skeleton.html

    return skeleton