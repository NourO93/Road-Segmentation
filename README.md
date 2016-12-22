Hello!

To run our software, one needs to do the following:

PREREQUISITES:
- the usual (NumPy, SciPy, TensorFlow...)
- we also use the Gurobi optimizer. It has a free academic license. One needs to:
   - get the license from Gurobi's website
   - download and install Gurobi and the license
   - install Gurobi's Python module (for Anaconda, `conda install gurobi` should do it).
   We used Gurobi 6.5.2, but the newest version (7.something) should work as well.


VERY SHORT VERSION:
- if you want to run prediction to an intermediate stage (this is optional), run:
    tf_aerial_images_big.py
- afterwards, run:
    post_processing.py


LONGER VERSION:
The steps we used were the following:

1. Prepare the input directories (this is already done in the archive whose contents you're reading):
      - training (as in the package we received)
      - test_set_images (here we moved all the .png files to the common directory, instead of one directory per file)
2. Run create_rotated_training_set.py. This will add several new files to the training directory
   (random rotations of a random subset of the training files). This is also already done for the archive.
3. Now, the options
        RESTORE_MODEL = True
        DO_PREDICTION_FOR_TESTING_SET = True
        DO_PREDICTION_FOR_VALIDATION_SET = False
        DO_PREDICTION_FOR_TRAINING_SET = False
   in tf_aerial_images_big.py control the action:
      - to train weights of the NN, set RESTORE_MODEL to False, otherwise set it to True
      - others are hopefully self-explanatory :) The first 30 images were our validation set,
        but for the submission, we trained the NN on all the images (obtained from also setting
        the validation set to be empty in create_rotated_training_set.py, so that more rotations were also used).
    If wanting to use a pre-trained set of weights, choose one from the `output` directory
    and set it in tf_aerial_images_big.py (look for `tf.app.flags.DEFINE_string('train_dir'`).
    We used `48, 5e-4, dropout, trained on all training data` to generate the submission.
4. Note that, since our model involves computing a value for each pixel of the test images,
   the predictions are not fast - it should take around 20-40 minutes to get one prediction on commodity hardware
   (we did it on a workstation with 20 cores - both training and prediction parallelizes very well).
   So we have included the prediction results on the testing set in our archive (under predictions_test/prediction_##.png).
   We are assuming that, while you may want to check one or two first testing images to see whether they match
   the ones provided by us, you will probably not want to run the prediction for all the 50 test images.

   This step generates "heat-maps" - i.e., for every test file e.g. test_3.png,
   it produces a file predictions_test/prediction_3.png of the same size,
   whose value in pixel (i,j) is our predicted likelihood that pixel (i,j) in test_3.png is a road.
   This is NOT what we eventually submit - it undergoes another processing phase.
5. Now run post_processing.py to "round" these results.
   Again, the options
       PROCESS_TESTING = True
       PROCESS_VALIDATION = False
   control its action.
   This will produce:
      - files bw_prediction_##.png containing our final prediction (which is just 0/1-valued and constant on 16x16 patches),
      - files overlay_##.png where this is laid over the input,
      - submission.csv.

Some other descriptions of what's in the archive:
   - the directory `predictions_training` stores the predictions of the 30 first training images, which we used
     as a validation set. These predictions were made using a model which was trained on the training set
     with these 30 images removed (this is stored under `48, 5e-4, dropout, trained without validation set`)
   - the directory `output` stores a few sets of pre-trained NN weights;
     it also stores, for each weight, a set of predictions (either test or validation set, depending on how these
     weights were trained).