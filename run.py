print('Please note that our neural network prediction is not fast (around 20-40 minutes per input image).')
print('For more details on this, please read README.md (point 4).')

answer = input('Would you like to run the time-consuming part of the prediction? (yes/no) ')
while answer not in ('yes', 'no'):
    answer = input('Please answer `yes` or `no`: ')

if answer == 'yes':
    exec(open('tf_aerial_images_big.py').read())

exec(open('post_processing.py').read())
