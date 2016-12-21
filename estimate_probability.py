import math
import numpy

'''
Given a 2D numpy array of size 16x16 of probability estimates for single pixels,
computes the probability estimates that at least foreground_threshold fraction
of the pixels are white,
ASSUMING THESE ARE INDEPENDENT EVENTS.
Uses dynamic programming, runs in time O(IMG_PATCH_SIZE ^ 4).
'''

foreground_threshold = 0.25

def is_integer(x):
    return abs(x-round(x)) < 1e-6

def estimate_probability_slow(patch_probabilities):
    IMG_PATCH_SIZE = patch_probabilities.shape[0]
    n = IMG_PATCH_SIZE * IMG_PATCH_SIZE
    prob = numpy.reshape(patch_probabilities, n)  # make a sequence out of the 2D array
    # dynamic programming:
    # dp[i][j] = probability that out of the first i pixels (prob[0 : i+1]), j were white
    dp = numpy.zeros((n + 5, n + 5))
    dp[0,0] = 1.0
    for i in range(1,n+1):
        dp[i,0] = dp[i-1,0] * (1.0 - prob[i-1])
        for j in range(1,n+1):
            dp[i,j] = dp[i-1,j] * (1.0 - prob[i-1]) + dp[i-1,j-1] * prob[i-1]
    min_above_threshold = math.ceil(n * foreground_threshold + 1e-6)
    result = numpy.sum(dp[n, min_above_threshold:])
    # if the threshold happens to be an integer, take half of its value also
    if is_integer(n * foreground_threshold):
        result += dp[n, min_above_threshold]/2
    return result


def shift_zero(x):
    return numpy.hstack(([0], x[:-1]))


def estimate_probability(patch_probabilities):
    IMG_PATCH_SIZE = patch_probabilities.shape[0]
    n = IMG_PATCH_SIZE * IMG_PATCH_SIZE
    prob = numpy.reshape(patch_probabilities, n)  # make a sequence out of the 2D array
    # dynamic programming:
    # dp[i][j] = probability that out of the first i pixels (prob[0 : i+1]), j were white
    dp = numpy.zeros(n+1)
    dp[0] = 1
    for i in range(1,n+1):
        dp = dp * (1 - prob[i-1]) + shift_zero(dp) * prob[i-1]
    min_above_threshold = math.ceil(n * foreground_threshold + 1e-6)
    result = numpy.sum(dp[min_above_threshold:])
    # if the threshold happens to be an integer, take half of its value also
    if is_integer(n * foreground_threshold):
        result += dp[min_above_threshold] / 2
    return result


if __name__ == '__main__':
    # some simple test
    patch_probabilities = numpy.full((16,16), 0.25)
    p = estimate_probability(patch_probabilities)
    print(p, numpy.log(p / (1-p)))  # close to 0.5 and to 0.0
