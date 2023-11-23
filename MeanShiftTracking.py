import numpy as np
import math

# the max value x or y can be is 25 when the other is 0, so we can reduce the search space to a 50x50 square
# generalizing this, we can reduce the search space to a 2*radius x 2*radius square
# we still need the check if the point is within the distance because the box contains points 
# that are not in the circle that we are searching through
def circularNeighbors(img, x, y, radius):
    feature_matrix = []
    for row in range(max(math.floor(-radius + y), 0), min(math.ceil(radius + y), img.shape[0])):
        for col in range(max(math.floor(-radius + x), 0), min(math.ceil(radius + x), img.shape[1])):
            if ((col - x)**2 + (row - y)**2) < (radius**2):
                feature_matrix.append(np.array([col, row, img[row,col,0], img[row,col,1], img[row,col,2]]))
                
    return feature_matrix


def createRanges(bins):
    # creates an evenly spaced array of bins that is along the range of 255
    splits = np.linspace(0, 255, bins + 1)
    
    ranges = []
    for i in range(len(splits) - 1):
        ranges.append((int(np.ceil(splits[i])), int(np.floor(splits[i+1]))))
        
    return ranges

def colorHistogram(X, bins, x, y, h):
    # creates the ranges to look through
    ranges = createRanges(bins)
    
    # makes a histogram that is of size bins x bins x bins
    # x = red, y = green, h = blue
    # so hist[x, y, z] = the number of pixels that have red values corresponding to the ranges that x represents, and so on.
    hist = np.zeros((bins, bins, bins))
    
    # finds the bin in given a certain pixel value
    def findBin(val):
        for i in range(len(ranges)):
            if val >= ranges[i][0] and val <= ranges[i][1]:
                return i
            
    # just a function that weights the pixels based on their distance from the center
    # the center being x and y
    # you can see that if the distance is 0, then the weight is 1
    def epanechnikovKernel(new_x, new_y, h):
        r = (np.sqrt((x - new_x)**2 + (y - new_y)**2) / h)**2
        if r < 1:
            return 1 - r
        else:
            return 0
        
    for feature in X:
        hist[findBin(feature[2]), findBin(feature[3]), findBin(feature[4])] += epanechnikovKernel(feature[0], feature[1], h)
    
    # this normalizes the histogram so that it sums to 1
    hist = hist / np.sum(hist)
    
    return hist

# q_model is the histogram of the thing that we are trying to track
# p_test is the histogram of the thing that we are predicting is the thing that we are trying to track
# X is the feature matrix of the image that we are trying to predict for a certain neighborhood
# this weight vector will point us in the direction of the thing that we are trying to track
def meanshiftWeights(X, q_model, p_test, bins):
    ranges = createRanges(bins)
    
    weights = []
    
    
    # finds the bin in given a certain pixel value
    def findBin(val):
        for i in range(len(ranges)):
            if val >= ranges[i][0] and val <= ranges[i][1]:
                return i
            
    # this is the bhattacharya weight thing that we have to compute
    # so what this does is take the square root of the ratio of the q_model to the p_test
    # so for each feature we have, we just find the value in the histogram and then take the square root of the ratio
    # the ratio has to be in this correct order according to the formula
    for x in X:
        model_pixel_value = q_model[findBin(x[2]), findBin(x[3]), findBin(x[4])]
        candidate_pixel_value = p_test[findBin(x[2]), findBin(x[3]), findBin(x[4])]
        weights.append(np.sqrt(model_pixel_value / candidate_pixel_value))
        
    return weights

def meanshiftTracking(img1, img2, starting_x, starting_y):
    radius = 10
    # starting_x = 149 # because we want the 150th column and python starts at 0 not 1 like a complete buffoon like MatLab
    # starting_y = 174 # because we want the 175th row and python starts at 0 not 1 like a complete buffoon like MatLab
    bins = 16
    iterations = 25

    X = circularNeighbors(img1, starting_x, starting_y, radius)
    model_hist = colorHistogram(X, bins, starting_x, starting_y, radius)

    tracking_x = starting_x
    tracking_y = starting_y
    dist = np.inf # insanely big value even though this doesn't matter

    # print the distance between the i = 23 and i = 24

    vector_distances = []

    for i in range(iterations):
        X_test = circularNeighbors(img2, tracking_x, tracking_y, radius)
        p_test = colorHistogram(X_test, bins, tracking_x, tracking_y, radius)
        weights = meanshiftWeights(X_test, model_hist, p_test, bins)
        
        # calculate the new centroid for x and y
        x_update = 0
        y_update = 0
        for i, x_test in enumerate(X_test):
            x_update += weights[i] * x_test[0]
            y_update += weights[i] * x_test[1]
        
        sum_of_weights = sum(weights)
        x_update = x_update / sum_of_weights
        y_update = y_update / sum_of_weights
        
        
        dist = np.sqrt((tracking_x - x_update)**2 + (tracking_y - y_update)**2)
        vector_distances.append(dist)
        
        # update before we potentially break out of the loop
        tracking_x = x_update
        tracking_y = y_update

    # print(f"These are the final x/column and y/row values: {tracking_x, tracking_y}")
    # print(f"The last two distances computed: {vector_distances[-2:]}")
    # print(vector_distances)
    return tracking_x, tracking_y