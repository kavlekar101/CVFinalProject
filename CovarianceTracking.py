import numpy as np
import scipy
import ImagePyramid as ip

def find_covariance(frame):
    # 33 x 35 x 3
    feature_matrix = []
    for y in range(len(frame)):
        for x in range(len(frame[0])):
            feature_matrix.append([x, y, frame[y][x][0], frame[y][x][1], frame[y][x][2]])
    
    feature_matrix = np.array(feature_matrix, dtype=object)
    
    feature_matrix = feature_matrix.astype(np.float64)
    model_cov = np.cov(feature_matrix, rowvar=False, bias=True)
    
    return model_cov

def find_windows(frame, shape):
    points = []
    windows = []
    for y in range(len(frame)):
        for x in range(len(frame[0])):
            windows.append(frame[y:y+shape[0], x:x+shape[1]])
            points.append([y, x])
    
    return np.array(windows, dtype=object), np.array(points, dtype=object)

def find_window_covariances(windows):
    window_covariances = []
    for window in windows:
        window_covariances.append(find_covariance(window))

    return np.array(window_covariances, dtype=object)

def riemannian_distance(eigenvalues, epsilon=1e-10):
    eigenvalues = np.maximum(eigenvalues, epsilon)
    return np.sqrt(np.sum(np.log(eigenvalues)**2))

def find_distances(covs, modelCovMatrix):
    distances = []
    for c in covs:
        if not c.shape == modelCovMatrix.shape:
            distances.append(np.inf)
            continue
        eigenvalues, _ = scipy.linalg.eig(c, modelCovMatrix)
        distances.append(riemannian_distance(eigenvalues).real)
        
def find_ball(model, frame):
    model_cov = find_covariance(model)
    
    pyramid = ip.create_pyramid(frame)[::-1] # reverse the pyramid so that the smallest image is first
    point = []
    search_window = [0, 0, pyramid[0].shape[0], pyramid[0].shape[1]]
    for level in pyramid:
        windows, points = find_windows(level[search_window[0]:search_window[0]+search_window[2], search_window[1]:search_window[1]+search_window[3]], model.shape)
        window_covariances = find_window_covariances(windows)
        distances = find_distances(window_covariances, model_cov)
        min_distance = np.argmin(distances)
        point = points[min_distance]
        search_window[0] = point[0] * 2
        search_window[1] = point[1] * 2
        search_window[2] = model.shape[0] * 2
        search_window[3] = model.shape[1] * 2
    
    return point
