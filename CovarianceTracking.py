import numpy as np

def find_covariance(frame):
    # 33 x 35 x 3
    feature_matrix = []
    for y in range(len(frame)):
        for x in range(len(frame[0])):
            feature_matrix.append([x, y, frame[y][x][0], frame[y][x][1], frame[y][x][2]])
    
    feature_matrix = np.array(feature_matrix, dtype=object)
    
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
        eigenvalues, _ = np.linalg.eig(c, modelCovMatrix)
        distances.append(riemannian_distance(eigenvalues).real)
        
def find_ball(model, frame):
    model_cov = find_covariance(model)
    windows, points = find_windows(frame, model.shape)
    window_covariances = find_window_covariances(windows)
    distances = find_distances(window_covariances, model_cov)
    min_distance = np.argmin(distances)
    return points[min_distance]

