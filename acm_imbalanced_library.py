import numpy as np

def randomOversample(X, y):
    """
    Randomly oversamples the minority class (y == 1)
    
    Inputs
    - X: A pandas dataframe containing the dataset
    - y: The target variable.
    
    Returns
    - X: Oversampled dataframe
    - y: Oversampled target variable
    """
    X_min = X[y == 1]
    X_maj = X[y == 0]
    num_min = X_min.shape[0]
    num_maj = X_maj.shape[0]
    X_min_oversamp = X_min
    
    while (X_maj.shape[0] > X_min_oversamp.shape[0]):
        X_min_idx = np.random.randint(num_min, size=num_min)
        X_min_oversamp = np.vstack((X_min_oversamp, X_min[X_min_idx]))
    
    X_oversamp = np.vstack((X_min_oversamp, X_maj))
    y_oversamp = np.vstack(np.array(([1] * num_maj) + ([0] * num_maj)))

    return (X_oversamp, y_oversamp)


def randomUndersample(X, y):
    """
    Randomly undersamples the majority class (y == 0)
    
    Inputs
    - X: A pandas dataframe containing the dataset
    - y: The target variable.
    
    Returns
    - X: undersampled dataframe
    - y: undersampled target variable
    """
    # Create a random sample with balanced examples
    X_min = X[y == 1]
    X_maj = X[y == 0]
    num_min = X_min.shape[0]
    num_maj = X_maj.shape[0]
    X_maj_idx = np.random.randint(num_maj, size=num_min)
    X_maj_undersamp = X_maj[X_maj_idx]
    
    # Combine minority and balanced majority examples
    X_undersamp = np.vstack((X_min, X_maj_undersamp))
    y_undersamp = np.vstack(np.array(([1] * num_min) + ([0] * num_min)))

    return (X_undersamp, y_undersamp)


def SMOTEoversample(X, y):
    """
    Oversamples minority examples using the SMOTE algorithm
    
    Inputs
    - X: A pandas dataframe containing the dataset
    - y: The target variable.
    
    Returns
    - X: SMOTE'd dataframe
    - y: SMOTE'd target variable
    """

    # Split data into minority and majority example arrays
    X_min = X[y == 1]
    X_maj = X[y == 0]
    num_all = X.shape[0]
    num_min = X_min.shape[0]
    num_maj = X_maj.shape[0]
    
    # SMOTE parameters
    ratio = (num_maj / num_min) - 1 # Amount of times to copy minority examples
    k = 5 # nearest neighbours to choose from
    
    # Generate matrix of 2-norms for the minority class examples
    knn_dist = np.zeros((num_min, num_min))
    for X_min_idx, X_min_val in enumerate(X_min):
        for X_min2_idx, X_min2_val in enumerate(X_min):
            knn_dist[X_min_idx, X_min2_idx] = np.sqrt(
            np.sum(np.square(X_min_val - X_min2_val)))
            
    # Now find the k-nearest neighbours by sorting across rows
    # and taking the first [1:k+1] entries. Entry [0] will be 0 as the minority 
    # example is included in its own calculation.
    knn_sorted_idx = np.argsort(knn_dist, -1)
    knn_idx = knn_sorted_idx[:,1:k+1]
    
    # Now until we have enough new examples
    X_smote = np.zeros((num_min * ratio, X.shape[-1]))
    for smote_run in xrange(ratio):
        for X_min_idx, X_min_val in enumerate(X_min):
            a = X_min[knn_idx[X_min_idx, np.random.randint(k)]]
            c = np.random.uniform()
            new_smote = X_min_val + (c * (a - X_min_val))
    #         print 'x_min_val = {}, a = {}, c = {}, new_smote = {}'.
    #         format(x_min_val, a, c, new_smote)
            X_smote[(smote_run*num_min) + X_min_idx] = new_smote
    
    y_smote = np.ones_like(X_smote[:,0])
    
    x_smote_out = np.vstack((X, X_smote))
    y_smote_out = np.hstack((y, y_smote))

    return (x_smote_out, y_smote_out)





