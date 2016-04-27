import numpy as np

def cleanCreditDf(credit_df):
    # Clean up the column names so it fits on one page width !
    credit_df.columns = ['ID', 'target', 'revolving_pct', 'age', 
                        'past_due_30_59', 'debt_ratio', 
                        'monthly_income', 'lines', 
                        'past_due_gt_90', 'mortgages', 
                        'past_due_60_89', 'dependents']
    
    # Replace missing values with 0 
    # (quick-n-dirty ! Should try imputation or reserved value for trees)
    credit_df[credit_df.isnull()] = 0
    
    # Convert dependents to an integer
    credit_df['dependents'] = credit_df['dependents'].astype('int64')

    return credit_df

def cleanRProcessedCreditDf(credit_df):
    # Clean up the column names so it fits on one page width !
    credit_df.columns = ["ID", "revolving_pct", "age", "past_due_30_59",
                         "debt_ratio", "monthly_income", "lines",
                         "past_due_gt_90", "mortgages", "past_due_60_89",
                         "dependents", "target"]
    
    # Replace missing values with 0 
    # (quick-n-dirty ! Should try imputation or reserved value for trees)
    credit_df[credit_df.isnull()] = 0
    
    # Convert dependents to an integer
    credit_df['dependents'] = credit_df['dependents'].astype('int64')

    return credit_df


def removeIDTargetFromCreditDf(credit_df):
    """ 
    returns a tuple of X, y, and id_val from pandas dataframe
    """
    y = credit_df['target']
    id_val = credit_df['ID']
    credit_df = credit_df.drop(['target'], axis=1)
    credit_df = credit_df.drop(['ID'], axis=1)
    X = credit_df

    return X, y, id_val


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
    
    print 'num_min = {}, num_maj = {}'.format(num_min, num_maj)
    
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
    
    # Generate matrix of 2-norms. Rows = minority, Cols = all samples
    knn_dist = np.zeros((num_min, num_all))
    for X_min_idx, X_min_val in enumerate(X_min):
        for X_all_idx, X_all_val in enumerate(X):
            knn_dist[X_min_idx, X_all_idx] = np.sqrt(
            np.sum(np.square(X_min_val - X_all_val)))
            
    # Now find the k-nearest neighbours by sorting across rows
    # and taking the first [1:k+1] entries. Entry [0] will be 0 as the minority 
    # example is included in its own calculation.
    knn_sorted_idx = np.argsort(knn_dist, -1)
    knn_idx = knn_sorted_idx[:,1:k+1]
    
    # Now until we have enough new examples
    X_smote = np.zeros((num_min * ratio, X.shape[-1]))
    for smote_run in xrange(ratio):
        for X_min_idx, X_min_val in enumerate(X_min):
            a = X[knn_idx[X_min_idx, np.random.randint(k)]]
            c = np.random.uniform()
            new_smote = X_min_val + (c * (a - X_min_val))
    #         print 'x_min_val = {}, a = {}, c = {}, new_smote = {}'.
    #         format(x_min_val, a, c, new_smote)
            X_smote[(smote_run*num_min) + X_min_idx] = new_smote
    
    y_smote = np.ones_like(X_smote[:,0])
    
    x_smote_out = np.vstack((X, X_smote))
    y_smote_out = np.hstack((y, y_smote))

    return (x_smote_out, y_smote_out)


def TomekUndersample(X, y):
    """
    Undersamples majority examples found in Tomek links
    
    Inputs
    - X: A pandas dataframe containing the dataset
    - y: The target variable.
    
    Returns
    - X: Dataset with majority examples in Tomek pairs removed
    - y: Undersampled target variable
    """
    # Split data into minority and majority example arrays
    X_min = X[y == 1]
    X_maj = X[y == 0]
    num_all = X.shape[0]
    num_min = X_min.shape[0]
    num_maj = X_maj.shape[0]
    
    # Generate matrix of 2-norms between minority and majority .
    # Rows are minority indexes, columns are majority
    knn_dist = np.zeros((num_all, num_all))
    for a_idx, a_val in enumerate(X):
        for b_idx, b_val in enumerate(X):
            knn_dist[a_idx, b_idx] = np.sqrt(np.sum(np.square(a_val - b_val)))
    
            
    tomek_pairs = []
    
    for idx in range(num_all):
    #     print 'Examining point {}'.format(idx)
        
        maj_idx = None
        min_idx = None
        
        # Skip majority points 
        if y[idx] == 0:
            continue
    
        # Find closest point to minority, check if it's a majority
        dist_min_idx = np.argsort(knn_dist[idx,:])
        assert(dist_min_idx[0] == idx) 
        closest_idx = dist_min_idx[1]
        if (y[closest_idx] == 0):
            #print 'Found closest majority point = {}'.format(closest_idx)
            maj_idx = closest_idx
        else:
            continue
    
        # Now check the closest point to majority example is the minority one
        dist_maj_idx = np.argsort(knn_dist[maj_idx,:])
        assert(dist_maj_idx[0] == maj_idx)
        closest_idx = dist_maj_idx[1]
        if (y[closest_idx] == 1):
            #print 'Found closest minority point = {}'.format(closest_idx)
            min_idx = closest_idx
    
        if (min_idx != None and maj_idx != None):
            #print 'Found Tomek pair at indexes {} and {}'.format(min_idx, maj_idx)
            tomek_pairs.append((min_idx, maj_idx))
        
    #print 'Found Tomek links (min,maj) = {}'.format(tomek_pairs)    
    
    # Now remove the majority examples in the Tomek links
    remove_list = []
    for pair in tomek_pairs:
        remove_list.append(pair[1])
        
    X_tomek = np.delete(X, remove_list, 0)
    y_tomek = np.delete(y, remove_list, 0)
    
    return (X_tomek, y_tomek)





