import numpy as np


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    For extra credit you can implement various forms of spatial/geometric
    verification of matches, e.g. using the x and y locations of the features.

    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    
    # parameter for tuning #
    nndr_ratio = 0.8 # the smaller, the less error
    norm_ord = 1 # 1 is really better than 2
    ########################
    
    n = features1.shape[0]
    m = features2.shape[0]
    match_conf = []
    
    for i in range(n):
        r_min = np.Inf
        r_min2 = r_min
        count_j = 0
        for j in range(m):
            r = np.linalg.norm(features1[i]-features2[j],ord=norm_ord)
            if r < r_min:
                r_min2 = r_min
                r_min = r
                count_j = j
        
        if r_min/r_min2 < nndr_ratio:
            match_conf.append([i,count_j,1-r_min/r_min2])

    match_conf = np.array(match_conf)
    match_conf = match_conf[match_conf[:,2].argsort(),:] # sort matches by confidence
    match_conf = match_conf[::-1,:]
    matches = match_conf[:,:2].astype('int32')
    confidences = match_conf[:,2]

#     raise NotImplementedError('`match_features` function in ' +
#         '`student_feature_matching.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return matches, confidences
