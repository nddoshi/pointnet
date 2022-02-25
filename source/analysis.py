import ipdb
import numpy as np


def feature_distance_to_set(feature_set, query_feature):
    '''
    2-norm distance from  vector query to every  vector in set
    querys is numpy array (N,)
    set is numpy array (N, M)
    '''
    query_to_set_vectors = feature_set - query_feature
    return np.linalg.norm(query_to_set_vectors, axis=1, ord=2)


def knn(feature_distance, k=1):
    '''
    return index of k nearest neighbors based on feature distance, which is (M,)
    '''
    assert k < feature_distance.shape[0] - 1
    sort_ind = np.argsort(feature_distance)
    return sort_ind[1:k+1].tolist()
