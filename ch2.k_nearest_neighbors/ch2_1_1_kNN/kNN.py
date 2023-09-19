from numpy import *

import operator

def create_data_set():
    group = array([[1.0,1.1],
    [1.0,1.0],[0,0],[0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, data_set, labels, k):
    data_set_size = data_set.shape[0]
    diff_max = tile(inX, (data_set_size, 1)) - data_set
    sq_diff_max = diff_max**2
    sq_distances = sq_diff_max.sum(axis=1)
    distances = sq_distances**0.5
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_I_label = labels[sorted_dist_indicies[i]]
        class_count[vote_I_label] = class_count.get(vote_I_label, 0) + 1
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]