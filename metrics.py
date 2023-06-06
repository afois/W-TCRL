import numpy as np

def euclidianToroidalDistance (x,
                               x_arr):

    if type(x) in [float,int, tuple]:
        x = np.array(x)
    if type (x_arr) in [float,int, tuple]:
        x_arr = np.array(x_arr)

    delta = np.abs(x - x_arr)
    toric_dist = np.where(delta < 0.5, delta, 1.0  - delta)
    # if the data dimension is 1, then euclidian distance is the same as the absolute value norm
    if len (toric_dist.shape) < 2 :
        return toric_dist
    else :
        return np.sqrt(np.sum(toric_dist**2, axis = -1))


def pairwise_euclidian_toroidal (data):
    len_data = len(data)
    pair_dist_mat = np.zeros((len_data,len_data), dtype = "float")
    for idx,vector in enumerate (data):
        pair_dist_mat[idx] = euclidianToroidalDistance(vector, data)
    return pair_dist_mat


# input_dist_mat : numpy array of parwise distances in the input space
# som_dist_mat : numpy array of parwise distances in the map
def emds (input_dist_mat, som_dist_mat):
    return np.mean((input_dist_mat - som_dist_mat)**2)

def getToricCentroid (weights, represented_val, nb_neurons = 10) :
    # we need to shift the weights around the center of the array
    shifted_weights = np.empty(shape = weights.shape)
    pos_max_weight = np.argmax(weights)
    pos_center = int((nb_neurons/2) -1)

    shift = pos_center - pos_max_weight

    # shift/translate the weights by the distance between pos_max and pos_center
    shifted_weights = np.roll(weights, shift)

    # compute the estimated centroid with the centered weights
    estimated_centroid = np.sum(shifted_weights * represented_val) / np.sum(shifted_weights)

    # then shift this centroid to get back the correct represented value
    estimated_centroid -= (shift / nb_neurons)

    return estimated_centroid

def rmse (x, y, max_error = 1, axis = None):
    assert np.array_equal(np.array(x.shape), np.array(y.shape))
    error = np.array((x - y) **2)
    error[np.isnan(error)] = max_error**2
    return np.sqrt(np.mean(error, axis = axis))

def mean_rmse (x, y, max_error = 1, axis = None):
    rmse_per_vec = [rmse (x_d, y_d) for x_d, y_d in zip (x,y)]
    return np.mean(rmse_per_vec)

def rmse_per_input (x, y, max_error = 1):
    assert np.array_equal(np.array(x.shape), np.array(y.shape))
    error = np.array((x - y) **2)
    error[np.isnan(error)] = max_error**2
    return np.sqrt(np.mean(error, axis = -1))
