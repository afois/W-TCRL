import numpy as np
from brian2 import *


def get_bmu_idx (spike_mon, min_t, delta_t, nb_samples = None):
    delta_t = np.float64(delta_t)
    new_min_t = np.float64(min_t)
    new_max_t = np.float64(min_t) + np.float64(delta_t)

    # extract the bmu for each sample from the spike monitor
    # corresponds to the first neuron that spiked during an oscillation
    if nb_samples :
        spike_mon_time = np.asarray(spike_mon.t)
        spike_mon_idx = np.asarray(spike_mon.i)
        bmu_idx = np.full(nb_samples, -9999, dtype = int)
        iteration = 0

        for idx, val in enumerate(spike_mon_time):
            if ((val >= new_min_t) & (val < new_max_t)) :
                bmu_idx[iteration] = spike_mon_idx[idx]
                iteration += 1
                new_min_t = new_max_t
                new_max_t += delta_t

            # case where no spike was emmitted during one or more oscillation
            elif val >= new_max_t :
                # increment might be greater than 1, e.g. if no spike emitted during 2 * delta_t, then increment = 2
                increment = int(np.ceil((val - new_max_t)/delta_t))
                iteration += increment
                new_min_t += delta_t * increment
                new_max_t += delta_t * increment

                # store the bmu at the correct index
                bmu_idx[iteration] = spike_mon_idx[idx]
                # go to next input vector
                iteration += 1
                new_min_t = new_max_t
                new_max_t += delta_t

    # extract the bmu for one sample from the spike monitor
    else :
        bmu_idx = np.full(1, -9999, dtype = int)
        spike_t = np.where((spike_mon.t >= new_min_t) & (spike_mon.t < new_max_t))[0]
        if(spike_t.size != 0):
            bmu_idx[0] = spike_mon.i[spike_t[0]]

    return bmu_idx


def get_nb_spikes_per_input (spike_mon, min_t, delta_t, nb_samples = None):
    delta_t = np.float64(delta_t)
    new_min_t = np.float64(min_t)
    new_max_t = np.float64(min_t) + np.float64(delta_t)

    # extract the nb of spikes emitted for each input sample from the spike monitor
    if nb_samples :
        spike_mon_time = np.asarray(spike_mon.t)
        nb_spikes = np.zeros(nb_samples, dtype = int)
        iteration = 0

        for val in spike_mon_time :
            # increment the number of spikes emitted in response for one input vector
            if ((val >= new_min_t) & (val < new_max_t)) :
                nb_spikes[iteration] += 1

            # increment the iteration when spike time outside the temporal window
            elif val >= new_max_t :
                increment = int(np.ceil((val - new_max_t)/delta_t))
                iteration += increment
                new_min_t += delta_t * increment
                new_max_t += delta_t * increment
                nb_spikes[iteration] += 1

    return nb_spikes

# Function used to compute the distance matrix : here squared euclidian distance between weights and input vector
def distance_matrix(inp_vec, weights):
    # sum (on last axis, e.g. if the shape is in 3D : 2D grid + 1D weights) the squared difference between each component of the weights and the vector
    return np.sum((weights - inp_vec) ** 2, axis = -1)