import numpy as np
from preprocessing import *


def valueToAngle(data, data_max=1):
    """ transform a data point into an angle (in radian)
    Parameters
    ----------
    data : scalar or numpy.ndarray
        Array of cartesian (position) data
    data_max : float
        system size
    Returns
    -------
    value : scalar or numpy.ndarray
    """
    return (data / data_max) * 2 * np.pi


def angleToValue(angle, data_max=1):
    """ transform a data point into an angle (in radian)
    Parameters
    ----------
    angle : float or numpy.ndarray
        angle assumed to be in radian
    data_max : float
        system size (size in cartesian coordinate)
    Returns
    -------
    angle : scalar or numpy.ndarray
    """
    return (data_max * angle) / (2 * np.pi)


def circularMean(data, weights=None, axis=-1):
    """ Computes the circular mean angle of an array of circular data
        (i.e. a system with periodic condition where 0° = 360°)
        if weights are provided, then it computes a center of mass for circular data
    Parameters
    ----------
    data : numpy.ndarray
        Array of circular (angular) data, which is assumed to be in radians
    axis : int, optional
        Axis along which circular means are computed. The default is to compute
        the mean of the flattened array.
    weights : numpy.ndarray, optional
        weights assigned to the data
    Returns
    -------
    Circular mean : numpy.ndarray
        the circular mean is an angle in radian
    """
    if weights is None:
        weights = np.ones((1, ))
    try:
        weights = np.broadcast_to(weights, data.shape)
    except ValueError:
        raise ValueError('Weights and data have inconsistent shape.')

    # convert the angle to a rectangular coordinate (2D) in a unit circle (with cos(data) ans sin(data))
    # then compute the weighted mean of these coordinates
    C = np.sum(weights * np.cos(data), axis) / np.sum(weights, axis)
    S = np.sum(weights * np.sin(data), axis) / np.sum(weights, axis)

    # map these values (C and S) to a new angle that corresponds to the circular (weighted) mean
    theta = np.arctan2(-S, -C) + np.pi
    return theta


def centroid_to_toric_centroid(data, weights=None, data_max=1, axis=-1):
    angles = valueToAngle(data, data_max)
    angle_centroid = circularMean(angles, weights, axis)
    cartesian_centroid = angleToValue(angle_centroid, data_max)
    return cartesian_centroid


def w_one_neur_to_quant_val(represented_values,
                            weights,
                            lb,
                            ub,
                            min_dataset,
                            max_dataset,
                            size_rf=1,
                            axis=-1):
    """ Compute the quantized value of a set of weights connecting a bank of neurons in u to a neuron in v
    Parameters
    ----------
    represented_values : numpy.ndarray
        Array of preferential values of presynaptic neurons (e.g. [0.05, 0.15, ...])
    weights : numpy.ndarray
        Array of weights between a bank of neurons in u and a neuron in v
    lb : float
        Lower bound used to normalize the dataset
    ub : float
        Upper bound used to normalize the dataset
    min_dataset : float
        minimum value of the dataset (before normalization)
    max_dataset : float
        maximum value of the dataset (before normalization)
    size_rf : float, optional
        size of receptive fields
    axis : int, optional
        Axis along which circular means are computed. The default is to compute
        the mean of the flattened array.

    Returns
    -------
    Quantized value : numpy.ndarray
    """
    data_dim, bank_size = weights.shape

    arr_quantized_val = np.full(data_dim, np.nan)

    for dim in range(data_dim):
        # compute circular centroid, value in range of {0,1}, i.e. the range of the circular receptive fields
        centroid = centroid_to_toric_centroid(represented_values,
                                              weights[dim, :], size_rf, axis)
        # cutoff (i.e. truncate) values not belonging to the range of the normalized dataset {lb, ub}
        # this is used to provide a safety margin for extremal values so that they don't wrap together
        # resulting in a high error (e.g. 0.01 is quantized while it was 1)
        if (centroid < lb):
            centroid = lb
        elif (centroid > ub):
            centroid = ub
        # map the value back to the original data range of the dataset
        arr_quantized_val[dim] = normalized_value_to_original_range(
            lb, ub, centroid, min_dataset, max_dataset)
    return arr_quantized_val


def w_all_neur_to_quant_val(represented_values,
                            weights_mat,
                            lb,
                            ub,
                            min_dataset,
                            max_dataset,
                            size_rf=1,
                            axis=-1):
    """ Compute the quantized value of a set of weights connecting a bank of neurons in u to a neuron in v
        Do it for the whole network.
    Parameters
    ----------
    represented_values : numpy.ndarray
        Array of preferential values of presynaptic neurons (e.g. [0.05, 0.15, ...])
    weights : numpy.ndarray
        Array of weights between a bank of neurons in u and a neuron in v
    lb : float
        Lower bound used to normalize the dataset
    ub : float
        Upper bound used to normalize the dataset
    min_dataset : float
        minimum value of the dataset (before normalization)
    max_dataset : float
        maximum value of the dataset (before normalization)
    size_rf : float, optional
        size of receptive fields
    axis : int, optional
        Axis along which circular means are computed. The default is to compute
        the mean of the flattened array.

    Returns
    -------
    Quantized value : numpy.ndarray
    """
    data_dim, bank_size, y_dim_SOM, x_dim_SOM = weights_mat.shape
    arr_quantized_val = np.full((y_dim_SOM, x_dim_SOM, data_dim), np.nan)

    for dim in range(data_dim):
        for y in range(y_dim_SOM):
            for x in range(x_dim_SOM):
                # compute circular centroid, value in range of {0,1}, i.e. the range of the circular receptive fields
                centroid = centroid_to_toric_centroid(
                    represented_values, weights_mat[dim, :, y, x], size_rf,
                    axis)
                # cutoff (i.e. truncate) values not belonging to the range of the normalized dataset {lb, ub}
                # this is used to provide a safety margin for extremal values so that they don't wrap together
                # resulting in a high error (e.g. 0.01 is quantized while it was 1)
                if (centroid < lb):
                    centroid = lb
                elif (centroid > ub):
                    centroid = ub
                # map the value back to the original data range of the dataset
                arr_quantized_val[y, x, dim] = normalized_value_to_original_range(
                    lb, ub, centroid, min_dataset,
                    max_dataset)
    return arr_quantized_val
