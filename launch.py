import os
import numpy as np
from joblib import Parallel, delayed
import itertools
import pickle

from model import *
from load_datasets import load_MNIST, load_natural_images


def parallel_sim(comb, optimized_params):
    dataset_name = comb[0]
    patch_size = comb[1]
    nb_neurons = comb[2]
    seed_nb = comb[3]
    new_im_size = comb[4]
    directory = "tmp/"

    if dataset_name == "MNIST":
        # scale the dataset in range [lb,ub]
        lb = 0.15
        ub = 0.85

        dataset = load_MNIST(seed=seed_nb,
                             nb_testing_img=1000,
                             nb_samples_train=60000,
                             nb_samples_test=None,
                             patch_size=patch_size,
                             new_im_size=new_im_size)

    if dataset_name == "NATURAL_IMAGES":
        # scale the dataset in range [lb,ub]
        lb = 0.05
        ub = 0.95

        dataset = load_natural_images(seed=seed_nb,
                                      nb_training_img=10,
                                      nb_testing_img=10,
                                      nb_samples_train=60000,
                                      nb_samples_test=None,
                                      patch_size=patch_size)

    res = run_model(seed_nb=seed_nb,
                    build_dir=directory,
                    nb_neur_v=nb_neurons,
                    train_dataset=dataset["train"],
                    test_dataset=dataset["test"],
                    lb=lb,
                    ub=ub,
                    dtype_dataset=np.float16,
                    nb_iter_get_weights_train = 10,
                    **optimized_params)

    res["dataset_name"] = dataset_name
    res["patch_size"] = patch_size
    res["nb_neurons"] = nb_neurons
    res["seed"] = seed_nb

    dir_name = "results/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    filename = f"{dir_name}{dataset_name}_{patch_size}_{nb_neurons}_{seed_nb}"
    with open(filename, 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return None


if __name__ == '__main__':
    n_jobs = 2  # number cores used for parallel sim

    optimized_params = {'A_plus_u2v': 0.004,
                        'scale_A_minus': 6.0,
                        'scale_max_inh': 91.0,
                        'scale_min_inh': 9.0,
                        'tau_f_aff_v_exc': 2.8,
                        'tau_f_lat_v': 0.3,
                        'tau_minus_u2v': 4.3,
                        'tau_m_v': 1.5,
                        'tau_plus_u2v': 1.3,
                        'theta_stdp_u2v': 0.1,
                        'theta_v': 0.25,
                        'w_offset': 0.2}

    datasets_names = ["MNIST", "NATURAL_IMAGES"]
    patch_sizes_MNIST = [(5, 5)]
    patch_sizes_NAT_IMG = [(16, 16)]
    nb_neurons = [16, 32, 64, 128, 256]
    seeds = np.arange(30)
    new_im_size = [(30, 30)]

    all_comb = []

    # combinations for MNIST dataset
    for comb in itertools.product([datasets_names[0]], patch_sizes_MNIST, nb_neurons, seeds, new_im_size):
        all_comb.append(comb)

    # combinations for Natural Images dataset
    for comb in itertools.product([datasets_names[1]], patch_sizes_NAT_IMG, nb_neurons, seeds, [None]):
        all_comb.append(comb)

    #launch parallel simulations
    with Parallel(n_jobs=n_jobs) as parallel:
        all_sim = parallel(delayed(parallel_sim)(
            comb, optimized_params) for comb in all_comb)