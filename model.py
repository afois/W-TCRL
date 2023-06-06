from brian2 import *
import numpy as np

from parameters import *
from preprocessing import *
from encoding import *
from utils import *
from neuronal_models import *
from synaptic_models import *
from decoding import w_one_neur_to_quant_val
from metrics import *
from evaluation import *
import time

def run_model(seed_nb,
              build_dir,
              nb_neur_v,
              train_dataset,
              test_dataset,
              lb=0.05,
              ub=0.95,
              dtype_dataset = np.float16,
              nb_iter_get_weights_train = 10,
              **optimized_params):

    # use float32 to save memory
    prefs['core.default_float_dtype'] = float32

    device.reinit()
    device.activate()
    defaultclock.dt = time_step
    # Use fast "C++ standalone mode"
    set_device('cpp_standalone', build_on_run=False)
    # Brian and numpy random seed
    seed(seed_nb)
    np.random.seed(seed_nb)

    nb_neur_v = int(nb_neur_v)

    data_dim = train_dataset.shape[1]
    nb_train_vec = train_dataset.shape[0]
    nb_test_vec = test_dataset.shape[0]
    training_simu_time = nb_train_vec * period_oscil
    testing_simu_time = nb_test_vec * period_oscil

    # start monitors at some point in the simulation
    start_mon_train = 0 * ms
    start_mon_test = nb_train_vec * period_oscil

    # get weights each "nb_iter_get_weights_train" oscillations
    delta_t_get_weights_train = period_oscil * nb_iter_get_weights_train
    nb_track_train_vec = int (nb_train_vec / nb_iter_get_weights_train)
    nb_iter_get_weights_test = 1
    delta_t_get_weights_test = period_oscil * nb_iter_get_weights_test

    # param found by brian
    tau_homeo_w_syn_v2v = (nb_train_vec * period_oscil) / 3

    shape_u = (data_dim, bank_size)
    nb_neur_u = np.prod(shape_u)

    # parameters relative to neurons in v layer
    tau_m_v = optimized_params["tau_m_v"] * ms
    theta_v = optimized_params["theta_v"] * data_dim * bank_size

    # time constant alpha functions of synapses connecting u -> v
    tau_f_aff_v_exc = optimized_params["tau_f_aff_v_exc"] * ms
    # time constant alpha functions of synapses connecting v <-> v
    tau_f_lat_v = optimized_params["tau_f_lat_v"] * ms

    # parameters relative to the STDP
    w_min = optimized_params["w_min"]
    w_offset = optimized_params["w_offset"]
    A_plus_u2v = optimized_params["A_plus_u2v"]
    A_minus_u2v = optimized_params["scale_A_minus"] * A_plus_u2v
    tau_plus_u2v = optimized_params["tau_plus_u2v"] * ms
    tau_minus_u2v = optimized_params["tau_minus_u2v"] * ms
    theta_stdp_u2v = optimized_params["theta_stdp_u2v"]

    # parameters relative to the neighborhood kernel
    min_inh = optimized_params["scale_min_inh"] * theta_v
    max_inh = optimized_params["scale_max_inh"] * theta_v

    ##### GENERATE THE DATASET #####
    dataset = np.concatenate((train_dataset, test_dataset), axis=0, dtype=dtype_dataset)

    min_dataset = np.min(dataset)
    max_dataset = np.max(dataset)
    #  scale the dataset in range [lb,ub]
    norm_dataset = normalize_in_range(lb, ub, dataset)

    activations = dataset_to_input_current(
        dataset=norm_dataset, bank_size=bank_size)

    represented_val = ReceptiveField(bank_size=bank_size).field_mu

    in_stimuli = TimedArray(activations, dt=max_period_spikes)

    u_layer = NeuronGroup(nb_neur_u,
                          u_layer_neuron_equ,
                          threshold='v>theta_u',
                          method=diff_method,
                          reset=reset_u_layer,
                          refractory=refrac_per_u)

    v_layer = NeuronGroup(nb_neur_v,
                          v_layer_neuron_equ,
                          threshold='v>theta_v',
                          method=diff_method,
                          reset=reset_v_layer,
                          refractory=refrac_per_v)

    # v (SOM) afferent connection (i.e. u -> v)
    u2v = Synapses(u_layer,
                   target=v_layer,
                   method=diff_method,
                   model=model_synapse_u2v,
                   on_pre=on_pre_synapse_u2v,
                   on_post=on_post_synapse_u2v)

    # warning ! Brian connect one presynaptic neuron to all postsynaptic neuron : (0,0), (0,1), (0,2), ...
    # not all presynaptic to one postsynaptic neuron
    u2v.connect()
    u2v.w_syn = '(rand() * 0.2) + 0.6'

    v2v = Synapses(v_layer,
                   v_layer,
                   method=diff_method,
                   model=model_synapse_v2v,
                   on_pre=on_pre_synapse_v2v)

    v2v.connect(condition='i!=j')
    v2v.w_syn = -min_inh

    # state and spike monitors
    # after a certain interval of time get the weight matrix of u2v
    statemon_w_u2v_train = StateMonitor(u2v, ['w_syn'], record=range(nb_neur_u * nb_neur_v),
                                        dt=delta_t_get_weights_train)
    v_spike_mon_train = SpikeMonitor(v_layer)
    v_spike_mon_test = SpikeMonitor(v_layer)

    #### TRAINING PHASE ####
    statemon_w_u2v_train.active = True
    v_spike_mon_train.active = True
    v_spike_mon_test.active = False

    # activate synaptic plasticity
    u2v.plasticity = 1
    run(training_simu_time)

    #### TESTING PHASE ####
    statemon_w_u2v_train.active = False
    v_spike_mon_train.active = False
    v_spike_mon_test.active = True

    # deactivate synaptic plasticity and fix global inhibition
    u2v.plasticity = 0
    v2v.w_syn = - max_inh
    run(testing_simu_time)

    # build and run simulation
    device.build(directory=build_dir + str(time.time()) + str(nb_neur_v) + str(seed_nb))


    ######################################################################################################################################################

    ############ ############ ############ ############  TRAIN RESULTS ANALYSIS  ############ ############ ############ ############

    ######################################################################################################################################################

    ### for each input vector, track number of emitted spikes ###
    train_spikes = get_nb_spikes_per_input(v_spike_mon_train, start_mon_train, period_oscil, nb_train_vec)

    w_u2v_train = statemon_w_u2v_train.w_syn.reshape((nb_neur_u, nb_neur_v, nb_track_train_vec))

    ### to store all decoded codevectors during training phase ###
    train_all_cv = np.full((nb_track_train_vec, nb_neur_v, data_dim), np.nan, dtype = dtype_dataset)

    # after each input vector, decode all codevectors
    for it in range(nb_track_train_vec):
        for idx_neur in range(nb_neur_v):
            train_all_cv[it, idx_neur] = w_one_neur_to_quant_val(
                represented_val,
                w_u2v_train[:, idx_neur, it].reshape(shape_u),
                lb,
                ub,
                min_dataset,
                max_dataset)

    # to store the sorted indices of neurons that minimize their distance w.r.t an input vector
    # doesn't care if neurons spiked or not (observer point of view)
    train_sorted_real_bmu_indices = np.zeros((nb_track_train_vec, nb_neur_v), dtype=int)

    # sort indices of neurons according to their euclidian distance with the input
    for it, inp_vec in enumerate(train_dataset[0:-1:nb_iter_get_weights_train]):
        dist_mat = distance_matrix(inp_vec, train_all_cv[it, :])
        train_sorted_real_bmu_indices[it] = np.argsort(dist_mat)

    ### for each input vector, get spiking bmu index and decoded spiking bmu codevector ###
    train_spik_bmu_idx = get_bmu_idx(v_spike_mon_train, start_mon_train, period_oscil, nb_train_vec)
    train_spik_bmu_cv = np.full((nb_track_train_vec, data_dim), np.nan, dtype = dtype_dataset)

    for it, idx_bmu in enumerate(train_spik_bmu_idx[0:-1:nb_iter_get_weights_train]):
        if (idx_bmu != -9999):
            train_spik_bmu_cv[it] = w_one_neur_to_quant_val(
                represented_val,
                w_u2v_train[:, idx_bmu, it].reshape(shape_u),
                lb,
                ub,
                min_dataset,
                max_dataset)


    # free the memory
    del statemon_w_u2v_train
    del v_spike_mon_train
    del w_u2v_train

    ######################################################################################################################################################

    ############ ############ ############ ############  TEST RESULTS ANALYSIS ############ ############ ############ ############

    ######################################################################################################################################################

    ### for each input vector, track number of emitted spikes ###
    test_spikes = get_nb_spikes_per_input(v_spike_mon_test, start_mon_test, period_oscil, nb_test_vec)

    w_u2v_test = u2v.w_syn[:].reshape((nb_neur_u, nb_neur_v))

    ### to store all decoded codevectors during testing phase ###
    test_all_cv = np.full((nb_neur_v, data_dim), np.nan, dtype = dtype_dataset)

    # decode all codevectors once (as they are fixed during testing phase)
    for idx_neur in range(nb_neur_v):
        test_all_cv[idx_neur] = w_one_neur_to_quant_val(
            represented_val,
            w_u2v_test[:, idx_neur].reshape(shape_u),
            lb,
            ub,
            min_dataset,
            max_dataset)

    # to store the sorted indices of neurons that minimize their distance w.r.t an input vector
    # doesn't care if neurons spiked or not (observer point of view)
    test_sorted_real_bmu_indices = np.zeros((nb_test_vec, nb_neur_v), dtype=int)
    # sort indices of neurons according to their euclidian distance with the input
    for it, inp_vec in enumerate(test_dataset):
        dist_mat = distance_matrix(inp_vec, test_all_cv[:])
        test_sorted_real_bmu_indices[it] = np.argsort(dist_mat)


    ### for each input vector, get spiking bmu index and decoded spiking bmu codevector ###
    test_spik_bmu_idx = get_bmu_idx(v_spike_mon_test, start_mon_test, delta_t_get_weights_test, nb_test_vec)
    test_spik_bmu_cv = np.full((nb_test_vec, data_dim), np.nan, dtype = dtype_dataset)

    for it, idx_bmu in enumerate(test_spik_bmu_idx):
        if (idx_bmu != -9999):
            test_spik_bmu_cv[it] = w_one_neur_to_quant_val(
                represented_val,
                w_u2v_test[:, idx_bmu].reshape(shape_u),
                lb,
                ub,
                min_dataset,
                max_dataset)

    ######################################################################################################################################################

    ############ ############ ############ ############  STORE RESULTS IN A DICTIONNARY ############ ############ ############ ############

    ######################################################################################################################################################

    results = {}

    results["train_dataset"] = train_dataset
    results["test_dataset"] = test_dataset

    results["train_spikes"] = train_spikes
    results["train_all_cv"] = train_all_cv
    results["train_sorted_real_bmu_indices"] = train_sorted_real_bmu_indices
    results["train_spik_bmu_idx"] = train_spik_bmu_idx
    results["train_spik_bmu_cv"] = train_spik_bmu_cv

    results["test_weights"] = u2v.w_syn[:].reshape(*shape_u, nb_neur_v)
    results["test_spikes"] = test_spikes
    results["test_all_cv"] = test_all_cv
    results["test_sorted_real_bmu_indices"] = test_sorted_real_bmu_indices
    results["test_spik_bmu_idx"] = test_spik_bmu_idx
    results["test_spik_bmu_cv"] = test_spik_bmu_cv

    device.delete()
    return results
