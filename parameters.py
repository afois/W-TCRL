from brian2 import *
import numpy as np

################################## SIMULATION PARAMETERS ##################################
n_jobs = 4
diff_method = "euler"
time_step = 0.1 * ms

################################## DATA PREPROCESSING PARAMETERS ##################################

# lower and upper bound for normalization of input data
lb = 0.05
ub = 0.95

# maximum reconstruction error when there is no BMU (no spikes)
max_error = 1

################################## NUMBER OF STEPS FOR INPUT DATA PROCESSING ##################################

# number of presentations of an input vector
nb_oscillations = 1
# maximum period (between last spike emission and 0 ms) for a scalar in range {O,1}
max_period_spikes = 12.5 * ms
# interval of time to process one input vector
# [0,max_period_spikes] : input_current from receptive fields
# [max_period_spikes, 2* max_period_spikes] : null current
period_oscil = max_period_spikes * 2


# time step to provide a new input vector to the network
delta_t_new_input =  nb_oscillations * period_oscil
# time step to read the weights between u and v layers
statemon_w_u2v_train = period_oscil * 100
statemon_w_u2v_test = period_oscil

################################## NEURONS AND SYNAPSES PARAMETERS ##################################

#########  (A) Number of neurons and network shape #########
bank_size = 10

######### (C) Neuronal parameters  ########

# time constants
tau_m = 10.0 * ms

# membrane potential after reset
v_rest = 0.0

# spiking thresholds
theta_u = 0.5

# reset after spike value
theta_reset_u = 0.0
theta_reset_v = 0.0

# refractory periods
# period for u is approximatively the max spike interval for a scalar in range {0,1}
refrac_per_u = 6 * ms
refrac_per_v = 6 * ms

########## Synaptic parameters for different synapse types #########

# maximum synaptic weight in the network (for plastic or static connections)
w_max = 1.0

####  u -> v synapses ####
# parameters for STDP in afferent connections

#### v <-> v  (SOM) ####
# parameters for DOG in lateral connections
sigma = 0.3
