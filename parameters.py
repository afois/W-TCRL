from brian2 import *
import numpy as np

################################## SIMULATION PARAMETERS ##################################
n_jobs = 4
diff_method = "euler"
time_step = 0.1 * ms

################################## DATA PREPROCESSING PARAMETERS ##################################

# maximum reconstruction error when there is no BMU (no spikes)
max_error = 1

################################## NUMBER OF STEPS FOR INPUT DATA PROCESSING ##################################

# maximum period (between last spike emission and 0 ms) for a scalar in range {O,1}
max_period_spikes = 12.5 * ms
# interval of time to process one input vector
# [0,max_period_spikes] : input_current from receptive fields
# [max_period_spikes, 2* max_period_spikes] : null current
period_oscil = max_period_spikes * 2


# time interval to provide a new input vector to the network
delta_t_new_input = period_oscil

################################## NEURONS AND SYNAPSES PARAMETERS ##################################

#########  Number of neurons to encode an input dimension #########
bank_size = 10

######### Neuronal parameters  ########

# time constants
tau_m_u = 10.0 * ms

# spiking thresholds
theta_u = 0.5

# membrane potential after reset
theta_reset_u = 0.0
theta_reset_v = 0.0

# refractory periods
# period for u is approximatively the max spike interval for a scalar in range [0,1]
refrac_per_u = 6 * ms
refrac_per_v = 6 * ms

# minimum and maximum synaptic weight in the network (for plastic or static connections)
w_min = 0.0
w_max = 1.0