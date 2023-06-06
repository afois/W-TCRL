# u neuron to v, excitatory connection (afferent connections)
model_synapse_u2v = '''
    plasticity : 1
    w_syn : 1
    # presynaptic and postsynaptic trace
    dapre/dt = -apre/tau_plus_u2v  : 1 (event-driven)
    dapost/dt = -apost/tau_minus_u2v  : 1 (event-driven)
'''

on_pre_synapse_u2v = '''
    s_aff_exc += w_syn
    apre = 1
    delta_w_dep = plasticity * -A_minus_u2v  * (1 - apost) * int(apost > theta_stdp_u2v)
    w_syn = clip(w_syn + delta_w_dep, w_min,  w_max)
'''

on_post_synapse_u2v = '''
    apost = 1
    delta_w_pot = plasticity * A_plus_u2v  * (1  + w_offset - apre - w_syn)  * int (apre > theta_stdp_u2v)
    w_syn = clip(w_syn + delta_w_pot, w_min,  w_max)
'''

# v2v synapses, lateral connections
model_synapse_v2v = '''
dw_syn/dt = (-max_inh - w_syn) / tau_homeo_w_syn_v2v : 1 (event-driven)
'''
#w_syn = clip(w_syn - get_decay_inh(t), min_inh, max_inh) : 1
on_pre_synapse_v2v = '''
s_lat += w_syn
'''
