# equation u layer neurons
u_layer_neuron_equ = '''
    I_ext = in_stimuli(t,i) : 1
    dv/dt = (-v + I_ext) / tau_m_u: 1 (unless refractory)
    '''

reset_u_layer = 'v = theta_reset_u'

# equation v layer neurons
v_layer_neuron_equ = '''
    # Afferent excitatory connections (from u layer to v layer)
    ds_aff_exc/dt = (-s_aff_exc)/tau_f_aff_v_exc: 1
    # Lateral excitatory connections (from v layer to v layer)
    ds_lat/dt = (-s_lat)/tau_f_lat_v: 1
    # membrane potential of v layer
    dv/dt = (-v + s_aff_exc + s_lat) / tau_m_v: 1 (unless refractory)
    '''

reset_v_layer = 'v = theta_reset_v'