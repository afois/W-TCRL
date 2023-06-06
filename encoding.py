import numpy as np

class ReceptiveField:
    # Parameter that is used in standard deviation definition
    gamma = 1.5

    def __init__(self, bank_size=10, I_min=0.05, I_max=0.95, isToric = True):
        self.bank_size = bank_size
        self.field_mu = np.array([(I_min + ((2 * i - 2) / 2) * ((I_max - I_min) / (bank_size - 1)))
                                  for i in range(1, bank_size + 1)])
        self.field_sigma = (1.0 / self.gamma) * (I_max - I_min)
        self.isToric = isToric

    def float_to_membrane_potential(self, input_vector):
        try:
            input_vector = input_vector.reshape((input_vector.shape[0], 1))

        except Exception as exc:
            print("Exception: {0}\nObject shape: {1}".format(repr(exc), input_vector.shape))
            exit(1)
        if (self.isToric):
            tor_dist = self.ToroidalDistance(input_vector)
            res = 0.5*(tor_dist/self.field_sigma)**2
        else :
            res = 0.5*((input_vector - self.field_mu)/self.field_sigma)**2

        return np.exp(-res)


    def ToroidalDistance (self, input_vector) :
        dx = np.abs(self.field_mu - input_vector)
        tor_dist = np.where(dx < 0.5, dx, 1.0 - dx)
        return tor_dist

def dataset_to_input_current (dataset, bank_size = 10, I_min = 0.05, I_max = 0.95):
    nb_vectors = dataset.shape[0]
    data_dim = dataset.shape[-1]

    rf = ReceptiveField(bank_size=bank_size, I_min=I_min, I_max=I_max)

    # null input current by default (null current, then feed input current in array, and so on)
    input_currents = np.zeros(shape = (nb_vectors, 2, data_dim * bank_size))
    for idx, input_vector in enumerate (dataset):
        input_currents[idx, 0] = rf.float_to_membrane_potential(input_vector).flatten()

    return input_currents.reshape(nb_vectors*2, data_dim * bank_size)