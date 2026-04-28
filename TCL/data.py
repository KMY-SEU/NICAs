import numpy as np

from Data.MLS_data import Modulating_Laplacian_Sources


class Data:
    def __init__(self, n_steps=400, n_size=4, data_type='MLS'):
        if data_type == 'MLS':
            self.data = Modulating_Laplacian_Sources(n_steps, n_size, is_tensor=False)
            self.labels = np.arange(0, n_steps)
            self.num_segment = n_steps

        else:
            raise ValueError
