import sys

project_root = '../'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Data.MLS_data import Modulating_Laplacian_Sources


class Data:
    def __init__(self, n_steps=400, n_size=4, data_type='MLS'):
        if data_type == 'MLS':
            self.data = Modulating_Laplacian_Sources(n_steps, n_size, is_tensor=True)

        else:
            ValueError
