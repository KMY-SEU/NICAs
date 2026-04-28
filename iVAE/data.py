import numpy as np
import torch

from Data.MLS_data import Modulating_Laplacian_Sources
from TCL.configs import batch_size


class Data(object):
    def __init__(self, n_steps=400, n_size=4, data_type='MLS', U_size=5, batch_size=64, random_seed=False):

        # generate data
        if data_type == 'MLS':

            data = Modulating_Laplacian_Sources(n_steps, n_size, is_tensor=True)

            labels = []
            segment_length = n_steps // U_size

            for i in range(U_size):
                labels += [i] * segment_length

            labels += [U_size - 1] * (n_steps - len(labels))
            labels = np.array(labels)

        else:
            raise ValueError

        # re-direct
        self.no_samples = n_steps
        self.n = n_size
        self.batch_size = batch_size
        self.random_seed = random_seed  # If random_seed = True - specifies that a random sample is required and the counter is not increased!

        self.data = data.X
        self.mixed_data = data.Y
        self.sample_labels = labels

        self.data_tuples = list(zip(self.data, self.sample_labels))  # list of tuples
        self.mixed_data_tuples = list(zip(self.mixed_data, self.sample_labels))  # list of tuples

        # shuffle mixed_data
        self.shuffled_data_index = np.arange(0, self.mixed_data.size(0), 1, dtype=int)

        if self.random_seed:
            np.random.shuffle(self.shuffled_data_index)

        # Convert self.sample_labels to torch.tensor
        self.sample_labels = torch.from_numpy(self.sample_labels)

    # turn the class into an iterator
    def __iter__(self):

        self.iter_cnt = 0  # initialises the iterator
        return self  # returns the iterator object

    def __next__(self):

        if not self.random_seed:
            start = self.iter_cnt * self.batch_size
            end = start + self.batch_size

            index = self.shuffled_data_index[start:end]

            if end <= len(self.mixed_data_tuples) + self.batch_size:

                self.iter_cnt += 1

                data = self.mixed_data[index, :]
                labels = self.sample_labels[index]

                return data, labels

            else:
                self.iter_cnt = 0
                raise StopIteration

        else:
            print("Random sampler is not implemented.")
            raise SystemExit
