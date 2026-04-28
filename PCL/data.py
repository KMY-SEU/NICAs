import numpy as np
import torch

from Data.MLS_data import Modulating_Laplacian_Sources


class Data:
    def __init__(self, n_steps=400, n_size=4, data_type='MLS'):

        # generate data
        if data_type == 'MLS':
            self.data = Modulating_Laplacian_Sources(n_steps, n_size, is_tensor=True)

        else:
            raise ValueError

        # re-direct
        self.no_samples = n_steps
        self.n = n_size
        self.batch_cnt = 0

        # make real and fake samples
        self.samples = self.data.X
        self.mixed_samples = self.data.Y

        # Make datasets
        self.Yreal = self.make_real()
        self.Yfake, self.Yfake_indices = self.make_fake()

        # Make test/train datasets
        self.Yreal_train, self.Yreal_test, self.Yfake_train, self.Yfake_test = self.train_test_split()

    def make_real(self):
        list_samples = []

        for i in range(self.no_samples - 1):
            list_samples.append((self.mixed_samples[i + 1, :], self.mixed_samples[i, :], 1))

        return list_samples

    def make_fake(self):  # Question, do the fake samples generate all the time or is it dynamic?
        # assuming static fake (maybe why they perform L2 regularisation)
        # Takes in the samples, returns the Yreal tuple of the form (Y1, Y2_fake, 0)
        list_samples = []
        fake_index = np.random.randint(0, self.no_samples, self.no_samples)

        for cnt, i in enumerate(fake_index, start=1):
            if i == cnt or abs(i - cnt) < 100:

                while fake_index[cnt] == cnt or abs(
                        fake_index[cnt] - cnt) < 100:  # while loop to ensure that you replace it with something else
                    fake_index[cnt] = np.random.randint(0, self.no_samples, 1)  # replace

        for i in range(self.no_samples - 1):
            list_samples.append((self.mixed_samples[i + 1, :], self.mixed_samples[fake_index[i], :], 0))

        return list_samples, fake_index

    def train_test_split(self):  # 80-20 split
        N = len(self.Yreal)  # Same length for Yfake

        rand_indices = list(range(N))
        np.random.seed(0)
        np.random.shuffle(rand_indices)

        self.no_train_samples = int(0.8 * N)

        self.rand_indices = rand_indices

        Yr_train = [self.Yreal[i] for i in self.rand_indices[:self.no_train_samples]]
        Yr_test = [self.Yreal[i] for i in self.rand_indices[self.no_train_samples:]]

        Yf_train = [self.Yfake[i] for i in self.rand_indices[:self.no_train_samples]]
        Yf_test = [self.Yfake[i] for i in self.rand_indices[self.no_train_samples:]]

        return Yr_train, Yr_test, Yf_train, Yf_test

    def batch_sample(self, batch_size, train_batch=True, include_end=False):

        actual_batch_size = int(batch_size / 2)

        if train_batch:
            Dreal = self.Yreal_train
            Dfake = self.Yfake_train

        else:  # use the test data
            Dreal = self.Yreal_test
            Dfake = self.Yfake_test

        N = len(Dreal)

        if self.batch_cnt > (N - 1) // actual_batch_size:  # check that you are not further than you can go
            self.batch_cnt = 0  # reset counter

            return None, None, None, False

        elif self.batch_cnt == (
                N - 1) // actual_batch_size and not include_end:  # end check no. 1 (for not including end)
            self.batch_cnt = 0  # reset counter

            return None, None, None, False

        elif self.batch_cnt == (
                N - 1) // actual_batch_size and include_end:  # end check no. 2 (for including end) - set the flag to true and then run with it
            flag = True
            start = self.batch_cnt * actual_batch_size
            end = min((self.batch_cnt + 1) * actual_batch_size, N)

        else:
            flag = True
            start = self.batch_cnt * actual_batch_size
            end = min((self.batch_cnt + 1) * actual_batch_size, N)

        sample_indices = np.arange(start, end, 1)

        Y1 = torch.zeros(2 * int(end - start), self.n)  # Double because start:end is half the batch size
        Y2 = torch.zeros(2 * int(end - start), self.n)
        labels = torch.zeros(2 * int(end - start))

        cnt = 0

        for i in sample_indices:
            Y1[cnt, :] = Dreal[i][0]
            Y2[cnt, :] = Dreal[i][1]
            labels[cnt] = Dreal[i][2]
            cnt += 1

        for i in sample_indices:  # Same indices are used from the fake case
            Y1[cnt, :] = Dfake[i][0]
            Y2[cnt, :] = Dfake[i][1]
            labels[cnt] = Dfake[i][2]
            cnt += 1

        self.batch_cnt += 1

        return Y1, Y2, labels, flag

    def random_sample(self, batch_size, train_batch=True):

        # Assuming equal batch sizes
        # Need to return the Y1, Y1 and the labels
        # 1 big thing, so both real and fake at the same time
        # Basically just randomly sample the indices, stack the x(t)s and x(t-1)s and the indices, return as a tuple

        if train_batch:
            Dreal = self.Yreal_train
            Dfake = self.Yfake_train

            rand_sample_real = np.random.randint(0, self.no_train_samples, batch_size // 2)
            rand_sample_fake = np.random.randint(0, self.no_train_samples, int(batch_size - batch_size // 2))

        else:  # use the test data
            Dreal = self.Yreal_test
            Dfake = self.Yfake_test

            rand_sample_real = np.random.randint(0, self.no_samples - self.no_train_samples - 1, batch_size // 2)
            rand_sample_fake = np.random.randint(0, self.no_samples - self.no_train_samples - 1,
                                                 int(batch_size - batch_size // 2))

        # Not the best, but a good start
        # Loop through the indices and extract everything

        Y1 = torch.zeros(batch_size, self.n)
        Y2 = torch.zeros(batch_size, self.n)
        labels = torch.zeros(batch_size)
        cnt = 0

        for i in rand_sample_real:
            Y1[cnt, :] = Dreal[i][0]
            Y2[cnt, :] = Dreal[i][1]
            labels[cnt] = Dreal[i][2]
            cnt += 1

        for i in rand_sample_fake:
            Y1[cnt, :] = Dfake[i][0]
            Y2[cnt, :] = Dfake[i][1]
            labels[cnt] = Dfake[i][2]
            cnt += 1

        return Y1, Y2, labels
