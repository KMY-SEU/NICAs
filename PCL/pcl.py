import torch
import torch.nn as nn


class Maxout_linear_layer(nn.Module):
    def __init__(self, in_features, out_features, k, bias=True):
        super(Maxout_linear_layer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.maxout_layer = nn.Linear(self.in_features, k * self.out_features, bias=self.bias)
        self.maxout_layer.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            # m.bias.data.fill_(0.01)

    def forward(self, x):
        maxout = self.maxout_layer(x)

        maxout = maxout.reshape(-1, self.out_features, self.k)

        maxout, indices = torch.max(maxout, 2)

        return maxout


class feature_MLP(nn.Module):  # exactly the same as the mixing MLP
    def __init__(self, n_size, n_layers, maxout_flag=False):
        super(feature_MLP, self).__init__()

        self.layers = []
        self.activation = nn.LeakyReLU(negative_slope=0.1)  # nn.LeakyReLU(negative_slope = 0.05) #nn.ReLU()#

        for i in range(n_layers):
            if maxout_flag:
                self.layers.append(Maxout_linear_layer(n_size, n_size, 2))

            else:

                n_hidden = 128

                if i == 0:
                    self.layers.append(nn.Linear(n_size, n_hidden))
                    self.layers.append(self.activation)
                elif i == n_layers - 1:
                    self.layers.append(nn.Linear(n_hidden, n_size))
                else:
                    self.layers.append(nn.Linear(n_hidden, n_hidden))
                    self.layers.append(self.activation)

        if maxout_flag:
            self.layers.pop(0)  # remove the first one, as we only care about Lfeature = Lmixture
            self.layers.append(nn.Linear(n_size, n_size))  # Append a final linear layer

        self.model = nn.Sequential(*self.layers)
        self.model.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            # m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.model(x)


class RegressionModel2(nn.Module):
    def __init__(self, n_size, n_layers=1):  # Takes in H1, H2 and returns a prediction
        super(RegressionModel2, self).__init__()

        self.layers = []
        self.activation = nn.Sigmoid()

        for i in range(n_layers):
            self.layers.append(nn.Linear(n_size * 2, n_size * 2))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(n_size * 2, 1))

        self.model = nn.Sequential(*self.layers)
        self.model.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, h1, h2):

        h_total = torch.hstack((h1, h2))
        out = self.activation(self.model(h_total))

        return out.reshape(-1)
