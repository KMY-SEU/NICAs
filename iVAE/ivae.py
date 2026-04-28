import torch
import torch.nn as nn


class Unflatten(nn.Module):
    def __init__(self, ModelDict):
        super(Unflatten, self).__init__()
        self.ModelDict = ModelDict

    def forward(self, input_tensor):
        First_no_channels = self.ModelDict["channels"][0]

        input_tensor = input_tensor.view(-1, First_no_channels, int(input_tensor.size(1) / First_no_channels))

        return input_tensor


class Flatten(nn.Module):  # Same name as tensorflow tf.keras.Flatten()
    def __init__(self, DisDict):
        super(Flatten, self).__init__()
        self.DisDict = DisDict

    def forward(self, input_tensor):
        input_tensor = input_tensor.view(input_tensor.size(0), -1)

        return input_tensor


class Encoder(nn.Module):
    def __init__(self, latent_size, Usize, data_size, encode_dict):
        super(Encoder, self).__init__()

        self.latent_size = latent_size
        self.Usize = Usize
        self.data_size = data_size
        self.encode_dict = encode_dict
        self.activation = nn.LeakyReLU(0.1)
        self.var_activation = nn.Softplus()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Check if it is a standard VAE through Usize
        if self.Usize == 0:
            self.standard_flag = True

        else:
            self.standard_flag = False

        self.layers = []  # Initialise layers

        if self.encode_dict["conv_flag"]:

            for i in range(len(self.encode_dict["channels"]) - 1):
                # append the layer
                self.layers.append(nn.Conv1d(in_channels=self.encode_dict["channels"][i],
                                             out_channels=self.encode_dict["channels"][i + 1],
                                             kernel_size=self.encode_dict["kernel_size"][i],
                                             stride=self.encode_dict["stride"][i],
                                             padding=self.encode_dict["padding"][i]))
                # append the activation function
                self.layers.append(self.activation)

            # append the transform to take the nn.linear to a convolutional layer
            self.layers.append(Flatten(self.encode_dict))

        for i in range(len(self.encode_dict["ff_layers"]) - 2):
            # append the layer
            self.layers.append(nn.Linear(in_features=self.encode_dict["ff_layers"][i],
                                         out_features=self.encode_dict["ff_layers"][i + 1], bias=True))
            # append the activation function
            self.layers.append(self.activation)

        self.layers.pop(-1)
        self.encode_net = nn.Sequential(*self.layers)
        self.mu_layer = nn.Linear(self.encode_dict["ff_layers"][-2], self.encode_dict["ff_layers"][-1])
        self.var_layer = nn.Sequential(nn.Linear(self.encode_dict["ff_layers"][-2], self.encode_dict["ff_layers"][-1]),
                                       self.var_activation)

        self.encode_net.apply(self.init_weights)
        self.mu_layer.apply(self.init_weights)
        self.var_layer.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            # m.bias.data.fill_(0.01)

    def one_hot_encode(self, labels):

        with torch.no_grad():
            label_mat = torch.zeros(labels.size(0), self.Usize)

            label_mat[range(labels.size(0)), labels] = 1

            return label_mat

    def forward(self, x, labels=None, cont_input=None):

        # Takes in both the x and u variables
        # Just so that you don't forget, you coded this so that it takes both inputs in, combines them and then spits out the output
        # Always stack as [continuous, discrete]

        if cont_input is not None and not self.standard_flag:
            x_input = torch.hstack((x, cont_input))

        if labels is not None and not self.standard_flag:
            u_input = self.one_hot_encode(labels).to(self.device)
            x_input = torch.hstack((x, u_input))

        else:
            x_input = x

        encode = self.encode_net(x_input)
        mu_z = self.mu_layer(encode)
        var_z = self.var_layer(encode)

        return mu_z, var_z


class Decoder(nn.Module):
    def __init__(self, latent_size, Usize, data_size, decode_dict, var_flag=False):
        super(Decoder, self).__init__()

        self.latent_size = latent_size
        self.Usize = Usize
        self.data_size = data_size
        self.decode_dict = decode_dict
        self.var_flag = var_flag
        self.activation = nn.LeakyReLU(0.1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.layers = []  # Initialise layers

        for i in range(len(self.decode_dict["ff_layers"]) - 2):
            # append the layer
            self.layers.append(nn.Linear(in_features=self.decode_dict["ff_layers"][i],
                                         out_features=self.decode_dict["ff_layers"][i + 1], bias=True))
            # append the activation function
            self.layers.append(self.activation)

        if self.decode_dict["conv_flag"]:
            # append the transform to take the nn.linear to a convolutional layer
            self.layers.append(Unflatten(self.decode_dict))

            for i in range(len(self.decode_dict["channels"]) - 1):
                # append the layer
                self.layers.append(nn.ConvTranspose1d(in_channels=self.decode_dict["channels"][i],
                                                      out_channels=self.decode_dict["channels"][i + 1],
                                                      kernel_size=self.decode_dict["kernel_size"][i],
                                                      stride=self.decode_dict["stride"][i],
                                                      padding=self.decode_dict["padding"][i]))
                # append the activation function
                self.layers.append(self.activation)

        self.layers.pop(-1)  # remove the final activation for linear outputs

        self.decode_net = nn.Sequential(*self.layers)
        self.gen_layer = nn.Linear(self.decode_dict["ff_layers"][-2], self.decode_dict["ff_layers"][-1])

        self.decode_net.apply(self.init_weights)
        self.gen_layer.apply(self.init_weights)

        if self.var_flag:
            self.var_layer = nn.Sequential(
                nn.Linear(self.decode_dict["ff_layers"][-2], self.decode_dict["ff_layers"][-1]), nn.Softplus())
            self.var_layer.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            # m.bias.data.fill_(0.01)

    @staticmethod
    def reparametrisation_trick(mu_data, var_data):
        with torch.no_grad():
            eta = torch.randn_like(mu_data)

        return mu_data + eta * torch.sqrt(var_data)

    def forward(self, mu_latent, var_latent):

        z_latent = self.reparametrisation_trick(mu_latent, var_latent)

        decode_out = self.decode_net(z_latent)

        x_out = self.gen_layer(decode_out)

        if self.var_flag:
            var_out = self.var_layer(decode_out)

        else:
            var_out = torch.ones_like(x_out).requires_grad_(False)

        return x_out, var_out


class ConditionalPrior(nn.Module):
    # Eq.~(7)
    # Can adapt to have parametric densities... (only a mean and variance parameter depending on the class)
    def __init__(self, latent_size, Usize, data_size, prior_dict, continuous_prior=True):
        super(ConditionalPrior, self).__init__()

        self.latent_size = latent_size
        self.Usize = Usize
        self.data_size = data_size
        self.prior_dict = prior_dict
        self.continuous_prior = continuous_prior

        self.activation = nn.LeakyReLU(0.1)
        self.var_activation = nn.Softplus()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Check if it is a standard VAE, if so, set continuous_prior to False and then set distribution to N(0, I)
        if self.Usize == 0:
            self.continuous_prior = False

        self.layers = []  # Initialise layers

        if self.continuous_prior:
            # Define model - essentially another generator but with only FF layers, by design

            for i in range(len(self.prior_dict["ff_layers"]) - 2):
                # append the layer
                self.layers.append(nn.Linear(in_features=self.prior_dict["ff_layers"][i],
                                             out_features=self.prior_dict["ff_layers"][i + 1], bias=True))
                # append the activation function
                self.layers.append(self.activation)

            self.layers.pop(-1)
            self.prior_net = nn.Sequential(*self.layers)
            self.prior_mu = nn.Linear(self.prior_dict["ff_layers"][-2], self.prior_dict["ff_layers"][-1])
            self.prior_var = nn.Linear(self.prior_dict["ff_layers"][-2], self.prior_dict["ff_layers"][-1])

            self.prior_net.apply(self.init_weights)
            self.prior_mu.apply(self.init_weights)
            self.prior_var.apply(self.init_weights)

        else:
            # Lambda functions that just return the mean and variance parameters at all the class locations of interest!

            self.prior_net = lambda U: U

            if self.Usize == 0:
                # self._prior_mu_ = nn.parameter.Parameter(torch.Tensor(1, self.latent_size))
                # self._prior_var_ = nn.parameter.Parameter(torch.Tensor(1, self.latent_size))
                self.register_parameter(name='_prior_mu_', param=torch.nn.Parameter(torch.Tensor(1, self.latent_size)))
                self.register_parameter(name='_prior_var_', param=torch.nn.Parameter(torch.Tensor(1, self.latent_size)))

            else:
                self.register_parameter(name='_prior_mu_',
                                        param=torch.nn.Parameter(torch.Tensor(self.Usize, self.latent_size)))
                self.register_parameter(name='_prior_var_',
                                        param=torch.nn.Parameter(torch.Tensor(self.Usize, self.latent_size)))
                # self._prior_mu_ = nn.parameter.Parameter(torch.Tensor(self.Usize, self.latent_size))
                # self._prior_var_ = nn.parameter.Parameter(torch.Tensor(self.Usize, self.latent_size))#torch.ones(self.Usize, self.latent_size).to(self.device)#

                self.prior_mu = lambda U: self._prior_mu_[U, :]
                self.prior_var = lambda U: self._prior_var_[U, :]

            with torch.no_grad():  # initialise parameters
                if self.Usize == 0:
                    # Set to N(0, I)
                    self._prior_mu_.fill_(0)
                    self._prior_var_.fill_(1)
                    # Turn off gradient flag
                    self._prior_mu_.requires_grad_(False)
                    self._prior_var_.requires_grad_(False)

                else:
                    self._prior_mu_.normal_(0, 0.1)
                    self._prior_var_.normal_(0, 0.1)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            # m.bias.data.fill_(0.01)

    def one_hot_encode(self, labels):

        with torch.no_grad():
            label_mat = torch.zeros(labels.size(0), self.Usize)
            label_mat[range(labels.size(0)), labels] = 1

            return label_mat

    def forward(self, labels=None, cont_input=None):
        # Always stack as [continuous, discrete]
        if self.Usize == 0:
            return self._prior_mu_, self._prior_var_

        else:
            if self.continuous_prior:
                u_input = self.one_hot_encode(labels)

            else:
                u_input = labels

            if cont_input is not None:
                u_input = torch.hstack((cont_input, u_input))

            prior_net = self.prior_net(u_input)
            mu = self.prior_mu(prior_net)
            var = self.var_activation(self.prior_var(prior_net))

            return mu, var


class VAE_model(nn.Module):
    def __init__(self, input_size, latent_size, U_size=None, EncodeDict=None, DecodeDict=None, PriorDict=None,
                 var_decode=False, continuous_prior=True):
        super(VAE_model, self).__init__()

        self.input_size = input_size
        self.latent_size = latent_size
        self.U_size = U_size
        self.encode_dict = EncodeDict
        self.decode_dict = DecodeDict
        self.prior_dict = PriorDict
        self.var_decode = var_decode
        self.continuous_prior = continuous_prior

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model_HI_names = ["HI_1"]

        self.encoder = Encoder(self.latent_size, self.U_size, self.input_size, self.encode_dict)
        self.decoder = Decoder(self.latent_size, self.U_size, self.input_size, self.decode_dict, var_flag=var_decode)
        self.prior = ConditionalPrior(self.latent_size, self.U_size, self.input_size, self.prior_dict,
                                      continuous_prior=self.continuous_prior)

        # print("\n\nVisualising the models\n¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯\n")
        # print(self.encoder)
        # print(self.decoder)
        # print(self.prior)

        if self.U_size == 0:
            print("\nInitialising a normal VAE!\n")
            self.standard_flag = True

        else:
            self.standard_flag = False

    def train(self):
        self.encoder.train()
        self.decoder.train()
        self.prior.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.prior.eval()

    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)
        self.prior.to(device)

    def one_hot_encode(self, labels):

        with torch.no_grad():
            label_mat = torch.zeros(labels.size(0), self.Usize)
            label_mat[range(labels.size(0)), labels] = 1

            return label_mat

    def compute_HIs(self, x, labels=None,
                    cont_input=None):  # Only useful if you are performing anomaly detection (specific to another project)
        with torch.no_grad():
            x_input = x

            if cont_input is not None:
                x_input = torch.hstack((x_input, cont_input))

            if labels is not None:
                x_input = torch.hstack((x_input, self.one_hot_encode(labels)))

            mu_latent, var_latent = self.encoder(x_input)

            x_recon1, var_decoder = self.decoder(mu_latent, var_latent)
            HI_1 = (1 / x_input.shape[1]) * torch.sum((x - x_recon1) ** 2 / (var_decoder), dim=1)

            return HI_1, mu_latent


class VAE_optimiser(object):
    def __init__(self, model, Params):
        print("Learning rate is just 1e-3. Need to feed in correct parameters.")
        ls = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.prior.parameters())
        self.VAE_opt = torch.optim.Adam(ls, lr=1e-3)

    def step(self):
        self.VAE_opt.step()

    def zero_grad(self):
        self.VAE_opt.zero_grad()
