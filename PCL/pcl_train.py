from datetime import datetime
import numpy as np

from tqdm import tqdm

from pcl import *


def train(n_size, n_layers, data, batch_size, train_dir, maxout_flag=False):
    """

    Args:
        n_size:
        n_layers:
        data:
        batch_size:
        train_dir:
        maxout_flag:

    Returns:

    """

    #
    f_model = feature_MLP(n_size, n_layers, maxout_flag)
    r_model = RegressionModel2(n_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # torch.device("cpu")#
    print(f_model, r_model)

    f_model.to(device)
    r_model.to(device)

    ls = list(f_model.parameters()) + list(r_model.parameters())

    optim = torch.optim.Adam(ls, lr=1e-3)  # , momentum = 0.25, weight_decay = 0.0001)
    loss = torch.nn.BCELoss()

    cost = []
    epochs = 100000

    for i in range(epochs):
        cost_val = 0
        N = 0
        flag = True

        while flag == True:
            Y1, Y2, l1, flag = data.batch_sample(batch_size, include_end=True)

            if flag:
                H1 = f_model(Y1.to(device))
                H2 = f_model(Y2.to(device))

                out = r_model(H1, H2)
                loss_iter = loss(out, l1.to(device))
                loss_iter.backward()
                optim.step()

                cost_val += loss_iter.item()
                N += 1
                optim.zero_grad()

        cost.append(cost_val / N)

        if i % 100 == 0:
            # Train
            no_samples_test = 8192
            Y1, Y2, l1 = data.random_sample(no_samples_test)

            with torch.no_grad():
                H1 = f_model(Y1.to(device))
                H2 = f_model(Y2.to(device))

                out = r_model(H1, H2)
                out = out.cpu().numpy()

                train_acc = (np.sum(out[0:no_samples_test // 2] > 0.5) + np.sum(
                    out[no_samples_test // 2:] < 0.5)) / no_samples_test

            ## Test
            Y1, Y2, l1 = data.random_sample(no_samples_test, train_batch=False)

            with torch.no_grad():
                H1 = f_model(Y1.to(device))
                H2 = f_model(Y2.to(device))

                out = r_model(H1, H2)
                out = out.cpu().numpy()

                test_acc = (np.sum(out[0:no_samples_test // 2] > 0.5) + np.sum(
                    out[no_samples_test // 2:] < 0.5)) / no_samples_test

            format_str = '%s: epoch %d, cost = %.2f, train_acc = %3.2f, test_acc = %3.2f'
            print(format_str % (datetime.now(), i, cost[-1], train_acc * 100, test_acc * 100))

            # save model
            torch.save(f_model.state_dict(), train_dir + 'model.pth')

            if int(train_acc * 100) == 100:
                cnt_early_stop += 1

                if cnt_early_stop >= 3:
                    print('Early step.')
                    break
            else:
                # reset
                cnt_early_stop = 0
