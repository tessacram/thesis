import torch
import random
from .models import MLP
from IPython.display import clear_output




class CFUtilities:

    def __init__(self, exp=None, d=None):

        self.exp = exp
        self.d = d

    def generate_cfs(self, data, exp=None, n_cfs=5):

        if exp is not None:
            self.exp = exp

        if self.exp is None:
            print("Please give me an explanor")
            return None

        self.d = self.exp.data

        if data == 'train':
            instances = self.d.data_torch_train
            n_instances = instances.shape[0]

        elif data == 'test':
            instances = self.d.data_torch_test
            n_instances = instances.shape[0]

        else:
            print("data is either train or test")
            return None

        cfs = torch.zeros((n_instances, n_cfs, instances.shape[1]))

        for n_instance in range(n_instances):
            # clear_output(wait=True)
            # print('instance ' + str(n_instance + 1) + '/' + str(n_instances))
            # print('percentage: ' + str(round(n_instance * 100 / n_instances, 2)) + '%')

            x = instances[n_instance]
            cfs[n_instance] = self.exp.generate_cfs(x, total_cfs=n_cfs)

            if n_instance % 100 == 0:
                torch.save(cfs, 'backup_cfs.pt')
        # clear_output(wait=True)
        print('Done!')

        return cfs

    def data_augmentation(self, f_fair, exp=None, n_cfs=3):

        if exp is not None:
            self.exp = exp

        if self.exp is None:
            print("Please give me an explanor")
            return None

        self.d = self.exp.data
        instances = self.d.data_torch_train

        n_instances = instances.shape[0]

        extra_datapoints = torch.zeros((n_instances, instances.shape[1]))
        targets = torch.zeros(n_instances)

        for n_instance in range(n_instances):
            # clear_output(wait=True)
            # print('instance ' + str(n_instance + 1) + '/' + str(n_instances))
            # print('percentage: ' + str(round(n_instance * 100 / n_instances, 2)) + '%')

            x = instances[n_instance]
            points = self.exp.generate_cfs(x, total_cfs=n_cfs, f_fair=f_fair)

            extra_datapoints[n_instance] = points[random.randint(0, 2)]
            targets[n_instance] = abs(self.d.target_torch_train[n_instance] - 1)

            if n_instance % 100 == 0:
                torch.save(extra_datapoints, 'backup_data_augmentation.pt')

        # _ = system('clear')
        # clear_output(wait=True)
        print('Done!')

        return extra_datapoints, targets

    def collect_feedback(self, cfs, instances=None):

        if instances is None:
            instances = self.d.data_torch_train

        if instances == 'train':
            instances = self.d.data_torch_train
        if instances == 'test':
            instances = self.d.data_torch_test

        cfs_per_instance = cfs.shape[1]
        n_features = cfs.shape[2]
        n_instances = len(instances)
        n_pairs = n_instances * cfs_per_instance

        n_pair = 0
        pairs = torch.zeros((n_pairs, n_features))
        targets = torch.zeros(n_pairs)

        n_fair_cfs = 0

        for i in range(n_instances):

            cfs[i] = self.d.arg_max(cfs[i])
            df = self.d.torch_to_df(cfs[i])

            for j in range(cfs_per_instance):

                # create pairs
                difference = cfs[i][j] - instances[i]
                pairs[n_pair] = difference

                # check (un)fairness
                if df['gender'][j] == self.d.data_df_train['gender'][i] and df['race'][j] == self.d.data_df_train['race'][i]:
                    targets[n_pair] = 0
                    n_fair_cfs += 1
                else:
                    targets[n_pair] = 1

                n_pair += 1

        return pairs, targets, (n_fair_cfs/n_pairs)


class ClassifierTraining:

    def __init__(self, hidden_dim = 100):

        self.hidden_dim = hidden_dim

    def train(self, d=None, x=None, y=None, tot_epoch=501, print_proces=True):

        if d is None:

            n_instances = len(y)
            split = int(0.75 * n_instances)

            x_train = x[:split,:]
            x_test = x[split:,:]
            y_train = y[0:split]
            y_test = y[split:]

        else:
            print("we have detached the things, don't worry!")
            x_train = d.data_torch_train
            x_test = d.data_torch_test
            y_train = d.target_torch_train
            y_test = d.target_torch_test

        input_dim = len(x_train[0])

        mlp_model = MLP(input_dim, self.hidden_dim)
        criterion = torch.nn.BCELoss()  # BCE = binary cross entropy - our targets are binary
        optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

        ### EVAL ###
        mlp_model.eval()  # here sets the PyTorch module to evaluation mode.
        y_train_hat = mlp_model(x_train)
        before_train = criterion(y_train_hat.squeeze(), y_train)
        print('Test loss before training', before_train.item())

        ### TRAIN ###
        mlp_model.train()  # here sets the PyTorch module to train mode.
        for epoch in range(tot_epoch):
            optimizer.zero_grad()
            # Forward pass
            y_train_hat = mlp_model(x_train)
            # Compute Loss
            loss = criterion(y_train_hat.squeeze(), y_train)

            if epoch % 100 == 0 and print_proces is True:
                y_test_hat = mlp_model(x_test)
                print('Epoch: {} -- train loss: {} -- accuracy (test set): {}'.format(epoch, round(loss.item(), 3),
                                                                                      mlp_model.accuracy(y_test_hat,
                                                                                                         y_test)))

            # Backward pass
            loss.backward()
            optimizer.step()

        ### EVAL ###
        mlp_model.eval()
        y_test_hat = mlp_model(x_test)
        after_train = criterion(y_test_hat.squeeze(), y_test)
        print('Test loss after Training', after_train.item())

        print('The accuracy scrore on the test set:', mlp_model.accuracy(y_test_hat, y_test))

        return mlp_model
