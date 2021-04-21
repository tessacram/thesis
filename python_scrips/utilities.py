import torch
from os import system
from .models import MLP


class GenerateALotCFs:

    def __init__(self, exp):

        self.exp = exp

    def generate(self, data, n_cfs = 5):

        n_instances = data.shape[0]

        cfs = torch.zeros((n_instances, n_cfs, data.shape[1]))

        for n_instance in range(n_instances):
            _ = system('clear')
            print('instance ' + str(n_instance + 1) + '/' + str(n_instances))
            print('percentage: ' + str(round(n_instance * 100 / n_instances, 2)) + '%')

            x = data[n_instance]
            cfs[n_instance] = self.exp.generate_cfs(x, total_cfs=n_cfs)

            if n_instance % 100 == 0:
                torch.save(cfs, 'backup_rename_this_bitch.pt')

        _ = system('clear')
        print('Done!')

        return cfs


class ClassifierTraining:

    def __init__(self, d, hidden_dim = 100):

        self.x_train = d.data_torch_train
        self.x_test = d.data_torch_test
        self.y_train = d.target_torch_train
        self.y_test = d.target_torch_test

        self.input_dim = len(self.x_train[0])
        self.hidden_dim = hidden_dim

    def train(self):

        mlp_model = MLP(self.input_dim, self.hidden_dim)
        criterion = torch.nn.BCELoss()  # BCE = binary cross entropy - our targets are binary
        optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

        ### EVAL ###
        mlp_model.eval()  # here sets the PyTorch module to evaluation mode.
        y_train_hat = mlp_model(self.x_train)
        before_train = criterion(y_train_hat.squeeze(), self.y_train)
        print('Test loss before training', before_train.item())

        ### TRAIN ###
        mlp_model.train()  # here sets the PyTorch module to train mode.
        tot_epoch = 501
        for epoch in range(tot_epoch):
            optimizer.zero_grad()
            # Forward pass
            y_train_hat = mlp_model(self.x_train)
            # Compute Loss
            loss = criterion(y_train_hat.squeeze(), self.y_train)

            if epoch % 100 == 0:
                y_test_hat = mlp_model(self.x_test)
                print('Epoch: {} -- train loss: {} -- accuracy (test set): {}'.format(epoch, round(loss.item(), 3),
                                                                                      mlp_model.accuracy(y_test_hat,
                                                                                                         self.y_test)))

            # Backward pass
            loss.backward()
            optimizer.step()

        ### EVAL ###
        mlp_model.eval()
        y_test_hat = mlp_model(self.x_test)
        after_train = criterion(y_test_hat.squeeze(), self.y_test)
        print('Test loss after Training', after_train.item())

        print('The accuracy scrore on the test set:', mlp_model.accuracy(y_test_hat, self.y_test))
