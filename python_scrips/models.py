import torch


class MLP(torch.nn.Module):

    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()

        # the sizes
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 2 layers and 2 activation functions
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        # feed forward to the 2 layers (plus activation functions)
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)

        return output

    def accuracy(self, prediction, grand_true):
        prediction = prediction.detach().round().reshape(1, -1)
        n_right = torch.sum(prediction == grand_true).item()
        accuracy_score = n_right / len(grand_true)
        return round(accuracy_score, 3)
