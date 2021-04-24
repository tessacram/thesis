import torch
import torch.nn.functional as F
import numpy as np
from IPython.display import clear_output


class Dice:

    def __init__(self, data, classifier, device="cpu"):

        self.device = device

        # SAVE DATA INSTANCE AND CLASSIFIER
        self.data = data
        self.classifier = classifier.to(self.device)
        self.classifier.eval()

        # INFORMATION ABOUT THE INPUT
        self.x = self.data.data_torch_train[0].to(self.device)
        self.n_columns = self.data.data_torch_train[0].shape[0]
        self.y = self.classifier(self.x)
        self.y_pref = 1 - self.y
        self.cont_indices = self.data.cont_indices
        self.mads = torch.from_numpy(self.data.mads)

        # INFORMATION ABOUT THE CFS
        self.cfs = []
        self.total_cfs = len(self.cfs)
        self.y_pred_list = None

    def generate_cfs(self, x, total_cfs=3, lr=0.01, max_iterations=1001, distance_weight=0.5,
                     diversity_weight=1, reg_weight=0.1, output='torch', print_progress=False,
                     f_fair=None):
        """ generates counterfactuals
        input:
        x - this torch instance is asking for cfs!
        total_cfs - the amount of cf's to be generated
        lr - learning rate for optimizer
        max_iterations - number of iterations to find cfs
        weights - weights for the loss function
        output:
        couterfactuals in pandas dataframe format """

        self.x = x.to(self.device)
        self.y = self.classifier(self.x)
        self.y_pref = torch.round(1 - self.y.detach())
        self.total_cfs = total_cfs
        self.initialize_cfs()

        optimizer = torch.optim.Adam([self.cfs], lr=lr)

        iteration = 0
        while iteration < max_iterations:

            self.y_pred_list = self.classifier(self.cfs)

            loss = self.compute_loss(distance_weight, diversity_weight, reg_weight, f_fair)

            loss.backward()
            optimizer.step()

            self.cfs.data = torch.clamp(self.cfs, 0, 1)

            if print_progress and iteration % 100 == 0:
                print(round(iteration * 100 / max_iterations, 2), '%  loss:', round(loss.item(), 3))
                # clear_output(wait=True)

            iteration += 1

            optimizer.zero_grad()

        self.cfs = self.data.arg_max(self.cfs)

        self.do_posthoc_sparsity_enhancement()

        if output == 'df':
            self.cfs = self.data.torch_to_df(self.cfs)
            self.cfs['target'] = self.y_pred_list.detach().numpy().round()
            # if abs(sum(self.y_pred_list.detach().numpy().round())/self.total_cfs - self.y_pref.item()) > 0.4:
            #     return 'sorry'

            return self.cfs, torch.round(self.y), self.y_pred_list

        return self.cfs

    def initialize_cfs(self):
        """ initializes the cf's with a small perturbation from x """
        self.cfs = self.x + 0.01 * torch.rand(self.total_cfs, self.n_columns).float().to(self.device)
        self.cfs.requires_grad = True

    def compute_loss(self, distance_weight, diversity_weight, reg_weight, f_fair):
        """ computes the total loss concerning an input x and its cf's """
        if f_fair is None:
            return self.compute_y_loss() + \
                   distance_weight * self.compute_distance_loss() - \
                   diversity_weight * self.compute_diversity_loss() + \
                   reg_weight * self.compute_regularisation_loss()
        else:
            return self.compute_fair_loss(f_fair) + \
                   distance_weight * self.compute_distance_loss() - \
                   diversity_weight * self.compute_diversity_loss() + \
                   reg_weight * self.compute_regularisation_loss()

    def compute_y_loss(self):
        """ a part of the loss function
            the prediction of the cf should be close to the preferred prediction of the cf
             we use 'hinge loss' """
        total_loss = 0
        z = -1 if self.y_pref == 0 else 1
        for y_pred in self.y_pred_list:
            total_loss += F.relu(1 - z * torch.log(y_pred / (1 - y_pred)))
            # total_loss += abs(self.y_pref - y_pred)
        return total_loss / self.total_cfs

    def compute_distance_loss(self):
        """ a part of the loss function
            the distance between de cfs and the input x should be small """
        total_loss = 0
        for cf in self.cfs:
            total_loss += self.compute_distance(self.x, cf)
        return total_loss / self.total_cfs

    def compute_diversity_loss(self):
        """ a part of the loss function
            the distance between the cfs should be large """
        K = torch.ones((self.total_cfs, self.total_cfs))
        for i in range(self.total_cfs):
            for j in range(self.total_cfs):
                K[(i, j)] = 1 / (1 + self.compute_distance(self.cfs[i], self.cfs[j]))
                if i == j:
                    K[(i, j)] += 0.0001

        return torch.det(K)

    def compute_fair_loss(self, f_fair):  # gebruik hinge loss
        # loss = 0
        # for i in range(self.total_cfs):
        #     difference = self.cfs[i] - self.x
        #     loss += f_fair(difference)
        # return loss / self.total_cfs

        loss = 0
        z = 1
        for i in range(self.total_cfs):
            difference = self.cfs[i] - self.x
            fairness_score = f_fair(difference)
            loss += F.relu(1 - z * torch.log(fairness_score / (1 - fairness_score)))
        return loss / self.total_cfs

    def compute_regularisation_loss(self):
        """ computes the 'extra' part of the loss function
            the sum of a one-hot-encoded feature should be close to 1 """
        total_loss = 0
        enc_length = self.data.enc_length
        for cf in self.cfs:
            index = 0
            for feature in range(len(enc_length)):
                total_loss += torch.pow(torch.sum(cf[index:index + enc_length[feature]]) - 1, 2)
                index += enc_length[feature]
        return total_loss / self.total_cfs

    def compute_distance(self, x1, x2):
        """ computes the distance between two vectors using MAD"""
        return torch.sum(torch.mul((torch.abs(x1 - x2)), self.mads.to(self.device)), dim=0)

    def do_posthoc_sparsity_enhancement(self):
        """ Performs a greedy linear search -
            moves the continuous and categorical features in CFs towards original values
            in query_instance greedily until the prediction class changes. """

        with torch.no_grad():

            enc_length = self.data.enc_length
            for cf in self.cfs:
                original_class = torch.round(self.classifier(cf))

                # CAT VARIABLES
                copy = torch.clone(cf)
                index = 0
                for feature in range(len(enc_length)):
                    class_instance = self.x[index:index + enc_length[feature]]
                    # what if we insert de original class??
                    copy[index:index + enc_length[feature]] = class_instance
                    new_prediction = torch.round(self.classifier(copy))
                    if new_prediction != original_class:
                        cf[index:index + enc_length[feature]] = class_instance

                    index += enc_length[feature]

                # CONT VARIABLES
                copy = torch.clone(cf)
                for i in self.cont_indices:
                    prev_value = copy[i]
                    for new_value in np.arange(cf[i].detach().cpu().numpy(), self.x[i].detach().cpu().numpy(), 0.01):
                        copy[i] = new_value
                        new_prediction = torch.round(self.classifier(copy))
                        if new_prediction != original_class:
                            copy[i] = prev_value
                            break
                        prev_value = new_value
                    cf[i] = copy[i]
