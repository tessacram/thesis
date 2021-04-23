import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder


class Data:

    def __init__(self, data_df, cont_indices, frac_train):

        # COLLECT ESSENTIAL INFORMATION
        self.target_name = data_df.columns[-1]
        self.data_df = data_df.drop(self.target_name, axis=1)
        self.target_df = data_df[self.target_name]
        self.cont_names = self.data_df.columns[cont_indices]

        self.min = self.data_df[self.cont_names].min()
        self.max = self.data_df[self.cont_names].max()

        # MOVE CONTINUE COLUMNS TO THE END
        columns_order = np.hstack((self.data_df.columns.difference(self.cont_names), self.cont_names))
        self.data_df = self.data_df.reindex(columns=columns_order)
        self.cont_indices = np.arange(len(cont_indices) * -1, 0)
        self.column_names = self.data_df.columns

        # CREATE THE ENCODERS
        self.enc = OneHotEncoder(sparse=False)
        self.enc.fit(self.data_df.drop(self.cont_names, axis=1).values)
        self.enc_length = self.data_df.drop(self.cont_names, axis=1).nunique()

        # CREATE TRAIN AND TEST SET
        split = int(len(self.data_df) * frac_train)
        self.data_df_train = self.data_df.iloc[:split, :]
        self.data_df_test = self.data_df.iloc[split:, :]
        self.target_df_train = self.target_df.iloc[:split]
        self.target_df_test = self.target_df.iloc[split:]
        self.len_train = len(self.target_df_train.index)
        self.len_test = len(self.target_df_test.index)

        # CREATE TORCH FORMAT
        self.data_torch_train = self.df_to_torch(self.data_df_train)
        self.data_torch_test = self.df_to_torch(self.data_df_test)
        self.target_torch_train = torch.from_numpy(self.target_df_train.values.astype(np.float32))
        self.target_torch_test = torch.from_numpy(self.target_df_test.values.astype(np.float32))

        # SAVE MADS
        self.mads = np.concatenate((np.ones(sum(self.enc_length.values)), self.get_MAD().values))

    def augmentation(self, x, y):

        self.len_train = self.len_train + len(y)
        with torch.no_grad():
            self.data_torch_train = torch.cat((self.data_torch_train, x), 0)
            self.target_torch_train = torch.cat((self.target_torch_train, y))
            self.data_df_train = self.torch_to_df(self.data_torch_train)
            self.target_df_train = pd.DataFrame(self.target_torch_train.numpy(), columns=['target'])

    def df_to_torch(self, df):
        """ from dataframe to one-hot-encoded torch vector(s) """
        df = df.copy()
        df = self.normalize(df)
        data_np = df.values
        one_hot_part = self.enc.transform(np.delete(data_np, self.cont_indices, axis=1))
        data_np = np.concatenate((one_hot_part, data_np[:, self.cont_indices]), axis=1)
        return torch.from_numpy(data_np.astype(np.float32))

    def torch_to_df(self, data_torch):
        """ from one-hot-encoded torch vector(s) to decoded dataframe """
        data_np_enc = data_torch.detach().numpy()
        cat_vars_dec = self.enc.inverse_transform(np.delete(data_np_enc, self.cont_indices, axis=1))
        data_np_dec = np.concatenate((cat_vars_dec, data_np_enc[:, self.cont_indices]), axis=1)
        df = pd.DataFrame(data=data_np_dec, columns=self.column_names)
        return self.denormalize(df)

    def normalize(self, df):
        """ normalizes the continues variables between 0 and 1 """
        df = df.copy()
        for var in self.cont_names:
            df[var] = (df[var] - self.min[var]) / (self.max[var] - self.min[var])
        return df

    def denormalize(self, df):
        """ denormalizes the continues variables to their original value """
        df = df.copy()
        for var in self.cont_names:
            df[var] = (df[var] * (self.max[var] - self.min[var])) + self.min[var]
        return df

    def arg_max(self, cfs):
        """ seeks the maximum values for the one-hot-encoded features and returns cf in one-hot notation """
        cfs = cfs.clone()

        index = 0
        n_vars = len(self.data_df.columns)
        n_columns = len(cfs[0])
        for var in range(n_vars):  # loop over all variables
            if index in self.cont_indices + n_columns:
                index += 1
                continue

            part = cfs[:, index:index + self.enc_length[var]]
            arg_max = torch.argmax(part, dim=1)
            cfs[:, index:index + self.enc_length[var]] = 0
            cfs[np.arange(len(cfs)), index+arg_max] = 1

            index += self.enc_length[var]

        return cfs

    def get_MAD(self):
        """ returns the MAD's of the continuous variables """
        return self.data_df[self.cont_names].mad()


class Income(Data):

    def __init__(self, frac_train=0.75, total_instances=None):

        self.data_df = self.get_df()
        if total_instances is not None:
            self.data_df = self.data_df.iloc[0:total_instances, :]
        self.cont_indices = [0, 7]
        super().__init__(self.data_df, self.cont_indices, frac_train=frac_train)

    @staticmethod
    def get_df():
        raw_data = np.genfromtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                                 delimiter=', ', dtype=str)

        #  column names from "https://archive.ics.uci.edu/ml/datasets/Adult"
        column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation',
                        'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week',
                        'native-country', 'income']

        adult_data = pd.DataFrame(raw_data, columns=column_names)

        # For more details on how the below transformations are made, please refer to https://rpubs.com/H_Zhu/235617
        adult_data = adult_data.astype({"age": np.int64, "educational-num": np.int64, "hours-per-week": np.int64})

        adult_data = adult_data.replace(
            {'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
        adult_data = adult_data.replace(
            {'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government', 'Local-gov': 'Government'}})
        adult_data = adult_data.replace(
            {'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
        adult_data = adult_data.replace(
            {'workclass': {'Never-worked': 'Self-Employed', 'Without-pay': 'Self-Employed'}})
        adult_data = adult_data.replace({'workclass': {'?': 'Other/Unknown'}})

        adult_data = adult_data.replace({'occupation': {'Adm-clerical': 'White-Collar', 'Craft-repair': 'Blue-Collar',
                                                        'Exec-managerial': 'White-Collar',
                                                        'Farming-fishing': 'Blue-Collar',
                                                        'Handlers-cleaners': 'Blue-Collar',
                                                        'Machine-op-inspct': 'Blue-Collar', 'Other-service': 'Service',
                                                        'Priv-house-serv': 'Service',
                                                        'Prof-specialty': 'Professional', 'Protective-serv': 'Service',
                                                        'Tech-support': 'Service',
                                                        'Transport-moving': 'Blue-Collar', 'Unknown': 'Other/Unknown',
                                                        'Armed-Forces': 'Other/Unknown', '?': 'Other/Unknown'}})

        adult_data = adult_data.replace({'marital-status': {'Married-civ-spouse': 'Married',
                                                            'Married-AF-spouse': 'Married',
                                                            'Married-spouse-absent': 'Married',
                                                            'Never-married': 'Single'}})

        adult_data = adult_data.replace({'race': {'Black': 'Other', 'Asian-Pac-Islander': 'Other',
                                                  'Amer-Indian-Eskimo': 'Other'}})

        adult_data = adult_data[['age', 'workclass', 'education', 'marital-status', 'occupation', 'race', 'gender',
                                 'hours-per-week', 'income']]

        adult_data = adult_data.replace({'income': {'<=50K': 0, '>50K': 1}})

        adult_data = adult_data.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                                       '11th': 'School', '10th': 'School', '7th-8th': 'School',
                                                       '9th': 'School',
                                                       '12th': 'School', '5th-6th': 'School', '1st-4th': 'School',
                                                       'Preschool': 'School'}})

        adult_data = adult_data.rename(columns={'marital-status': 'marital_status', 'hours-per-week': 'hours_per_week'})

        adult_data = adult_data.sample(frac=1, random_state=1).reset_index(drop=True)

        return adult_data


class Compas(Data):

    def __init__(self, frac_train=0.75):

        self.data_df = pd.read_csv('compass_data.csv')
        self.data_df = self.data_df.sample(frac=1, random_state=1).reset_index(drop=True)
        self.cont_indices = [1, 3]
        super().__init__(self.data_df, self.cont_indices, frac_train=frac_train)


class Import(Data):

    def __init__(self, data_df, cont_indices, frac_train=0.75):

        self.data_df = data_df
        self.data_df = self.data_df.sample(frac=1, random_state=1).reset_index(drop=True)
        self.cont_indices = cont_indices
        super().__init__(self.data_df, self.cont_indices, frac_train=frac_train)
