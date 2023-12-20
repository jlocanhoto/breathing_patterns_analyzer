import os
import csv
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

FLOW_LABEL = "flow"
VOLUME_LABEL = "volume"
EXERCISE_LABEL = "exercise"
UUID_LABEL = "uuid"

COLUMNS_LABELS = [UUID_LABEL, EXERCISE_LABEL, FLOW_LABEL, VOLUME_LABEL]
CSV_DELIMITER = ';'


class Dataset:
    def __init__(self, dataset_file):
        with open(dataset_file) as dset_file:
            self.dataset = pd.read_csv(dset_file, sep=CSV_DELIMITER, names=COLUMNS_LABELS)

    def get_qty_exercises(self):
        return self.dataset[EXERCISE_LABEL].nunique()

    def get_samples_of_exercise(self, exercise_number):
        exercise = 'exerc_' + str(exercise_number)

        return self.dataset.loc[self.dataset[EXERCISE_LABEL] == exercise]

    def get_samples_of_uuid(self, uuid):
        return self.dataset.loc[self.dataset[UUID_LABEL] == uuid]
    
    def split_train_test_uuid(self, uuid):
        X_train = pd.DataFrame(columns=[FLOW_LABEL, VOLUME_LABEL])
        X_test = pd.DataFrame(columns=[FLOW_LABEL, VOLUME_LABEL])
        y_train = pd.Series()
        y_test = pd.Series()

        selected_indexes = [1, 2, 2, 2, 1, 1, 3]

        uuid_curves = self.dataset.loc[self.dataset[UUID_LABEL] == uuid]
        non_uuid_curves = self.dataset.loc[self.dataset[UUID_LABEL] != uuid]

        idx = random.randint(0, 2)

        for i in range(1, self.get_qty_exercises()):
            exercise = 'exerc_' + str(i)
            exercises = uuid_curves.loc[uuid_curves[EXERCISE_LABEL] == exercise]
            X, y = exercises[[FLOW_LABEL, VOLUME_LABEL]], exercises[EXERCISE_LABEL]

            j = selected_indexes[i-1]
            X_train = X_train.append(X.iloc[[j]])
            y_train = y_train.append(y.iloc[[j]])

            for k in range(len(X)):
                if k != j:
                    X_test = X_test.append(X.iloc[[k]])
                    y_test = y_test.append(y.iloc[[k]])
            
            exercises = non_uuid_curves.loc[non_uuid_curves[EXERCISE_LABEL] == exercise]
            X, y = exercises[[FLOW_LABEL, VOLUME_LABEL]], exercises[EXERCISE_LABEL]
            X_test = X_test.append(X)
            y_test = y_test.append(y)

        return X_train, X_test, y_train, y_test


    def split_train_test(self, test_size=0.25):
        # i ranges from 1 to 7 as 8th exercise is a compound one
        X_train = pd.DataFrame(columns=[FLOW_LABEL, VOLUME_LABEL])
        X_test = pd.DataFrame(columns=[FLOW_LABEL, VOLUME_LABEL])
        y_train = pd.Series()
        y_test = pd.Series()

        for i in range(1, self.get_qty_exercises()):
            exercise = 'exerc_' + str(i)
            exercises = self.dataset.loc[self.dataset[EXERCISE_LABEL] == exercise]

            X, y = exercises[[FLOW_LABEL, VOLUME_LABEL]], exercises[EXERCISE_LABEL]
            X_train_exerc, X_test_exerc, y_train_exerc, y_test_exerc = train_test_split(X, y, test_size=test_size)
            
            X_train = X_train.append(X_train_exerc)
            X_test = X_test.append(X_test_exerc)
            y_train = y_train.append(y_train_exerc)
            y_test = y_test.append(y_test_exerc)
        
        return X_train, X_test, y_train, y_test
            
class Data:
    def __init__(self, data_file):
        filename = os.path.basename(data_file)
        splitted_filename = filename.split('.')
        splitted_vpt_filename = splitted_filename[0].split('__')

        self.vpt_file = {
            'device': splitted_vpt_filename[0],
            'exercise': splitted_vpt_filename[1],
            'uuid': splitted_vpt_filename[2],
            'date': splitted_vpt_filename[3],
            'extension': '.' + splitted_filename[1]
        }

        with open(data_file, 'r', newline='') as csvfile:
            fieldnames = ['time', 'event_0', 'event_1', 'event_2', 'volume', 'flow']
            reader = csv.DictReader(csvfile, delimiter=' ', fieldnames=fieldnames)

            self.vpt_data_joined = []
            self.vpt_data_separated = {}

            for fieldname in fieldnames:
                self.vpt_data_separated[fieldname] = np.array([])
            
            for row in reader:
                del row[None]
                self.vpt_data_joined.append(row)

                for fieldname in fieldnames:
                    self.vpt_data_separated[fieldname] = np.append(self.vpt_data_separated[fieldname], float(row[fieldname]))
    
    def get_data(self, separated_vars=True):
        if separated_vars:
            return self.vpt_data_separated
        else:
            return self.vpt_data_joined

    def get_UUID(self):
        return self.vpt_file['uuid']
    
    def get_exercise(self):
        return self.vpt_file['exercise']


if __name__ == "__main__":
    dataset = Dataset('../datasets/dataset_12_06_2019.csv')