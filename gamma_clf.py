from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

telescope = pd.read_csv('radiation_dataset.csv')

telescope_shuffle = telescope.iloc[np.random.permutation(len(telescope))]
dataset = telescope_shuffle.reset_index(drop=True)

dataset['class'] = dataset['class'].map({'g':0, 'h':1})
dataset_class = dataset['class'].values

training_indices, validation_indices = training_indices, testing_indices = train_test_split(dataset.index, stratify=dataset_class, train_size=0.75, test_size=0.25)

clf = DecisionTreeClassifier()
clf.fit(dataset.drop('class', axis=1).loc[training_indices].values, dataset.loc[training_indices, 'class'].values)

score = clf.score(dataset.drop('class', axis=1).loc[testing_indices].values, dataset.loc[testing_indices, 'class'].values)

print (score)
