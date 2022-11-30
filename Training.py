from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf


class Person:
  def __init__(self, person_sex, age, n_siblings_spouses, ticket_fare, ticket_class, deck, embark_town, alone, parch=0):
    self.person_sex = person_sex
    self.age = age
    self.n_siblings_spouses = n_siblings_spouses
    self.parch = parch
    self.ticket_fare = ticket_fare
    self.ticket_class = ticket_class
    self.deck = deck
    self.embark_town = embark_town
    self.alone = alone
    self.dataframe = pd.DataFrame({"sex":[self.person_sex],
                      "age":[self.age],
                      "n_siblings_spouses":[self.n_siblings_spouses],
                      "parch":[self.parch],
                      "fare":[self.ticket_fare],
                      "class":[self.ticket_class],
                      "deck":[self.deck],
                      "embark_town":[self.embark_town],
                      "alone":[self.alone]})

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')



CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
  
 
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)


linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on testing data

clear_output()



number = 148
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
result = list(linear_est.predict(eval_input_fn))



def did_they_survive():
  person = Person('female', 72, 0, 157.0, 'Second', 'A', 'Southampton', 'y',0)
  dataframe = person.dataframe

  eval_1 = pd.DataFrame({0:[0]})
  #print(dataframe)
  eval_input_fn = make_input_fn(dataframe, eval_1, num_epochs=2, shuffle=False)
  result = list(linear_est.predict(eval_input_fn))
  print(result[0]['probabilities'][1])

did_they_survive()
