import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

name = input('What is your name: ')
M1 = int(input("First Midterm Mark: "))
M2 = int(input('Second Midterm Mark: '))

column_names = ['G1', 'G2', 'G3']
raw_dataset = pd.read_csv('student-mat.csv', sep=';')
raw_dataset = raw_dataset[column_names]
dataset = raw_dataset.copy()

train_data = dataset.sample(frac=0.9, random_state=0)
test_data = dataset.drop(train_data.index)

train_label = train_data.pop('G3')
test_label = test_data.pop('G3')

train_stats = train_data.describe()
train_stats = train_stats.transpose()

def normalize(data):
    return (data-train_stats['mean'])/train_stats['std']
norm_train_data = normalize(train_data)
norm_test_data = normalize(test_data)

model = keras.models.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[len(test_data.keys())]),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer="nadam", loss='mse', metrics=['accuracy'])
model.fit(norm_train_data, train_label, epochs=300)
model.evaluate(norm_test_data, test_label)

M1 = normalize(M1)
M2 = normalize(M2)

predict = model.predict([[M1, M2]])
predict = int(predict[0])
print(name, "this is your predicted final grade:", predict)

