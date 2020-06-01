import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv('car_data.txt')
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
clss = le.fit_transform(list(data['class']))

predict = 'class'

new_data = list(zip(buying, maint, door, persons, lug_boot, safety))
new_label = list(clss)

train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(new_data, new_label, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(train_data, train_label)
acc = model.score(test_data, test_label)
print(acc)

predicted = model.predict(test_data)
names = ["unacc", "acc", "good", "vgood"]

for i in range(len(test_data)):
    print("Predicted: ", names[predicted[i]], "Data: ", test_data[i], "Actual: ", names[test_label[i]])