import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
# when creating training/testing data, only pass in 80-90%

(train_images, train_labels), (test_images, test_labels) = data.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images/255
test_images = test_images/255
# creating model

model = keras.Sequential([  # creating neural network
    keras.layers.Flatten(input_shape=(28,28)),  # Flatten input layer
    keras.layers.Dense(128, activation='relu'),  # Set hidden layer, 128 neurons
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')  # output layer. Softmax turns into probabilities that add to one
])

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # setting loss function and optimizer
model.fit(train_images, train_labels, epochs=10)  # training

# test_loss, test_ac = model.evaluate(test_images, test_labels)
# print("Tested:", test_ac)
#
prediction = model.predict(test_images)
print(class_names[np.argmax(prediction[0])])

for i in range(5):
    plt.grid = False
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()
    