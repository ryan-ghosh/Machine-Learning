import tensorflow as tf
from tensorflow import keras
import numpy as np

## getting data in correct format
data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)
word_index = data.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=256)

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


## building model
model = keras.Sequential([
    keras.layers.Embedding(10000, 16),  # generate word vectors and group together based on how similar they are
    keras.layers.GlobalAveragePooling1D(),  # lowers dimension
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # output either 0 or 1
])


model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # binary used since output is 0 or 1
x_val = train_data[:10000]
x_train = train_data[10000:]
y_val = train_labels[:10000]
y_train = train_labels[10000:]

fit_model = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val,y_val), verbose=1)  # training
results = model.evaluate(test_data, test_labels)

print(results)
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: "+str(predict[0]))
print("Actual: " + str(test_labels[0]))
print(results)


## saving model
#model.save("model.h5")

## making model work for text file review

# def encode_review(s):
#     encoded = [1]
#     for word in s:
#         if word.lower() in word_index:
#             encoded.append(word_index[word])
#         else:
#             encoded.append(2)
#     return encoded

# model = model.keras.load_model("model.h5")
# with open("text.txt", encoding="utf-9") as f:
#     for line in f.readlines():
#         nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
#         encode = review_encode(nline)
#         encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding='post', maxlen=256)
#         predict = model.predict(encode)
#         print(line)
#         print(encode)
#         print(predict[0])