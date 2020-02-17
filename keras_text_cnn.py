#from keras.datasets import imdb
from keras.preprocessing import sequence
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import History, LearningRateScheduler, Callback
from keras import layers
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Lambda, LSTM, GRU, Embedding

sentence = np.load('sentence.npy')
label = np.load('label.npy')

#(x_train, y_train), (x_test, y_test) = imdb.load_data()
#print(type(x_train))
#print(type(x_test))
x_train=sentence[:100000]
y_train=label[:100000]
from keras.preprocessing import sequence
x_train = sequence.pad_sequences(x_train,maxlen=886)



x_test=sentence[100000:273696]
y_test=label[100000:273696]
x_test= sequence.pad_sequences(x_test)

print((x_train).shape)
print((x_test).shape)
print((x_test).shape[1]-(x_train).shape[1])

#np.pad(x_train, [(0,0), (0, 334)], 'constant')

print((x_train).shape)
print((x_test).shape)

max_words=87171 # 単語のインデックスの数
max_len=886
model = Sequential()
model.add(layers.Embedding(max_words, 128, input_length=max_len))
model.add(layers.SeparableConv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.SeparableConv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=5, batch_size=128, validation_split=0.2)

model.save('mlp_cnn.h5')

# 精度検証
_, acc = model.evaluate(x_test, y_test, verbose=1)
print('\nTest accuracy: {0}'.format(acc))