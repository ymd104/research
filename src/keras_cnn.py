from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam

from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import plot_model, np_utils
from keras.callbacks import TensorBoard
from keras.datasets import cifar10

import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

import os
import numpy as np
import warnings

import matplotlib.pyplot as plt
import plotly
import seaborn

plotly.offline.init_notebook_mode(connected=False)
warnings.filterwarnings('ignore')
seaborn.set_style("whitegrid", {'grid.linestyle': '--'})

img_rows, img_cols = 32, 32
img_channels = 3
nb_classes = 10
nb_epochs = 50
batch_size = 64

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


old_session = KTF.get_session()
session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)


model = Sequential()
model.add(Conv2D(64, 3, input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, 3))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32, 2))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(2048))
model.add(Activation('relu'))
model.add(Dropout(rate=0.2))

model.add(Dense(nb_classes, activation='softmax'))

adam = Adam(lr=1e-4)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

tb_cb = keras.callbacks.TensorBoard('/logs', histogram_freq=1)
cbks = [tb_cb]

model.summary()

history = model.fit(X_train, Y_train, batch_size=batch_size,
                   nb_epoch=nb_epochs, verbose=1, validation_split=0.2, callbacks=cbks)

TensorBoard(log_dir='/logs')


KTF.set_session(old_session)
session.close()


import plotly.plotly as py
import plotly.graph_objs as go

accuracy = go.Scatter(
    y = history.history['acc'],
    x = np.array(list(range(nb_epochs))) + 1,
    name='accuracy',
    mode='lines+markers',
    line=dict(
        color=('#086039')
    ),
    marker=dict(
        color=('#086039')
    )
)

val_accuracy = go.Scatter(
    y = history.history['val_acc'],
    x = np.array(list(range(nb_epochs))) + 1,
    name='val_acc',
    mode='lines+markers',
    line=dict(
        color=('#f44262')
    ),
    marker=dict(
        color=('#f44262')
    )
)


data = [accuracy, val_accuracy]
plotly.offline.iplot(data)

loss = go.Scatter(
    y = history.history['loss'],
    x = np.array(list(range(nb_epochs))) + 1,
    name='loss',
    mode='lines+markers',
    line=dict(
        color=('#a2fc23')
    ),
    marker=dict(
        color=('#a2fc23')
    )
)

val_loss = go.Scatter(
    y = history.history['val_loss'],
    x = np.array(list(range(nb_epochs))) + 1,
    name='val_loss',
    mode='lines+markers',
    line=dict(
        color=('#00dfc3')
    ),
    marker=dict(
        color=('#00dfc3')
    )
)


data = [loss, val_loss]
plotly.offline.iplot(data)

fig = plt.figure(figsize=(10,5),dpi=200)

seaborn.lineplot(
    color='#086039',
    data=np.array(history.history['acc']),
    label="accuracy",
    marker="o"
)

seaborn.lineplot(
    markers=True,
    color='#f44262',
    data=np.array(history.history['val_acc']),
    label="Validation Accuracy",
    marker="o"
)

fig = plt.figure(figsize=(10,5),dpi=200)

seaborn.lineplot(
    color='#a2fc23',
    data=np.array(history.history['loss']),
    label="Loss",
    marker="o"
)

seaborn.lineplot(
    markers=True,
    color='#00dfc3',
    data=np.array(history.history['val_loss']),
    label="Validation Loss",
    marker="o"
)

