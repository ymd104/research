import keras
from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np

from keras.preprocessing import sequence
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import History, LearningRateScheduler, Callback
from keras import layers
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Input, Lambda, LSTM, GRU, Embedding, Dropout

batch_size = 128
num_classes = 2
epochs = 20

# the data, shuffled and split between train and test sets
(x_train_u, y_train_u), (x_test_u, y_test_u) = mnist.load_data()

print(x_train_u.shape)
print(y_train_u.shape)



x_train_=np.zeros((11846,28,28))
y_train=np.zeros((11846,))
x_test_=np.zeros((1960,28,28))
y_test=np.zeros((1960,))

count1 = 0
count_one = 0
for i in range(60000):
    if(y_train_u[i]==0):
        #print(x_train[i].type)
        #print(x_train_u[i])
        x_train_[count1]=(x_train_u[i])
        y_train[count1]=(y_train_u[i])
        count1 += 1
    else: 
        if(y_train_u[i]==1 and count_one<5923):
            x_train_[count1]=(x_train_u[i])
            y_train[count1]=(y_train_u[i])
            count1 = count1 + 1
            count_one += 1



count1 = 0
count_one = 0
for i in range(10000):
    if(y_test_u[i]==0):
        x_test_[count1]=(x_test_u[i])
        y_test[count1]=(y_test_u[i])
        count1 = count1 + 1
    else:
        if(y_test_u[i]==1 and count_one<980):
            x_test_[count1]=(x_test_u[i])
            y_test[count1]=(y_test_u[i])
            count1 = count1 + 1
            count_one += 1


x_train = np.zeros((11846,4,4))
x_test = np.zeros((1960,4,4))


#28*28の画像ファイルを4*4に圧縮する関数
def compress(num):
    for l in range(4):
        for m in range(4):
            sum = 0
            for i in range(7):
                for j in range(7):
                    sum = sum + x_train_[num][l*7+i][m*7+j]
            x_train[num][l][m]=sum/49

def compress2(num):
    for l in range(4):
        for m in range(4):
            sum2 = 0
            for i in range(7):
                for j in range(7):
                    sum2 = sum2 + x_test_[num][l*7+i][m*7+j]
            x_test[num][l][m]=sum2/49
            


for t in range(11846):
    compress(t)
for t in range(1960):
    compress2(t)

print(x_train[0])
print(x_train[1])
print(x_train[2])

x_train = x_train.reshape(11846, 16) # 2次元配列を1次元に変換
x_test = x_test.reshape(1960, 16)
x_train = x_train.astype('float32')   # int型をfloat32型に変換
x_test = x_test.astype('float32')
x_train /= 255                        # [0-255]の値を[0.0-1.0]に変換
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



'''

model = Sequential()
model.add(Dense(8, activation='sigmoid', input_shape=(16,)))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])





history = model.fit(x_train, y_train,  # 画像とラベルデータ
                    batch_size=batch_size,
                    epochs=epochs,     # エポック数の指定
                    verbose=1,         # ログ出力の指定. 0だとログが出ない
                    validation_data=(x_test, y_test))

'''

#model.save('mnist.h5')
model = load_model('mnist.h5')

'''

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''

#print(x_train[0])
from keras import backend as K

get_1 = K.function([model.layers[0].input],[model.layers[0].input])
get_2 = K.function([model.layers[0].input],[model.layers[0].output])
get_3 = K.function([model.layers[0].input],[model.layers[1].input])
get_4 = K.function([model.layers[0].input],[model.layers[1].output])
get_5 = K.function([model.layers[0].input],[model.layers[2].input])
get_6 = K.function([model.layers[0].input],[model.layers[2].output])


#LRP

def sg(x):
    if(x>=0) :return 1.0
    else :return -1.0

eps = 0.001

def lrp(n):

    x1 = get_1(x_train[n:n+1])
    x2 = get_3(x_train[n:n+1])
    x3 = get_5(x_train[n:n+1])

    w1 = (model.get_weights()[0])
    b1 = (model.get_weights()[1])
    w2 = (model.get_weights()[2])
    b2 = (model.get_weights()[3])
    w3 = (model.get_weights()[4])
    b3 = (model.get_weights()[5])

    if(y_train[n][1]==0):
        x4 = get_6(x_train[n:n+1])[0][0][0] - get_6(x_train[n:n+1])[0][0][1] 
    else:
        #x4 = get_6(x_train[n:n+1])[0][0][1]
        x4 = get_6(x_train[n:n+1])[0][0][0] - get_6(x_train[n:n+1])[0][0][1] 
        #”0らしさ”をLRPにより分解する

    print("x4:", x4)


    sum = np.dot(x3,w3)[0][0][0]
    l3 = []
    for i in range(8):
        l3.append(((x3[0][0][i]*w3[i][0] + eps * sg(sum) / 8) / (sum + eps*sg(sum)) ) * x4 )
    #print(l3)

    c = np.dot(x2,w2)[0][0]
    l2 = [0,0,0,0,0,0,0,0]
    for i in range(8):
        sum = c[i]
        for j in range(8):
            l2[j] += ((x2[0][0][j]*w2[j][i]) + eps * sg(sum) / 8)/(sum + eps*sg(sum)) * l3[i]
    #print(l2)

    l1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    c = np.dot(x1,w1)[0][0]
    for i in range(8):
        sum = c[i]
        for j in range(16):
            l1[j] += ((x1[0][0][j]*w1[j][i]) + eps * sg(sum) / 16)/(sum + eps*sg(sum)) * l2[i]
            #print(i,j,w1[j][i])
            

    ar = np.array(l1)
    a = np.reshape(ar,(4,4))

    #print(a)
    return a

