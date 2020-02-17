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
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Input, Lambda, LSTM, GRU, Embedding

#シェルで省略せずに表示
np.set_printoptions(threshold=np.inf)

sentence = np.load('sentence.npy')
label = np.load('label.npy')

#(x_train, y_train), (x_test, y_test) = imdb.load_data()
#print(type(x_train))
#print(type(x_test))
x_train=sentence[:1000]
y_train=label[:1000]
from keras.preprocessing import sequence
x_train = sequence.pad_sequences(x_train)

#LRPに用いる文をここで指定
x_test=sentence[3:1000]
y_test=label[3:1000]
x_test= sequence.pad_sequences(x_test)

'''
max_words=87171 # 単語のインデックスの数
model2 = Sequential()
model2.add(Embedding(max_words, 32))
model2.add(LSTM(3))
model2.add(Dense(1, activation='sigmoid'))

model2.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model2.fit(x_train, y_train,
                    epochs=1, batch_size=128, validation_split=0.2)

'''

#model.save('mlp.h5')
model = load_model('mlp.h5')

'''

# 精度検証

_, acc = model2.evaluate(x_test, y_test, verbose=1)
print('\nTest accuracy: {0}'.format(acc))

'''

'''
print(sentence[2:34])
print(label[2:34])
print(model.predict(x_train[2:34], batch_size=None, verbose=1, steps=None), label[2:34])
for i in range(32):
    print(model.predict(x_train[i+2:i+3], batch_size=None, verbose=1, steps=None), label[i+2:i+3])
    _, acc = model.evaluate(x_test[i:i+1], label[i+2:i+3], verbose=1)
    print('\nTest accuracy: {0}'.format(acc))

'''

from keras import backend as K
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[2].output])
#K.function(入力,出力)で、そういう関数が作れるっぽい

print(model.layers[0].input_shape)
print(model.layers[0].output_shape)
print(model.layers[1].input_shape)
print(model.layers[1].output_shape)
print(model.layers[2].input_shape)
print(model.layers[2].output_shape)

#print(sentence[2:4])
#print(x_test[0:2])

'''
for i in range (32):
    layer_output = get_3rd_layer_output([x_test[i:i+1]])
    print(layer_output)

    print(model.predict(x_test[i:i+1], batch_size=1, verbose=1, steps=None), y_test[i:i+1])
'''

get_1 = K.function([model.layers[0].input],[model.layers[0].input])
get_2 = K.function([model.layers[0].input],[model.layers[0].output])
get_3 = K.function([model.layers[0].input],[model.layers[1].input])
get_4 = K.function([model.layers[0].input],[model.layers[1].output])
get_5 = K.function([model.layers[0].input],[model.layers[2].input])
get_6 = K.function([model.layers[0].input],[model.layers[2].output])
'''
print(get_1(x_test[0:1]))
print(get_2(x_test[0:1]))
print(get_3(x_test[0:1]))
print(get_4(x_test[0:1]))
print(get_5(x_test[0:1]))
print(get_6(x_test[0:1]))
'''

#print(model.get_weights())
'''
print(model.get_weights()[0].shape)
print(model.get_weights()[1].shape)
print(model.get_weights()[2].shape)
print(model.get_weights()[3].shape)
print(model.get_weights()[4].shape)
print(model.get_weights()[5].shape)

#print(model.get_weights()[1])
#print(model.get_weights()[2])
print(model.get_weights()[3])
'''


from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, expand_nested=True)


#print(model.get_weights()[4])

'''
print(get_1(x_test[0:1]))
print(get_2(x_test[0:1]))

print((get_1(x_test[0:1]))[0])
print((get_2(x_test[0:1]))[0][0].shape)
'''
'''
print(get_1(x_test[0:1]))
print(get_2(x_test[0:1]))
print(get_4(x_test[0:1]))
print("decoy")

#print(model.layers[1].get_weights())
print(len(model.layers[1].get_weights()))
#print(model.layers[1].get_weights()[0])
'''

'''
print(len(model.get_weights()))
print(model.get_weights()[0].shape)
print(model.get_weights()[1].shape)
print(model.get_weights()[2].shape)
print(model.get_weights()[3].shape)
print(model.get_weights()[4].shape)
print(model.get_weights()[5].shape)


print(get_5(x_test[0:1])[0][0])
print(model.get_weights()[4])
print(get_6(x_test[0:1])[0][0])
'''

lrp_last = get_6(x_test[0:1])[0][0]
lrp_1 = get_5(x_test[0:1])[0][0]

eps = 0.001

def sg(x):
    if(x>=0) :return 1.0
    else :return -1.0

#最後の全結合層
score_last = lrp_last
w = model.get_weights()[4]
n = 32
score = []
sum = 0
for i in range(32):
    score.append(score_last * ( (lrp_1[i]*w[i]) + eps * sg(lrp_last) / n ) / (lrp_last + eps * sg(lrp_last)) )
    #print(score_last, lrp_1[i], w[i], lrp_last)
    #score.append(score_last * ( (lrp_1[i]*w[i]) ) / (lrp_last + eps * sg(lrp_last)) )
    #print(i, ":", score[i])
    #sum = sum + score[i]

#print(sum)

'''
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

print(model.get_weights()[5])
t = 0
for i in range(32):
    t += lrp_1[i]*w[i]
t += model.get_weights()[5]
print(t)
print(sigmoid(t))
'''

#LSTM層
#入力
score_lstm_input = get_3(x_test[0:1])[0][0]
#出力
score_lstm_output = score
#重み
w = (model.get_weights()[1])[0:32,64:96]

'''
print(score_lstm_input)
print(score_lstm_output)
print(w)
'''


score_li = []
ci = []
wi = []

for i in range(179):
    score_li.append(0)

sum_cw=[]
for j in range(32):
    tmp = 0
    for i in range(179):
        tmp += np.dot((score_lstm_input[i:i+1,:]),(w[:,j:j+1]))
    sum_cw.append(tmp)
        

for i in range(179):
    for j in range(32):
        score_li[i] += np.dot((score_lstm_input[i:i+1,:]),(w[:,j:j+1])) / (sum_cw[j] + eps*sg(np.dot((score_lstm_input[i:i+1,:]),(w[:,j:j+1])))) * score_lstm_output[j]

'''
for i in range(179):
    s = 0
    su = 0
    for j in range(32):
        ci.append(score_lstm_input[i:i+1,:])
        wi.append(w[:,j:j+1])
        su += np.dot(ci[i],wi[j])
    for j in range(32):
        s += (np.dot(ci[i],wi[j])) / (su + eps * sg(np.dot(ci[i],wi[j]))) * score_lstm_output[j]
        score_li.append(s)
'''

print(get_1(x_test[0:1]))
#print(get_2(x_test[0:1]))

for i in range(179):
    print(score_li[i])

#print(len(score_li))

for i in range(179):
    if(score_li[i] > 0.01):
        print(get_1(x_test[0:1])[0][0][i])