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

print(y_train)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


'''
#実験で用いたDNNの構築および学習フェーズ.

model = Sequential()
model.add(Dense(8, activation='sigmoid', input_shape=(16,)))
#model.add(Dropout(0.2))
model.add(Dense(8, activation='sigmoid'))
#model.add(Dropout(0.2))
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

from keras.utils import plot_model
plot_model(model, show_shapes = True, to_file='model.png')



score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#print(x_train[0])
from keras import backend as K


get_1 = K.function([model.layers[0].input],[model.layers[0].input])
get_2 = K.function([model.layers[0].input],[model.layers[0].output])
get_3 = K.function([model.layers[0].input],[model.layers[1].input])
get_4 = K.function([model.layers[0].input],[model.layers[1].output])
get_5 = K.function([model.layers[0].input],[model.layers[2].input])
get_6 = K.function([model.layers[0].input],[model.layers[2].output])



dict = {}

avg0 = np.array([0.00033098, 0.06487112, 0.1871493,  0.01351397, 0.02650124, 0.44755507, 0.41085368, 0.18600947, 0.121864 ,  0.384588,   0.35221705, 0.14019826, 0.01915544, 0.26886532, 0.15775806, 0.00315101])
avg1 = np.array([0.00006289, 0.02068981, 0.10744543, 0.00580878, 0.00019739, 0.0816993, 0.36046848, 0.00279409, 0.00047182, 0.24658336, 0.20450355, 0.00068207,0.00267432, 0.12762639, 0.06014046, 0.00026174])

use_input = 1000
threshold = 0.2
subject = 0



for i in range(4):
    for j in range(4):
        x_train[0][i*4+j], x_train[subject][i*4+j] = x_train[subject][i*4+j], x_train[0][i*4+j]

y_train[0], y_train[subject] = y_train[subject], y_train[0]


print(x_train[subject])

'''


def distance(a,b):
    t = 0
    for i in range(16):
        t += math.pow((a[i]-b[i]),2)
    return math.sqrt(t)

table = np.zeros((use_input, 0b1111111111111111+1))



counter = 0
uselist = []
for c in range(use_input):
    uselist.append(c+1)
    #iの値について、1のあるところの素子を使わない
    for i in range(0b1111111111111111 + 1):
        tmp = x_train[c+1:c+2].copy()
        for j in range(len(bin(i))):
            if(bin(i)[j]=='1'):
                tmp[0][16 - (len(bin(i)) - j)]=x_train[0][16 - (len(bin(i)) - j)]
        table[c][i] = distance(x_train[0:1][0],tmp[0])

        del tmp
        counter += 1


np.save('table_for_shapley.npy', table)

'''


table = np.load('table_for_shapley.npy')

print("table complete!")

'''
for i in range(0b1111111111111111 + 1):
    tmp = x_train[0:1].copy()
    for j in range(len(bin(i))):
        if(bin(i)[j]=='1'):
            tmp[0][16 - (len(bin(i)) - j)]=avg0[16 - (len(bin(i)) - j)]
    dict[i] = get_6(tmp)[0][0][0]
    del tmp
'''
'''
out = np.zeros((use_input,))
for i in range(use_input):
    out[i] = get_6(x_train[i+1:i+2])[0][0][0]
'''


out = np.zeros((use_input,))
counter_ = 0
for i in range(use_input):
    '''
    if(y_train[i+1][0]==0):
        i += 1
        continue
    '''
    out[counter_] = get_6(x_train[i+1:i+2])[0][0][0]
    counter_ += 1


for i in range(0b1111111111111111 + 1):
    sum = 0
    count = 0
    for j in range(use_input):
        if(table[j][i]<threshold ):
            count += 1
            sum += out[j]
    if(not(count==0)):
        dict[i] = sum/count
    else:
        dict[i] = 0.5

#print(dict)
#
# dictの中身を, swapper(c,d)で入れ替えることができる
# やりたいことはswapperで入れ替えた中身をもとに複数のshapley値を計算すること
# 一旦、dictを保存しておく
dict_tmp = dict.copy()

'''
#通常のシャプレー値のための計算
#16ピクセルそれぞれについて計算したかったらここのコメントアウトを戻す
import itertools
import math
l_ = []
for i in range(17):
    l_ += list(itertools.combinations(range(16), i))
#print(l_)

#print(l_[1])
list_ans =[]
for i in (l_):
    tmp = 0
    for j in range(16):
        if(j in i):
            tmp += math.pow(2,j)
    list_ans.append(tmp)

#print(list_ans)
#print(len(list_ans))

list_for_shapley = []
for i in range(len(list_ans)):
    list_for_shapley.append(dict[0b1111111111111111-list_ans[i]]-0.5)
#print(list_for_shapley)
print(len(list_for_shapley))
print(list_for_shapley)
'''

list_adj = []
'''
#隣接ビットのリスト化
for i in range(15):
    if(i%4==3):
        list_adj.append([i,i+4])
    else:
        if(i>=12):
            list_adj.append([i,i+1])
        else:
            list_adj.append([i,i+1])
            list_adj.append([i,i+4])
'''
for i in range(16):
    for j in range(16):
        if(i<j):
            list_adj.append([i,j])

import itertools
import math
dict_tmp = {}


#swapper
num_bit = 16
k = pow(2,num_bit)
dict2=[-1]*k

def biter(y):
    str = ''
    for i in range(num_bit):
        if(y%2==1):
            str = '1' + str
        else:
            str = '0' + str
        y = int(y/2)
    return str

def rev_biter(st):
    c=0
    for i in range(num_bit):
        if(st[i]=='1'):
            c += pow(2,num_bit-1-i)
    return c

# c,d=0~15 c<d
def swapper(c,d):
    if(c>d):
        swapper(d,c)
    if(c!=1):
        #右からc桁目の数と右から0桁目の数とを入れ替える
        #pow(2,c+1)で割り切れ、pow(2,c+2)で割り切れなければ1
        for i in range(k):
            s = biter(i)

            tmp1 = s[len(s)-1- 0 ]
            tmp2 = s[len(s)-1- c ]
            s = s[:len(s)-1- c ] + tmp1 + s[len(s)-1-c+1:]
            s = s[:len(s)-1- 0 ] + tmp2 + s[len(s)-1-0+1:]

            tmp1 = s[len(s)-1- 1 ]
            tmp2 = s[len(s)-1- d ]
            s = s[:len(s)-1- d ] + tmp1 + s[len(s)-1-d+1:]
            s = s[:len(s)-1- 1 ] + tmp2 + s[len(s)-1-1+1:]

            j = rev_biter(s)
            dict2[j]=dict_tmp[i]
            

    else:
        for i in range(k):
            s = biter(i)

            tmp1 = s[len(s)-1- 0 ]
            tmp2 = s[len(s)-1- d ]
            s = s[:len(s)-1- d ] + tmp1 + s[len(s)-1-d+1:]
            s = s[:len(s)-1- 0 ] + tmp2 + s[len(s)-1-0+1:]

            j = rev_biter(s)
            dict2[j]=dict_tmp[i]
            

#shapley value calculation
from itertools import combinations
import math
import bisect
import sys
n = 15
characteristic_function = []
def power_set(List):
    PS = [list(j) for i in range(len(List)) for j in combinations(List, i+1)]
    return PS

def shp():


    tempList = list([i for i in range(n)])
    N = power_set(tempList)

    shapley_values = []
    for i in range(n):
        shapley = 0
        for j in N:
            if i not in j:
                cmod = len(j)
                Cui = j[:]
                bisect.insort_left(Cui,i)
                l = N.index(j)
                k = N.index(Cui)
                temp = float(float(characteristic_function[k]) - float(characteristic_function[l])) *\
                           float(math.factorial(cmod) * math.factorial(n - cmod - 1)) / float(math.factorial(n))
                shapley += temp
                # if i is 0:
                #     print j, Cui, cmod, n-cmod-1, characteristic_function[k], characteristic_function[l], math.factorial(cmod), math.factorial(n - cmod - 1), math.factorial(n)

        cmod = 0
        Cui = [i]
        k = N.index(Cui)
        temp = float(characteristic_function[k]) * float(math.factorial(cmod) * math.factorial(n - cmod - 1)) / float(math.factorial(n))
        shapley += temp

        shapley_values.append(shapley)

    print (shapley_values)

b = True
'''
#単一のshapley値計算の場合、ここをオープン
list_adj = [[0,1]]
b = False
#ここまで
'''

if(b):
    loop = 15
else:
    loop = 16


for count in range(len(list_adj)):

    dict_tmp = dict.copy()
    swapper(list_adj[count][0], list_adj[count][1])
    print(list_adj[count][0], list_adj[count][1],"におけるshapley値を計算しています")

    l_ = []
    for i in range(loop+1):
        l_ += list(itertools.combinations(range(loop), i))
    #print(l_)

    #print(l_[1])
    list_ans =[]
    for i in (l_):
        tmp = 0
        for j in range(loop):
            if(j in i):
                tmp += pow(2,j)
        list_ans.append(tmp)

    #print(list_ans)
    #print(len(list_ans))

    list_for_shapley = []
    if(b):
        for i in range(len(list_ans)):
            list_for_shapley.append(dict2[0b1111111111111111-(list_ans[i])*2-list_ans[i]%2]-0.5)
    #print(list_for_shapley)
    #print(len(list_for_shapley))
    #print(list_for_shapley)
        characteristic_function = list_for_shapley
        shp()

    else:
        for i in range(len(list_ans)):
            list_for_shapley.append(dict2[0b1111111111111111-(list_ans[i])]-0.5)
        characteristic_function = list_for_shapley
        shp()

    del dict_tmp 





'''
[-0.030900065248309596, 0.04592114226108106, 0.10949564375655482, 0.04613471674684974, 
0.042560292719545964, 0.03232713551693958, 0.03564632378280154, 0.021313193078251448, 
0.03288748576356951, 0.025240598958340354, 0.027434083004715143, 0.022888447415347554, 
0.022872296520835188, 0.02496405329881586, 0.016244438272598312, 0.021957139754022834]


[-0.03113138581829844, -0.016020399711233826, 0.19044164980527445, -0.02899322369601251, 
0.03331161035050455, 0.03898688700132033, 0.0670981053419035, -0.00517335424537042, 
0.08920391000885405, -0.08981144985197728, 0.2878904256130694, -0.03113138581829844, 
-0.031100663023046416, 0.08453429946020985, -0.029986713996640442, -0.03113138581829844]


↓ドントケアを出力側の平均値にする
[0.029065900338557372, 0.029116566337213247, 0.029830503039354774, 0.029028411977884368, 
0.029097489595459207, 0.02939693145161248, 0.028105082392762517, 0.028249708185769627, 
0.03004846744433293, 0.031105659168784005, 0.029283697460505798, 0.029207519909101805, 
0.029048950304059417, 0.028899838999229172, 0.028783085907571306, 0.029072492373072436]

↓データ数を1000に
[0.02903787936744869, 0.028375762949945576, 0.02881303552071442, 0.028941354440565694, 
0.028986142357946106, 0.028749278988724834, 0.02865691969635969, 0.028440077870253173, 
0.02849390236681639, 0.02812133517702058, 0.028436629341752066, 0.02900931195505206, 
0.028889766919599623, 0.028393597940916438, 0.028777870821078397, 0.029031946332903007]

↓データ数1000, 0も1も平均に含める
[-0.001959104794182115, -0.019554603446559696, 0.120531513877179, -0.0032867552143977367, 
0.010465446118911727, 0.03700096314326078, 0.007375450212954786, -0.025981782804611345, 
0.09092888610698695, -0.0042480336199870105, 0.23744806072916225, -0.004187812195395766, 
-0.0033214934391844705, 0.06708120757497904, -0.01096883797045617, -0.001741797761533174]

'''



'''
#データ0における
0 1 におけるshapley値を計算しています
-0.016193921088516174
0 4 におけるshapley値を計算しています
0.012221724011861
1 2 におけるshapley値を計算しています
0.09687458119014256
1 5 におけるshapley値を計算しています
0.024764957622351192
2 3 におけるshapley値を計算しています
0.10764696820629048
2 6 におけるshapley値を計算しています
0.11790154926373843
3 7 におけるshapley値を計算しています
-0.02379749525418102
4 5 におけるshapley値を計算しています
0.05736326046617216
4 8 におけるshapley値を計算しています
0.11003960993311805
5 6 におけるshapley値を計算しています
0.0564039126348622
5 9 におけるshapley値を計算しています
0.04482546895222821
6 7 におけるshapley値を計算しています
-0.01225220068384781
6 10 におけるshapley値を計算しています
0.21528286390901555
7 11 におけるshapley値を計算しています
-0.024236478645850098
8 9 におけるshapley値を計算しています
0.09657757237334566
8 12 におけるshapley値を計算しています
0.08692313339049061
9 10 におけるshapley値を計算しています
0.2133184130353442
9 13 におけるshapley値を計算しています
0.06319261977379183
10 11 におけるshapley値を計算しています
0.20127369373891318
10 14 におけるshapley値を計算しています
0.19983158486763786
11 15 におけるshapley値を計算しています
-0.00428952055205962
12 13 におけるshapley値を計算しています
0.06166543376552309
13 14 におけるshapley値を計算しています
0.05432977945216718
14 15 におけるshapley値を計算しています
-0.010155311169126992

[-0.0021112203235780116, -0.015655354278850786, 0.10289073290373146, 0.009007833572208818, 0.014217860661031746, 0.03258087459124777, 0.010343253272689694, -0.008215157991614534, 0.060004000025809995, 0.020662713115256397, 0.08407122853312465, 0.10742923155655719, 0.03982159222814878, -0.007254304066920826, 0.047788022718282454]
'''