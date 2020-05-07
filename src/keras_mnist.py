import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np

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

print(x_train_u[11845])

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

'''
print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))

print(x_train[0])
print(x_train[0][5])
print(x_train[0][5][10])
print(x_train[0][5][11])
'''

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

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('mnist.h5')
#model = load_model('mlp.h5')

#print(x_train[0])
from keras import backend as K

print("layer size:", len(model.layers))

get_1 = K.function([model.layers[0].input],[model.layers[0].input])
get_2 = K.function([model.layers[0].input],[model.layers[0].output])
get_3 = K.function([model.layers[0].input],[model.layers[1].input])
get_4 = K.function([model.layers[0].input],[model.layers[1].output])
get_5 = K.function([model.layers[0].input],[model.layers[2].input])
get_6 = K.function([model.layers[0].input],[model.layers[2].output])
'''
get_7 = K.function([model.layers[0].input],[model.layers[3].input])
get_8 = K.function([model.layers[0].input],[model.layers[3].output])
get_9 = K.function([model.layers[0].input],[model.layers[4].input])
get_10 = K.function([model.layers[0].input],[model.layers[4].output])
'''

print(get_1(x_train[0:1]))
print(get_6(x_train[0:1]))

print(get_1(x_train[1:2]))
print(get_6(x_train[1:2]))

print(get_1(x_train[2:3]))
print(get_6(x_train[2:3]))
'''
print(get_7(x_train[0:1]))
print(get_8(x_train[0:1]))
print(get_9(x_train[0:1]))
print(get_10(x_train[0:1]))
'''


'''
・入力ラベルの比を1:1にする

・特性関数の計測

4*4=16素子それぞれの参加について、
2^16通りの特性関数の値を算出する。算出については、使わない素子の入力を0にしてモデルを通し、
出力と0.5との差を値とする

・Shapley値の計測

pythonのコードが転がっているので、試しながら使おうとしてみる
まずはn=3ぐらいのテストケースから。入力をどうすればいいかを確かめ、
入力に向けてのセットアップをする

・Shapley値の使用

例えば素子Aに関して、Shapley値の大きさで順序ができる
値が大きいものはより良い提携構造である
各提携を大きい順に印字する

・LRPの実装

構造が単純なのでやりやすい
LRP値では要素同士の組み合わせを考慮できないが、shapley値を提携構造問題に適用することで考慮できる
'''