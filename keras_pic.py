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


from PIL import Image

l = []
for t in range(30):
    l.append(Image.new('L',(4,4)))
    
'''
for count in range(30):
    for i in range(4):
        for j in range(4):
            l[count].putpixel((j,i),255-int(x_train[count][i][j])) 
            
    print(count, y_train[count])
    print(x_train[count])
    l[count] = l[count].resize((400,400))
    l[count].save('input' + str(count) + '.jpg')
'''


'''


ar = np.array([[-0.001959104794182115, -0.019554603446559696, 0.120531513877179, -0.0032867552143977367], 
[0.010465446118911727, 0.03700096314326078, 0.007375450212954786, -0.025981782804611345], 
[0.09092888610698695, -0.0042480336199870105, 0.23744806072916225, -0.004187812195395766], 
[-0.0033214934391844705, 0.06708120757497904, -0.01096883797045617, -0.001741797761533174]])


ar2 = np.array([[-0.0019961846612748385, -0.02284438809024696, 0.1282332216172438, -0.0030745361558027337]
, [0.00450360426861432, 0.02483209519050256, 0.0038979472039848122, -0.028490296470109335]
, [0.17833359555189304, -0.009665201721610239, 0.17931962545426824, -0.004517265029023952]
, [-0.003541790356850251, 0.06519009166741466, -0.012493505562098566, -0.0017409334143337883]])

ar3 = np.array([[0.00003,   0.01518, -0.49913, -0.00232],
 [0.00003,  -3.25778,  1.28797, -0.65011],
[-0.03514,  2.14672,  -0.32629, -0.08657],
[-0.02557,  2.46107,  -0.03113,  0.00003]])

print(ar)

clear = ar.max()
im2 = Image.new('RGB', (4,4))
for i in range(4):
    for j in range(4):
        if(ar[i][j]>0):
            im2.putpixel((j,i),(255, 255-int(ar[i][j]*255 / clear), 255-int(ar[i][j]*255 / clear)))
        else:
            im2.putpixel((j,i),(255-abs(int(ar[i][j]*255 / clear)), 255-abs(int(ar[i][j]*255 / clear)), 255))

im2 = im2.resize((400,400))
#im2.show()
#im2.save('output_th02.jpg')

clear = ar2.max()
im3 = Image.new('RGB', (4,4))
for i in range(4):
    for j in range(4):
        if(ar2[i][j]>0):
            im3.putpixel((j,i),(255, 255-int(ar2[i][j]*255 / clear), 255-int(ar2[i][j]*255 / clear)))
        else:
            im3.putpixel((j,i),(255-abs(int(ar2[i][j]*255 / clear)), 255-abs(int(ar2[i][j]*255 / clear)), 255))

im3 = im3.resize((400,400))
#im3.show()
#im3.save('output_th015.jpg')

clear = max(ar3.max(), -ar3.min())
clear += 0.001
#print(clear)
im4 = Image.new('RGB', (4,4))
for i in range(4):
    for j in range(4):
        if(ar3[i][j]>0):
            im4.putpixel((j,i),(255, 255-int(ar3[i][j]*255 / clear), 255-int(ar3[i][j]*255 / clear)))
        else:
            im4.putpixel((j,i),(255-abs(int(ar3[i][j]*255 / clear)), 255-abs(int(ar3[i][j]*255 / clear)), 255))

im4 = im4.resize((400,400))
#im4.show()
#im4.save('output_lrp.jpg')

'''

'''
k_ = np.array([0.04821338563138876, 0.0072004374945569445, 0.030732107502205654, 0.048213382282206896, 0.04821338563138876, 0.013277427729828248, 0.011067742622186407, 0.048213382282206896, 0.04821338563138876, 0.013662941282023871, 0.00315640781861568, 0.048213382282206896, 0.04821338563138876, 0.016922838271272285, 0.018250746566475254, 0.04821336920351238])
k = np.reshape(k_, (4,4))
clear = max(k.max(), -k.min())
clear += 0.001
im = Image.new('RGB', (4,4))
for i in range(4):
    for j in range(4):
        if(k[i][j]>0):
            im.putpixel((j,i),(255, 255-int(k[i][j]*255 / clear), 255-int(k[i][j]*255 / clear)))
        else:
            im.putpixel((j,i),(255-abs(int(k[i][j]*255 / clear)), 255-abs(int(k[i][j]*255 / clear)), 255))

im = im.resize((400,400))
im.show()
#im.save('output_0_th020_notmnist.jpg')
'''

'''
1
[[-0.0022000422503551443, -0.05531784468261412, 0.003823760430477545, -0.004020347653274551], 
[-0.03630135801670663, -0.09059731269724083, 0.005718235135373449, -0.03395016169332001],
[-0.053799180451758834, -0.022236322360982396, -0.13443975417696413, -0.004526079916752426], 
[-0.0031115821468382047, -0.030585373887423276, -0.017141866134300503, -0.00168086955125167]])

2
[[-0.0022505205131039584, 0.027154389762007028, -0.11382774118039024, -0.004439560803254317], 
[-0.03762398666598078, 0.07582174247527336, -0.10143536715146932, -0.03601617003863254], 
[-0.05892587425001711, -0.03880181053050746, -0.07249727266470671, -0.006165305483306902], 
[-0.0037263917902721154, -0.052005863655107995, -0.003188104352056199, -0.0016846171854215516]])

3
[-0.002195211635011149, -0.00617766046837449, -0.07532534064284027, -0.0034588037474700822, -0.034306088770969224, -0.01704497375864995, -0.07482439989417146, -0.03224871021708517, -0.050494944806559666, -0.02477248379237001, -0.09291881752308638, -0.0054729649566338575, -0.0030981764628406267, -0.047579117451500244, -0.015526237580022306, -0.0016800464519698963]

4
[-0.0022799066787010354, -0.010379999997490088, -0.07246875939970979, -0.003690083627418939, -0.040156416124465175, -0.05397657190896568, -0.04460598032980899, -0.03525252527007011, -0.055814231509757305, -0.02160787448472411, -0.06230517712708717, -0.005901209340914223, -0.003208320241202224, -0.05874492805156377, -0.008108453630263714, -0.001714001691326912]

5
[-0.0019890162130612324, 0.005383164826766372, 0.057435694495490575, -0.0032109148162161065, 0.010459724744091022, 0.026433101689072362, 0.005066880664640123, 0.12028939253489622, 0.05354893293588783, 0.035100971035085124, 0.17937609025360826, -0.004096529634774115, -0.0007431415126658277, 0.025732139048622434, -0.009028428077550839, -0.0017405124591178074]

12
[-0.001506594601829098, 0.15048505164783876, 0.01357386847364896, -0.0021961003021209215, 0.16264429684172912, -0.05736234159996589, 0.01403174232092889, 0.008557786454801581, 0.05450041409812546, -0.002006275192189584, 0.163294736826262, -0.002982915058725115, -0.0008406111087874492, -0.009460375618490273, 0.010678616815033255, -0.0017405560540013647]

16
[0.007592123545752406, 0.09720638056872691, -0.0033030129747448496, -0.0033191175141894878, 0.16887676005525823, 0.011602455781896106, -0.016555391957560187, 0.0634712482538653, 0.12757988607818124, 0.00021245010346739382, -0.07836331640471896, 0.06913085477772102, -0.002771202982279231, 0.02074505911532486, 0.036397601653493004, -0.0017430057994738902]


LRP値
0
[[ 0.00003177  0.01513576 -0.49761855 -0.00231426]
 [ 0.00003177 -3.24793633  1.28407973 -0.64814853]
 [-0.03503123  2.14023258 -0.32530511 -0.08631228]
 [-0.02549605  2.45362843 -0.03103475  0.00003177]]

1
[[-0.00010998 -0.00010998 -0.20893367 -0.00429831]
 [-0.00010998 -0.03195764  0.34985965 -0.01776246]
 [-0.00010998 -0.3771884  -0.12123233 -0.00010998]
 [-0.00010998 -0.54415662 -0.00010998 -0.00010998]]

2
[[-0.00006996, -0.07939215, -0.08146447, -0.00006996],
 [-0.00006996, -0.37232843,  0.14317616, -0.00006996],
 [-0.00006996, -0.07209669, -0.26853331, -0.00006996],
 [-0.00006996, -0.00006996, -0.17000122, -0.00006996]]

3
[[-0.00012922, -0.03732961, -0.07718548, -0.00012922],
 [-0.00012922, -0.46949318,  0.18321484, -0.00012922],
 [-0.00012922, -0.06595306, -0.30628817, -0.00012922],
 [-0.00012922, -0.00012922, -0.20880605, -0.00012922]]

4
[[-0.00009691, -0.06877732, -0.01645406, -0.00009691],
 [-0.00009691, -0.4587818 ,  0.11148137, -0.00009691],
 [-0.00009691, -0.14236958, -0.16331412, -0.00009691],
 [-0.00009691, -0.08966857, -0.13534155, -0.00009691]]

5
[[ 0.00002821,  0.00002821,  0.01248175, -0.01867898],
 [ 0.00002821, -0.24407167,  0.19534368, -0.0018364 ],
 [ 0.10913838,  0.36288648,  0.03678932,  0.04340415],
 [-0.00138493,  0.48760327,  0.01464275,  0.00002821]]

12
[[ 0.00002132,  0.03337109,  0.03153376,  0.00002132],
 [ 0.00679327,  0.17471546, -0.01440355,  0.07477326],
 [ 0.04909549,  0.18985374,  0.00745768,  0.18923203],
 [ 0.00002132,  0.16997985,  0.08368926,  0.0033358 ]]

 16
 [[ 0.00003927,  0.07931841,  0.06830594,  0.00003927],
 [ 0.04765666, -0.00052894,  0.02460834,  0.0595211 ],
 [ 0.10212836,  0.11951267,  0.00003927,  0.2174473 ],
 [ 0.00003927,  0.17768794,  0.10406355, -0.00682698]]


shapley(not mnist)
0
[0.04821338563138876, 0.0072004374945569445, 0.030732107502205654, 0.048213382282206896, 0.04821338563138876, 0.013277427729828248, 0.011067742622186407, 0.048213382282206896, 0.04821338563138876, 0.013662941282023871, 0.00315640781861568, 0.048213382282206896, 0.04821338563138876, 0.016922838271272285, 0.018250746566475254, 0.04821336920351238]

LRP(not mnist)
0
[[ 0.10801351,  0.04696358,  0.01664421,  0.11228534],
 [ 0.11924609, -0.00687479,  0.00000798,  0.13114487],
 [ 0.09640855, -0.00046077, -0.00075436,  0.10622714],
 [ 0.1155514,   0.06521315, -0.00905391,  0.09939354]]
'''

'''
k = np.array([[ 0.10801351,  0.04696358,  0.01664421,  0.11228534],
 [ 0.11924609, -0.00687479,  0.00000798,  0.13114487],
 [ 0.09640855, -0.00046077, -0.00075436,  0.10622714],
 [ 0.1155514,   0.06521315, -0.00905391,  0.09939354]])
clear = max(k.max(), -k.min())
clear += 0.001
im = Image.new('RGB', (4,4))
for i in range(4):
    for j in range(4):
        if(k[i][j]>0):
            im.putpixel((j,i),(255, 255-int(k[i][j]*255 / clear), 255-int(k[i][j]*255 / clear)))
        else:
            im.putpixel((j,i),(255-abs(int(k[i][j]*255 / clear)), 255-abs(int(k[i][j]*255 / clear)), 255))

im = im.resize((400,400))
im.show()
#im.save('output_0_lrp_notmnist.jpg')
'''

for i in range(30):
    print(y_train[i])


l0 = np.zeros((4,4))
l1 = np.zeros((4,4))
for i in range(11846):
    if(y_train[i]==0):
        for j in range(4):
            for k in range(4):
                l0[j][k] += x_train[i][j][k]
    else:
        for j in range(4):
            for k in range(4):
                l1[j][k] += x_train[i][j][k]

print(l0/5923)
print(l1/5923)
l0 = l0/5923
l1 = l1/5923

from statistics import stdev
l2 = [ [[],[],[],[]], [[],[],[],[]], [[],[],[],[]], [[],[],[],[]] ]
l3 = [ [[],[],[],[]], [[],[],[],[]], [[],[],[],[]], [[],[],[],[]] ]
for i in range(11846):
    if(y_train[i]==0):
        for j in range(4):
            for k in range(4):
                l2[j][k].append(x_train[i][j][k])
    else:
        for j in range(4):
            for k in range(4):
                l3[j][k].append(x_train[i][j][k])
for j in range(4):
    for k in range(4):
        print(j,k,stdev(l2[j][k])/l0[j][k])
for j in range(4):
    for k in range(4):
        print(j,k,stdev(l3[j][k])/l1[j][k])


from statistics import median
l2 = [ [[],[],[],[]], [[],[],[],[]], [[],[],[],[]], [[],[],[],[]] ]
l3 = [ [[],[],[],[]], [[],[],[],[]], [[],[],[],[]], [[],[],[],[]] ]
for i in range(11846):
    if(y_train[i]==0):
        for j in range(4):
            for k in range(4):
                l2[j][k].append(x_train[i][j][k])
    else:
        for j in range(4):
            for k in range(4):
                l3[j][k].append(x_train[i][j][k])
for j in range(4):
    for k in range(4):
        print(j,k,median(l2[j][k]))
for j in range(4):
    for k in range(4):
        print(j,k,median(l3[j][k]))



m = 0
for i in range(4):
    for j in range(4):
        m = max(m,l0[i][j])
for i in range(4):
    for j in range(4):
        m = max(m,l1[i][j])


l = Image.new('L',(4,4))
for i in range(4):
    for j in range(4):
        l.putpixel((j,i),255-int(200*l0[i][j]/m)) 
l = l.resize((400,400))
l.show()
l.save('inputavg0.jpg')

l = Image.new('L',(4,4))
for i in range(4):
    for j in range(4):
        l.putpixel((j,i),255-int(200*l1[i][j]/m)) 
l = l.resize((400,400))
l.show()
l.save('inputavg1.jpg')





m = 0
for i in range(4):
    for j in range(4):
        m = max(m,median(l2[i][j]))
for i in range(4):
    for j in range(4):
        m = max(m,median(l3[i][j]))


l = Image.new('L',(4,4))
for i in range(4):
    for j in range(4):
        l.putpixel((j,i),255-int(200*median(l2[i][j]/m))) 
l = l.resize((400,400))
l.show()
l.save('inputmed0.jpg')

l = Image.new('L',(4,4))
for i in range(4):
    for j in range(4):
        l.putpixel((j,i),255-int(200*median(l3[i][j]/m)) )
l = l.resize((400,400))
l.show()
l.save('inputmed1.jpg')