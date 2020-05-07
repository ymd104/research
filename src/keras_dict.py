import collections
import sys
import MeCab
import numpy as np

f = open('data_4vs4_w.txt')
list_sentence = f.readlines()

mecab = MeCab.Tagger("-Owakati")

word2id = collections.defaultdict(lambda: len(word2id) )

l = []
label = []

for sentence_row in list_sentence:
    #sentence = "午前 2時 フミキリ に 望遠鏡 を 担いで いった 。 フミキリ に 望遠鏡 ラジオ"

    sentence = mecab.parse(sentence_row)
    

    def convert_word(sentence):
        return [word2id[word.lower()] for word in sentence.split()]

    #print("sentence    :", sentence)
    #print("id_sentence :", *convert_word(sentence) )
    #print(type(convert_word(sentence)))
    #print("dict        :", dict(word2id) )
    x = convert_word(sentence)
    label.append(x[0])
    del x[0]
    l.append(x)


arr_l = np.array(l)
arr_label = np.array(label)
print(arr_l)
print(arr_label)

print(len(word2id))

np.save('sentence.npy', arr_l)
np.save('label.npy', arr_label)


f.close()