import sys
import re
path = 'natsume.txt'
bindata = open(path, "rb")
lines = bindata.readlines()
for line in lines:
    text = line.decode('Shift_JIS')
    text = re.split(r'\r',text)[0]
    text = re.split(r'底本',text)[0]
    text = text.replace('｜','')
    text = re.sub(r'《.+?》','',text)
    text = re.sub(r'［＃.+?］','',text)
    print(text)
file = open('data_bocchan.txt','a',encoding='utf-8').write(text)