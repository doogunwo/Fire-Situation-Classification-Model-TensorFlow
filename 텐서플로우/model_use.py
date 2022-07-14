import os,re,glob
import cv2
import numpy as np
import shutil
from keras.models import  load_model

#https://luminitworld.tistory.com/63 참고함
def zation(path):
    img =cv2.imread(path)
    return img/256

src =[]
name = []
test=[]

dir = 'test/'

for file in os.listdir(dir):
    if file.find('.jpg') != -1:
        src.append(dir+file)
        name.append(file)
        test.append(zation(dir+file))

test =np.array(test,dtype=object).astype(float)

model = load_model('model/model3.h5')
predict = model.predict(test)

print(predict.shape)
print("img: , Predict: [mask,nomask]")
for i in range(len(test)):
    print(name[i]+":, Predict :"+str(predict[i]))