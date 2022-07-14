from PIL import Image
import os,glob,numpy as np
from sklearn.model_selection import  train_test_split
acc_dir = "./data/Test"
cg =["Fire","Non-Fire","Smoke"]
nb = len(cg)
img_w =64
img_h =64

pixels = img_w * img_h *3

X = []
Y = []

for idx ,cat in enumerate(cg):
    label =[0 for i in range(nb)]
    label[idx] = 1
    img_dir = acc_dir+ "/" + cat
    files = glob.glob(img_dir+"/*.jpg")

    for i,f in enumerate(files):
        img =Image.open(f)
        img =img.convert("RGB")
        img =img.resize((img_w,img_h))
        data = np.asarray(img)
        X.append(data)
        Y.append(label)
        if i %10==0:
            print(i,"/n",data)
X = np.array(X)
Y = np.array(Y)

X_train , X_test , y_train , y_test = train_test_split(X,Y)
xy = (X_train,X_test,y_train,y_test)

print('>>>저장중')
np.save("data.npy",xy)
print("ok",len(Y))